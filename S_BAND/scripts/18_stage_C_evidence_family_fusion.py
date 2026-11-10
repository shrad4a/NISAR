"""
18_stage_C_evidence_family_fusion.py

QGPT-WM Version 2 - Stage C evidence-family fusion.

This script groups related measurements into four evidence families so
that HH/HV channels and their texture measurements do not each act as
fully independent votes.

Families
--------
1. Backscatter: HH-low, HV-low
2. Polarimetric contrast: HV/HH-ratio-low
3. Texture: HH-low-texture, HV-low-texture
4. Spatial: local candidate support

Important:
- September remains included.
- September Stage A status is FLAG, so HV-low, HV/HH ratio and
  HV-texture are retained as diagnostics but unavailable for scoring.
- covariance_gamma2 remains diagnostic only. It is not treated as a
  direct water indicator.
- Family scores are the mean of available feature indicators in that
  family. The final score is a weighted mean across available families.
- Three weight configurations are evaluated. Equal-family weighting is
  the primary configuration; the others are sensitivity experiments.
- Outputs are exploratory evidence classes, not validated water labels.

Class codes:
  0 = INVALID
  1 = NON_WATER_SUPPORTED
  2 = WATER_SUPPORTED
  3 = UNCERTAIN
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------------ CONFIG --------------------------------

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
STAGE_A_DIR = BASE_DIR / "acquisition_quality_output"
STAGE_B_DIR = BASE_DIR / "stage_B_candidate_output"
C1_DIR = BASE_DIR / "stage_C1_validation_output"
TEXTURE_DIR = BASE_DIR / "stage_C_texture_output"
C17_DIR = BASE_DIR / "stage_C_scoring_corrected_output"
OUTPUT_DIR = BASE_DIR / "stage_C_family_fusion_output"

DATES = ["20260819", "20260924", "20261018"]

# Family support thresholds. These are initial screening settings,
# not calibrated physical boundaries.
WATER_SUPPORT_THRESHOLD = 2.0 / 3.0
NON_WATER_SUPPORT_THRESHOLD = 0.25

# Weight configurations for sensitivity analysis.
WEIGHT_CONFIGS = {
    "equal": {
        "backscatter": 1.0,
        "polarimetric": 1.0,
        "texture": 1.0,
        "spatial": 1.0,
    },
    "spatial_emphasis": {
        "backscatter": 1.0,
        "polarimetric": 1.0,
        "texture": 1.0,
        "spatial": 1.5,
    },
    "texture_emphasis": {
        "backscatter": 1.0,
        "polarimetric": 1.0,
        "texture": 1.5,
        "spatial": 1.0,
    },
}

EPS = 1e-12


# ------------------------------ HELPERS -------------------------------

def load_array(folder: Path, stem: str):
    path = folder / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return np.asarray(np.load(path, mmap_mode="r"))


def get_acquisition_status(flags):
    if np.any(flags == 2):
        return "FLAG"
    if np.any(flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


def safe_stats(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if not v.size:
        return {}
    return {
        "p02": float(np.percentile(v, 2)),
        "median": float(np.percentile(v, 50)),
        "p98": float(np.percentile(v, 98)),
        "mean": float(np.mean(v)),
    }


def family_mean(indicators):
    """Mean of available binary indicators, NaN if none are available."""
    stack = np.stack(indicators, axis=0).astype(np.float32)
    valid = np.isfinite(stack)
    count = valid.sum(axis=0)
    total = np.nansum(stack, axis=0)
    result = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    good = count > 0
    result[good] = total[good] / count[good]
    return result


def weighted_family_fusion(families, weights):
    """Weighted mean over families available at each pixel."""
    numerator = np.zeros(next(iter(families.values())).shape, dtype=np.float32)
    denominator = np.zeros_like(numerator)

    for name, values in families.items():
        w = float(weights[name])
        valid = np.isfinite(values)
        numerator[valid] += values[valid] * w
        denominator[valid] += w

    score = np.full(numerator.shape, np.nan, dtype=np.float32)
    good = denominator > 0
    score[good] = numerator[good] / denominator[good]
    return score, denominator


def save_class_map(classes, date, status, config_name, path):
    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(classes, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "Invalid", "Non-water supported", "Water supported", "Uncertain"
    ])
    ax.set_title(
        f"{date}: family-fusion classes | {config_name} | Stage A: {status}"
    )
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -------------------------- LOAD THRESHOLDS ----------------------------

def load_thresholds():
    with open(C1_DIR / "stage_C1_report.json", "r", encoding="utf-8") as f:
        c1_report = json.load(f)
    with open(TEXTURE_DIR / "texture_reference_thresholds.json",
              "r", encoding="utf-8") as f:
        texture_report = json.load(f)

    thresholds = dict(c1_report["common_thresholds"])
    thresholds.update(texture_report["thresholds"])
    return thresholds


# -------------------------- PROCESS EACH DATE --------------------------

def process_date(date, thresholds):
    aoi = AOI_DIR / date
    bdir = STAGE_B_DIR / date
    c1dir = C1_DIR / date
    tdir = TEXTURE_DIR / date
    c17dir = C17_DIR / date

    hh = load_array(aoi, "HHHH").astype(np.float32)
    hv = load_array(aoi, "HVHV").astype(np.float32)
    mask = load_array(aoi, "mask")
    flags = load_array(STAGE_A_DIR, f"{date}_quality_flags")
    candidate = load_array(bdir, "water_candidate_mask").astype(bool)

    hh_db = load_array(c1dir, "HH_dB").astype(np.float32)
    hv_db = load_array(c1dir, "HV_dB").astype(np.float32)
    ratio_db = load_array(c1dir, "HV_HH_ratio_dB").astype(np.float32)
    gamma2 = load_array(c1dir, "covariance_gamma2").astype(np.float32)

    local_support = load_array(
        tdir, "local_candidate_fraction"
    ).astype(np.float32)
    hh_texture = load_array(
        tdir, "HH_texture_CV_5x5"
    ).astype(np.float32)
    hv_texture = load_array(
        tdir, "HV_texture_CV_5x5"
    ).astype(np.float32)

    previous_class = load_array(
        c17dir, "revised_classification"
    )

    all_arrays = [
        hh, hv, mask, flags, candidate, hh_db, hv_db, ratio_db,
        gamma2, local_support, hh_texture, hv_texture, previous_class
    ]
    if len({a.shape for a in all_arrays}) != 1:
        raise ValueError(f"{date}: input array shapes do not match.")

    status = get_acquisition_status(flags)
    base_valid = (
        (mask != 0) & (mask != 255)
        & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
    )
    hv_valid = base_valid & np.isfinite(hv) & (hv > 0)
    candidate = candidate & base_valid

    # Feature indicators are 1=positive support, 0=no positive support,
    # NaN=feature unavailable at that pixel.
    hh_low = np.full(hh.shape, np.nan, dtype=np.float32)
    hh_ok = base_valid & np.isfinite(hh_db)
    hh_low[hh_ok] = (
        hh_db[hh_ok] <= thresholds["HH_low_threshold_dB"]
    ).astype(np.float32)

    hv_low = np.full(hh.shape, np.nan, dtype=np.float32)
    hv_ok = hv_valid & np.isfinite(hv_db)
    if status == "ACCEPT":
        hv_low[hv_ok] = (
            hv_db[hv_ok] <= thresholds["HV_low_threshold_dB"]
        ).astype(np.float32)

    ratio_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ratio_ok = hv_valid & np.isfinite(ratio_db)
    if status == "ACCEPT":
        ratio_low[ratio_ok] = (
            ratio_db[ratio_ok]
            <= thresholds["HV_HH_ratio_low_threshold_dB"]
        ).astype(np.float32)

    hh_texture_low = np.full(hh.shape, np.nan, dtype=np.float32)
    hh_tex_ok = base_valid & np.isfinite(hh_texture)
    hh_texture_low[hh_tex_ok] = (
        hh_texture[hh_tex_ok]
        <= thresholds["HH_texture_low_CV_threshold"]
    ).astype(np.float32)

    hv_texture_low = np.full(hh.shape, np.nan, dtype=np.float32)
    hv_tex_ok = hv_valid & np.isfinite(hv_texture)
    if status == "ACCEPT":
        hv_texture_low[hv_tex_ok] = (
            hv_texture[hv_tex_ok]
            <= thresholds["HV_texture_low_CV_threshold"]
        ).astype(np.float32)

    spatial_support = np.full(hh.shape, np.nan, dtype=np.float32)
    spatial_ok = base_valid & np.isfinite(local_support)
    spatial_support[spatial_ok] = (
        local_support[spatial_ok] >= 0.50
    ).astype(np.float32)

    # Four evidence families. Correlated measurements are averaged within
    # a family before family-level fusion.
    families = {
        "backscatter": family_mean([hh_low, hv_low]),
        "polarimetric": ratio_low,
        "texture": family_mean([hh_texture_low, hv_texture_low]),
        "spatial": spatial_support,
    }

    family_available_count = np.zeros(hh.shape, dtype=np.uint8)
    family_positive_count = np.zeros(hh.shape, dtype=np.uint8)
    for values in families.values():
        available = np.isfinite(values)
        family_available_count[available] += 1
        family_positive_count[available & (values >= 0.5)] += 1

    out_date = OUTPUT_DIR / date
    out_date.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        fused_score, weight_sum = weighted_family_fusion(families, weights)

        classes = np.zeros(hh.shape, dtype=np.uint8)
        classes[base_valid] = 3

        # Only Stage B candidates are classified at this stage.
        eligible = (
            candidate
            & (family_available_count >= 3)
            & np.isfinite(fused_score)
        )
        water = eligible & (fused_score >= WATER_SUPPORT_THRESHOLD)
        nonwater = eligible & (
            fused_score <= NON_WATER_SUPPORT_THRESHOLD
        )

        classes[water] = 2
        classes[nonwater] = 1
        classes[candidate & ~eligible] = 3

        config_dir = out_date / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        np.save(config_dir / "family_fusion_score.npy", fused_score)
        np.save(config_dir / "family_available_count.npy",
                family_available_count)
        np.save(config_dir / "family_positive_count.npy",
                family_positive_count)
        np.save(config_dir / "family_fusion_class.npy", classes)

        for family_name, family_values in families.items():
            np.save(
                config_dir / f"{family_name}_family_support.npy",
                family_values
            )

        save_class_map(
            classes, date, status, config_name,
            config_dir / "family_fusion_classification_map.png"
        )

        # Per-date comparison with corrected pixel-wise score classes.
        comparable = base_valid & (previous_class > 0)
        changed = comparable & (classes != previous_class)

        summaries.append({
            "date": date,
            "stage_A_status": status,
            "weight_configuration": config_name,
            "weights": json.dumps(weights),
            "candidate_pixels": int(candidate.sum()),
            "eligible_candidate_pixels": int(eligible.sum()),
            "family_available_count_statistics": json.dumps(
                safe_stats(family_available_count[candidate])
            ),
            "family_positive_count_statistics": json.dumps(
                safe_stats(family_positive_count[candidate])
            ),
            "fusion_score_statistics": json.dumps(
                safe_stats(fused_score[candidate])
            ),
            "water_supported_pixels": int(np.sum(classes == 2)),
            "nonwater_supported_pixels": int(np.sum(classes == 1)),
            "uncertain_pixels": int(np.sum(classes == 3)),
            "changed_vs_script17_pixels": int(changed.sum()),
            "comparison_pixels": int(comparable.sum()),
            "polarimetric_family_available": status == "ACCEPT",
            "interpretation": (
                "Exploratory family-level evidence fusion. Correlated "
                "measurements are averaged within families. September's "
                "HV-derived evidence is unavailable for scoring."
            ),
        })

    # Save feature/family diagnostic image once per date.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (name, values) in zip(axes.flat, families.items()):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{name.capitalize()} family support")
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{date}: evidence-family support | Stage A: {status}"
    )
    fig.tight_layout()
    fig.savefig(out_date / "evidence_family_support.png", dpi=180)
    plt.close(fig)

    # Covariance gamma2 is intentionally diagnostic only.
    return summaries, {
        "date": date,
        "stage_A_status": status,
        "covariance_gamma2_statistics": safe_stats(gamma2),
        "feature_validity": {
            "HH": int(hh_ok.sum()),
            "HV": int(hv_ok.sum()),
            "HV_HH_ratio": int(ratio_ok.sum()),
            "HH_texture": int(hh_tex_ok.sum()),
            "HV_texture": int(hv_tex_ok.sum()),
            "spatial": int(spatial_ok.sum()),
        },
        "september_policy": (
            "If Stage A status is FLAG, HV, HV/HH ratio and HV texture "
            "are not made available to family fusion. They remain in "
            "the upstream diagnostic products."
        ),
    }


def main():
    required = [
        AOI_DIR, STAGE_A_DIR, STAGE_B_DIR,
        C1_DIR, TEXTURE_DIR, C17_DIR
    ]
    for folder in required:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    thresholds = load_thresholds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    diagnostics = {}

    for date in DATES:
        print(f"\nProcessing {date} ...")
        summaries, diag = process_date(date, thresholds)
        all_summaries.extend(summaries)
        diagnostics[date] = diag

        for result in summaries:
            print(
                f"  {result['weight_configuration']}: "
                f"water={result['water_supported_pixels']:,}, "
                f"non-water={result['nonwater_supported_pixels']:,}, "
                f"uncertain={result['uncertain_pixels']:,}"
            )

    pd.DataFrame(all_summaries).to_csv(
        OUTPUT_DIR / "stage_C_family_fusion_statistics.csv",
        index=False
    )

    report = {
        "stage": "QGPT-WM V2 - Stage C evidence-family fusion",
        "dates_processed": DATES,
        "families": {
            "backscatter": ["HH_low", "HV_low"],
            "polarimetric": ["HV_HH_ratio_low"],
            "texture": ["HH_low_texture", "HV_low_texture"],
            "spatial": ["local_candidate_support"],
        },
        "thresholds": thresholds,
        "class_thresholds": {
            "water_support": WATER_SUPPORT_THRESHOLD,
            "non_water_support": NON_WATER_SUPPORT_THRESHOLD,
        },
        "weight_configurations": WEIGHT_CONFIGS,
        "class_codes": {
            "0": "INVALID",
            "1": "NON_WATER_SUPPORTED",
            "2": "WATER_SUPPORTED",
            "3": "UNCERTAIN",
        },
        "september_policy": (
            "September remains included. HV backscatter, HV/HH ratio and "
            "HV texture are unavailable to scoring because Stage A flagged "
            "the acquisition. These measurements remain diagnostic upstream."
        ),
        "covariance_policy": (
            "Covariance gamma2 is retained as a diagnostic and is not used "
            "as a direct water indicator."
        ),
        "warning": (
            "Family-fusion classes are exploratory evidence classes. The "
            "weights and cutoffs are screening choices, not trained or "
            "independently validated parameters. Compare configurations "
            "and obtain independent reference labels before accuracy claims."
        ),
        "acquisition_diagnostics": diagnostics,
    }

    with open(OUTPUT_DIR / "stage_C_family_fusion_report.json",
              "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\nSTAGE C EVIDENCE-FAMILY FUSION COMPLETE")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Compare equal, spatial_emphasis and texture_emphasis maps.")
    print("Do not select a configuration based on visual appearance alone.")


if __name__ == "__main__":
    main()

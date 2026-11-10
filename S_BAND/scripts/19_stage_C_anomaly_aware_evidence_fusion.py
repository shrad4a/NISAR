"""
19_stage_C_anomaly_aware_evidence_fusion.py

QGPT-WM - Stage C anomaly-aware evidence-family fusion.

All three acquisitions remain in the processing pipeline, including the
September acquisition flagged by Stage A.

Design:
- HH and spatial evidence remain fully weighted.
- HV backscatter, HV/HH ratio, and HV texture are retained and scored.
- For Stage-A FLAG acquisitions, the reliability of HV-derived evidence is
  estimated from the acquisition-wide median HV/HH ratio relative to the
  ACCEPT reference acquisitions (August and October).
- Anomaly reliability is applied as a continuous weight, not as a hard
  exclusion. A small floor preserves a trace contribution while preventing
  a large anomalous response from dominating the score.
- The reliability estimate is a transparent screening heuristic, not a
  calibrated probability or sensor-quality measurement.

Classes:
  0 = INVALID
  1 = NON_WATER_SUPPORTED
  2 = WATER_SUPPORTED
  3 = UNCERTAIN

Outputs:
  stage_C_anomaly_aware_output/
    <date>/
      anomaly_diagnostics.json
      hv_anomaly_diagnostic.png
      evidence_family_support.png
      <configuration>/
        anomaly_aware_fusion_score.npy
        family_available_count.npy
        family_positive_count.npy
        family_effective_weight.npy
        family reliability/support arrays
        anomaly_aware_class.npy
        anomaly_aware_classification_map.png
    stage_C_anomaly_aware_statistics.csv
    stage_C_anomaly_aware_report.json

Run from the S_BAND working directory containing the existing Stage A/B/C
output folders created by the preceding scripts.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------------ CONFIG ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
STAGE_A_DIR = BASE_DIR / "acquisition_quality_output"
STAGE_B_DIR = BASE_DIR / "stage_B_candidate_output"
C1_DIR = BASE_DIR / "stage_C1_validation_output"
TEXTURE_DIR = BASE_DIR / "stage_C_texture_output"
C17_DIR = BASE_DIR / "stage_C_scoring_corrected_output"
OUTPUT_DIR = BASE_DIR / "stage_C_anomaly_aware_output"

DATES = ["20260819", "20260924", "20261018"]
REFERENCE_DATES = ["20260819", "20261018"]

WATER_SUPPORT_THRESHOLD = 2.0 / 3.0
NON_WATER_SUPPORT_THRESHOLD = 0.25

# Reliability heuristic for Stage-A FLAG acquisitions.
# 12 dB is the e-folding scale; 0.05 is a small non-zero reliability floor.
ANOMALY_SCALE_DB = 12.0
MIN_HV_RELIABILITY = 0.05

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


# ------------------------------ HELPERS --------------------------------

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
        "count": int(v.size),
        "p02": float(np.percentile(v, 2)),
        "median": float(np.percentile(v, 50)),
        "p98": float(np.percentile(v, 98)),
        "mean": float(np.mean(v)),
    }


def weighted_feature_mean(features, reliabilities):
    """Pixel-wise weighted mean; unavailable features do not enter."""
    shape = features[0].shape
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape, dtype=np.float32)
    for values, reliability in zip(features, reliabilities):
        valid = np.isfinite(values) & (reliability > 0)
        weight = float(reliability)
        numerator[valid] += values[valid] * weight
        denominator[valid] += weight
    result = np.full(shape, np.nan, dtype=np.float32)
    good = denominator > 0
    result[good] = numerator[good] / denominator[good]
    return result


def weighted_family_fusion(families, weights, family_reliability):
    shape = next(iter(families.values())).shape
    numerator = np.zeros(shape, dtype=np.float32)
    denominator = np.zeros(shape, dtype=np.float32)
    for name, values in families.items():
        effective_weight = float(weights[name]) * float(family_reliability[name])
        valid = np.isfinite(values) & (effective_weight > 0)
        numerator[valid] += values[valid] * effective_weight
        denominator[valid] += effective_weight
    score = np.full(shape, np.nan, dtype=np.float32)
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
        f"{date}: anomaly-aware fusion | {config_name} | Stage A: {status}"
    )
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_thresholds():
    with open(C1_DIR / "stage_C1_report.json", "r", encoding="utf-8") as f:
        c1_report = json.load(f)
    with open(TEXTURE_DIR / "texture_reference_thresholds.json",
              "r", encoding="utf-8") as f:
        texture_report = json.load(f)
    thresholds = dict(c1_report["common_thresholds"])
    thresholds.update(texture_report["thresholds"])
    return thresholds


def inspect_date(date):
    aoi = AOI_DIR / date
    c1dir = C1_DIR / date
    flags = load_array(STAGE_A_DIR, f"{date}_quality_flags")
    mask = load_array(aoi, "mask")
    hh = load_array(aoi, "HHHH").astype(np.float32)
    hv = load_array(aoi, "HVHV").astype(np.float32)
    ratio_db = load_array(c1dir, "HV_HH_ratio_dB").astype(np.float32)

    valid = (
        (mask != 0) & (mask != 255) & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
        & np.isfinite(hv) & (hv > 0)
        & np.isfinite(ratio_db)
    )
    status = get_acquisition_status(flags)
    if not np.any(valid):
        raise ValueError(f"{date}: no valid pixels for anomaly assessment.")
    return {
        "date": date,
        "status": status,
        "median_ratio_db": float(np.median(ratio_db[valid])),
        "valid_pixel_count": int(valid.sum()),
    }


def estimate_hv_reliability(date_diagnostics):
    refs = [
        d["median_ratio_db"] for d in date_diagnostics
        if d["date"] in REFERENCE_DATES and d["status"] == "ACCEPT"
    ]
    if not refs:
        raise RuntimeError(
            "No ACCEPT reference acquisitions available to estimate "
            "HV anomaly reliability."
        )
    reference_median = float(np.median(refs))
    results = {}
    for d in date_diagnostics:
        delta = d["median_ratio_db"] - reference_median
        if d["status"] == "ACCEPT":
            reliability = 1.0
        else:
            reliability = max(
                MIN_HV_RELIABILITY,
                float(np.exp(-abs(delta) / ANOMALY_SCALE_DB))
            )
        results[d["date"]] = {
            "stage_A_status": d["status"],
            "median_HV_HH_ratio_dB": d["median_ratio_db"],
            "reference_median_HV_HH_ratio_dB": reference_median,
            "deviation_from_reference_dB": float(delta),
            "HV_evidence_reliability": float(reliability),
            "reliability_method": (
                "ACCEPT acquisitions use 1.0. FLAG acquisitions use "
                "max(floor, exp(-abs(deviation_dB)/scale_dB))."
            ),
        }
    return results


# -------------------------- PROCESS EACH DATE --------------------------

def process_date(date, thresholds, anomaly_info):
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

    local_support = load_array(tdir, "local_candidate_fraction").astype(np.float32)
    hh_texture = load_array(tdir, "HH_texture_CV_5x5").astype(np.float32)
    hv_texture = load_array(tdir, "HV_texture_CV_5x5").astype(np.float32)
    previous_class = load_array(c17dir, "revised_classification")

    arrays = [
        hh, hv, mask, flags, candidate, hh_db, hv_db, ratio_db, gamma2,
        local_support, hh_texture, hv_texture, previous_class
    ]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"{date}: input array shapes do not match.")

    status = get_acquisition_status(flags)
    base_valid = (
        (mask != 0) & (mask != 255) & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
    )
    hv_valid = base_valid & np.isfinite(hv) & (hv > 0)
    candidate = candidate & base_valid

    hv_reliability = float(anomaly_info[date]["HV_evidence_reliability"])

    # Binary evidence indicators: 1=support, 0=no support, NaN=unavailable.
    hh_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = base_valid & np.isfinite(hh_db)
    hh_low[ok] = (
        hh_db[ok] <= thresholds["HH_low_threshold_dB"]
    ).astype(np.float32)

    hv_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = hv_valid & np.isfinite(hv_db)
    hv_low[ok] = (
        hv_db[ok] <= thresholds["HV_low_threshold_dB"]
    ).astype(np.float32)

    ratio_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = hv_valid & np.isfinite(ratio_db)
    ratio_low[ok] = (
        ratio_db[ok] <= thresholds["HV_HH_ratio_low_threshold_dB"]
    ).astype(np.float32)

    hh_texture_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = base_valid & np.isfinite(hh_texture)
    hh_texture_low[ok] = (
        hh_texture[ok] <= thresholds["HH_texture_low_CV_threshold"]
    ).astype(np.float32)

    hv_texture_low = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = hv_valid & np.isfinite(hv_texture)
    hv_texture_low[ok] = (
        hv_texture[ok] <= thresholds["HV_texture_low_CV_threshold"]
    ).astype(np.float32)

    spatial_support = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = base_valid & np.isfinite(local_support)
    spatial_support[ok] = (local_support[ok] >= 0.50).astype(np.float32)

    # HV-derived indicators remain available on September, but are downweighted.
    families = {
        "backscatter": weighted_feature_mean(
            [hh_low, hv_low], [1.0, hv_reliability]
        ),
        "polarimetric": ratio_low,
        "texture": weighted_feature_mean(
            [hh_texture_low, hv_texture_low], [1.0, hv_reliability]
        ),
        "spatial": spatial_support,
    }

    family_reliability = {
        "backscatter": 1.0,
        "polarimetric": hv_reliability,
        "texture": 1.0,
        "spatial": 1.0,
    }

    family_available_count = np.zeros(hh.shape, dtype=np.uint8)
    family_positive_count = np.zeros(hh.shape, dtype=np.uint8)
    for values in families.values():
        available = np.isfinite(values)
        family_available_count[available] += 1
        family_positive_count[available & (values >= 0.5)] += 1

    out_date = OUTPUT_DIR / date
    out_date.mkdir(parents=True, exist_ok=True)

    with open(out_date / "anomaly_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(anomaly_info[date], f, indent=2)

    # Plot September's HV-derived diagnostic ratio without hiding it.
    fig, ax = plt.subplots(figsize=(9, 7))
    ratio_display = np.where(hv_valid, ratio_db, np.nan)
    im = ax.imshow(ratio_display, cmap="coolwarm", vmin=-45, vmax=0)
    ax.set_title(
        f"{date}: HV/HH ratio diagnostic | Stage A: {status} | "
        f"HV reliability={hv_reliability:.3f}"
    )
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.colorbar(im, ax=ax, label="HV/HH ratio (dB)")
    fig.tight_layout()
    fig.savefig(out_date / "hv_anomaly_diagnostic.png", dpi=180)
    plt.close(fig)

    summaries = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        fused_score, effective_weight = weighted_family_fusion(
            families, weights, family_reliability
        )

        classes = np.zeros(hh.shape, dtype=np.uint8)
        classes[base_valid] = 3
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

        np.save(config_dir / "anomaly_aware_fusion_score.npy", fused_score)
        np.save(config_dir / "family_available_count.npy", family_available_count)
        np.save(config_dir / "family_positive_count.npy", family_positive_count)
        np.save(config_dir / "family_effective_weight.npy", effective_weight)
        np.save(config_dir / "anomaly_aware_class.npy", classes)

        for family_name, family_values in families.items():
            np.save(
                config_dir / f"{family_name}_family_support.npy",
                family_values
            )

        save_class_map(
            classes, date, status, config_name,
            config_dir / "anomaly_aware_classification_map.png"
        )

        comparable = base_valid & (previous_class > 0)
        changed = comparable & (classes != previous_class)

        summaries.append({
            "date": date,
            "stage_A_status": status,
            "weight_configuration": config_name,
            "weights": json.dumps(weights),
            "HV_evidence_reliability": hv_reliability,
            "HV_HH_ratio_deviation_dB": anomaly_info[date][
                "deviation_from_reference_dB"
            ],
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
            "effective_weight_statistics": json.dumps(
                safe_stats(effective_weight[candidate])
            ),
            "water_supported_pixels": int(np.sum(classes == 2)),
            "nonwater_supported_pixels": int(np.sum(classes == 1)),
            "uncertain_pixels": int(np.sum(classes == 3)),
            "changed_vs_script17_pixels": int(changed.sum()),
            "comparison_pixels": int(comparable.sum()),
            "interpretation": (
                "All dates and all valid feature measurements are retained. "
                "HV-derived evidence is continuously reliability-weighted; "
                "September remains included and flagged."
            ),
        })

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (name, values) in zip(axes.flat, families.items()):
        im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{name.capitalize()} family support")
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{date}: anomaly-aware evidence families | Stage A: {status}"
    )
    fig.tight_layout()
    fig.savefig(out_date / "evidence_family_support.png", dpi=180)
    plt.close(fig)

    return summaries, {
        "date": date,
        "stage_A_status": status,
        "HV_evidence_reliability": hv_reliability,
        "covariance_gamma2_statistics": safe_stats(gamma2),
        "feature_validity": {
            "HH": int((base_valid & np.isfinite(hh_db)).sum()),
            "HV": int((hv_valid & np.isfinite(hv_db)).sum()),
            "HV_HH_ratio": int((hv_valid & np.isfinite(ratio_db)).sum()),
            "HH_texture": int((base_valid & np.isfinite(hh_texture)).sum()),
            "HV_texture": int((hv_valid & np.isfinite(hv_texture)).sum()),
            "spatial": int((base_valid & np.isfinite(local_support)).sum()),
        },
    }


def main():
    required = [
        AOI_DIR, STAGE_A_DIR, STAGE_B_DIR, C1_DIR, TEXTURE_DIR, C17_DIR
    ]
    for folder in required:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    thresholds = load_thresholds()

    date_diagnostics = [inspect_date(date) for date in DATES]
    anomaly_info = estimate_hv_reliability(date_diagnostics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    diagnostics = {}

    for date in DATES:
        print(f"\nProcessing {date} ...")
        summaries, diag = process_date(date, thresholds, anomaly_info)
        all_summaries.extend(summaries)
        diagnostics[date] = diag
        for result in summaries:
            print(
                f"  {result['weight_configuration']}: "
                f"HV reliability={result['HV_evidence_reliability']:.3f}, "
                f"water={result['water_supported_pixels']:,}, "
                f"non-water={result['nonwater_supported_pixels']:,}, "
                f"uncertain={result['uncertain_pixels']:,}"
            )

    pd.DataFrame(all_summaries).to_csv(
        OUTPUT_DIR / "stage_C_anomaly_aware_statistics.csv", index=False
    )

    report = {
        "stage": "QGPT-WM Stage C anomaly-aware evidence fusion",
        "dates_processed": DATES,
        "reference_dates_for_HV_reliability": REFERENCE_DATES,
        "thresholds": thresholds,
        "class_thresholds": {
            "water_support": WATER_SUPPORT_THRESHOLD,
            "non_water_support": NON_WATER_SUPPORT_THRESHOLD,
        },
        "anomaly_reliability_parameters": {
            "scale_dB": ANOMALY_SCALE_DB,
            "minimum_reliability": MIN_HV_RELIABILITY,
            "method": (
                "For Stage-A FLAG acquisitions: max(minimum_reliability, "
                "exp(-abs(median_HV_HH_ratio_deviation_dB)/scale_dB)). "
                "ACCEPT acquisitions receive reliability 1.0."
            ),
            "caveat": (
                "This is a transparent heuristic for evidence weighting, "
                "not a calibrated probability, physical sensor-quality "
                "metric, or substitute for independent validation."
            ),
        },
        "acquisition_anomaly_diagnostics": anomaly_info,
        "weight_configurations": WEIGHT_CONFIGS,
        "class_codes": {
            "0": "INVALID",
            "1": "NON_WATER_SUPPORTED",
            "2": "WATER_SUPPORTED",
            "3": "UNCERTAIN",
        },
        "september_policy": (
            "September is fully retained. HH, HV, HV/HH ratio, HH texture, "
            "HV texture and spatial evidence are calculated. HV-derived "
            "evidence is downweighted using the measured acquisition-level "
            "HV/HH anomaly deviation and remains available in diagnostic "
            "outputs. The Stage-A FLAG is preserved."
        ),
        "covariance_policy": (
            "Covariance gamma2 is retained as a diagnostic and is not used "
            "as a direct water indicator."
        ),
        "warning": (
            "These are exploratory evidence classes. Reliability scale, "
            "floor and classification cutoffs are screening settings, not "
            "trained or independently validated parameters. Do not report "
            "classification accuracy without independent reference labels."
        ),
        "processing_diagnostics": diagnostics,
    }

    with open(
        OUTPUT_DIR / "stage_C_anomaly_aware_report.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\nSTAGE C ANOMALY-AWARE FUSION COMPLETE")
    print(f"Outputs: {OUTPUT_DIR}")
    print("September remains included; HV evidence is reliability-weighted.")
    print("Compare all three configurations. Do not choose by appearance alone.")


if __name__ == "__main__":
    main()

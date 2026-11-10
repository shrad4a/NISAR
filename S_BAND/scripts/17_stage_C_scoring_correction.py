"""
17_stage_C_scoring_correction.py

QGPT-WM Version 2 - Stage C scoring correction.

Purpose
-------
Corrects acquisition-level feature-count assumptions in
16_stage_C_texture_enhanced.py by computing evidence availability
per pixel.

Features:
  1. HH low backscatter
  2. HV low backscatter
  3. HV/HH low ratio
  4. Spatial candidate support
  5. HH low texture (5x5 local CV)
  6. HV low texture (5x5 local CV)

September remains included. Because its Stage A status is FLAG,
HV, HV/HH and HV-texture evidence are unavailable for scoring.
They are not counted as negative evidence.

Class codes:
  0 = INVALID
  1 = NON_WATER_SUPPORTED
  2 = WATER_SUPPORTED
  3 = UNCERTAIN

The output is exploratory evidence classification, not validated
water mapping. Thresholds are screening parameters.
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
OUTPUT_DIR = BASE_DIR / "stage_C_scoring_corrected_output"

DATES = ["20260819", "20260924", "20261018"]

# Retain the scoring policy used in the texture-enhanced experiment,
# but apply it to the number of features actually available per pixel.
WATER_SUPPORT_FRACTION = 2.0 / 3.0
NON_WATER_SUPPORT_FRACTION = 1.0 / 6.0
MIN_AVAILABLE_FEATURES = 3

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


def stats(values):
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


def save_class_map(classes, date, status, path):
    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(classes, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "Invalid", "Non-water supported", "Water supported", "Uncertain"
    ])
    ax.set_title(f"{date}: corrected per-pixel evidence classes ({status})")
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# -------------------------- PROCESS EACH DATE --------------------------

def process_date(date, thresholds):
    aoi = AOI_DIR / date
    bdir = STAGE_B_DIR / date
    tdir = TEXTURE_DIR / date
    c1dir = C1_DIR / date

    hh = load_array(aoi, "HHHH").astype(np.float32)
    hv = load_array(aoi, "HVHV").astype(np.float32)
    mask = load_array(aoi, "mask")
    flags = load_array(STAGE_A_DIR, f"{date}_quality_flags")
    candidate = load_array(bdir, "water_candidate_mask").astype(bool)

    hh_db = load_array(c1dir, "HH_dB").astype(np.float32)
    hv_db = load_array(c1dir, "HV_dB").astype(np.float32)
    ratio_db = load_array(c1dir, "HV_HH_ratio_dB").astype(np.float32)
    local_support = load_array(tdir, "local_candidate_fraction").astype(np.float32)
    hh_texture = load_array(tdir, "HH_texture_CV_5x5").astype(np.float32)
    hv_texture = load_array(tdir, "HV_texture_CV_5x5").astype(np.float32)
    old_c1_class = load_array(c1dir, "preliminary_class")
    old_texture_class = load_array(tdir, "texture_enhanced_class")

    arrays_to_check = [
        hh, hv, mask, flags, candidate, hh_db, hv_db, ratio_db,
        local_support, hh_texture, hv_texture, old_c1_class,
        old_texture_class
    ]
    if len({x.shape for x in arrays_to_check}) != 1:
        raise ValueError(f"{date}: input array shapes do not match.")

    status = get_acquisition_status(flags)

    base_valid = (
        (mask != 0) & (mask != 255)
        & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
    )
    hv_valid = base_valid & np.isfinite(hv) & (hv > 0)

    candidate = candidate & base_valid

    # Feature availability is computed independently for every pixel.
    available = np.zeros(hh.shape, dtype=np.uint8)
    positive = np.zeros(hh.shape, dtype=np.uint8)

    # Feature 1: HH low backscatter.
    hh_available = base_valid & np.isfinite(hh_db)
    hh_positive = hh_available & (
        hh_db <= thresholds["HH_low_threshold_dB"]
    )
    available[hh_available] += 1
    positive[hh_positive] += 1

    # Feature 2: spatial support.
    spatial_available = base_valid & np.isfinite(local_support)
    spatial_positive = spatial_available & (local_support >= 0.50)
    available[spatial_available] += 1
    positive[spatial_positive] += 1

    # Feature 3: HH low texture.
    hh_texture_available = base_valid & np.isfinite(hh_texture)
    hh_texture_positive = hh_texture_available & (
        hh_texture <= thresholds["HH_texture_low_CV_threshold"]
    )
    available[hh_texture_available] += 1
    positive[hh_texture_positive] += 1

    # HV-derived evidence is score-eligible only for ACCEPT acquisitions.
    hv_scoring_enabled = status == "ACCEPT"

    hv_available = hv_valid & np.isfinite(hv_db)
    ratio_available = hv_valid & np.isfinite(ratio_db)
    hv_texture_available = hv_valid & np.isfinite(hv_texture)

    if hv_scoring_enabled:
        hv_positive = hv_available & (
            hv_db <= thresholds["HV_low_threshold_dB"]
        )
        ratio_positive = ratio_available & (
            ratio_db <= thresholds["HV_HH_ratio_low_threshold_dB"]
        )
        hv_texture_positive = hv_texture_available & (
            hv_texture <= thresholds["HV_texture_low_CV_threshold"]
        )

        available[hv_available] += 1
        positive[hv_positive] += 1

        available[ratio_available] += 1
        positive[ratio_positive] += 1

        available[hv_texture_available] += 1
        positive[hv_texture_positive] += 1
    else:
        # Keep HV-derived diagnostics in outputs, but do not count them
        # as available scoring evidence for a flagged acquisition.
        hv_positive = np.zeros(hh.shape, dtype=bool)
        ratio_positive = np.zeros(hh.shape, dtype=bool)
        hv_texture_positive = np.zeros(hh.shape, dtype=bool)

    # Evidence fraction is defined only where at least one feature exists.
    evidence_fraction = np.full(hh.shape, np.nan, dtype=np.float32)
    has_evidence = available > 0
    evidence_fraction[has_evidence] = (
        positive[has_evidence] / available[has_evidence]
    )

    # Classification is applied to Stage B candidates only.
    eligible = candidate & (available >= MIN_AVAILABLE_FEATURES)

    classes = np.zeros(hh.shape, dtype=np.uint8)
    classes[base_valid] = 3  # uncertain by default

    water = eligible & (
        evidence_fraction >= WATER_SUPPORT_FRACTION
    )
    nonwater = eligible & (
        evidence_fraction <= NON_WATER_SUPPORT_FRACTION
    )

    classes[water] = 2
    classes[nonwater] = 1

    # Explicitly retain pixels with too few measurements as uncertain.
    insufficient = candidate & (available < MIN_AVAILABLE_FEATURES)
    classes[insufficient] = 3

    # Change maps compare corrected classes with both previous experiments.
    changed_vs_c1 = (
        base_valid
        & (old_c1_class > 0)
        & (classes != old_c1_class)
    )
    changed_vs_texture = (
        base_valid
        & (old_texture_class > 0)
        & (classes != old_texture_class)
    )

    out = OUTPUT_DIR / date
    out.mkdir(parents=True, exist_ok=True)

    outputs = {
        "positive_evidence_count": positive,
        "available_evidence_count": available,
        "evidence_confidence_fraction": evidence_fraction,
        "revised_classification": classes,
        "changed_vs_original_C1": changed_vs_c1.astype(np.uint8),
        "changed_vs_texture_classification": changed_vs_texture.astype(np.uint8),
        "HV_evidence_available_for_scoring": (
            hv_scoring_enabled & hv_valid
        ).astype(np.uint8),
    }

    for name, arr in outputs.items():
        np.save(out / f"{name}.npy", arr)

    save_class_map(
        classes, date, status,
        out / "revised_classification_map.png"
    )

    # Evidence availability map.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (available, "Available evidence count", 0, 6),
        (positive, "Positive evidence count", 0, 6),
        (evidence_fraction, "Positive / available fraction", 0, 1),
    ]
    for ax, (arr, title, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{date}: per-pixel evidence accounting ({status})")
    fig.tight_layout()
    fig.savefig(out / "evidence_availability_diagnostics.png", dpi=180)
    plt.close(fig)

    n_valid = int(base_valid.sum())
    n_candidates = int(candidate.sum())
    n_eligible = int(eligible.sum())
    comparison_count = int((base_valid & (old_c1_class > 0)).sum())

    return {
        "date": date,
        "stage_A_status": status,
        "HV_derived_scoring_enabled": hv_scoring_enabled,
        "valid_pixels": n_valid,
        "candidate_pixels": n_candidates,
        "eligible_candidate_pixels": n_eligible,
        "insufficient_evidence_candidate_pixels": int(insufficient.sum()),
        "available_evidence_count_statistics": stats(
            available[candidate]
        ),
        "positive_evidence_count_statistics": stats(
            positive[candidate]
        ),
        "evidence_fraction_statistics": stats(
            evidence_fraction[candidate]
        ),
        "water_supported_pixels": int(np.sum(classes == 2)),
        "nonwater_supported_pixels": int(np.sum(classes == 1)),
        "uncertain_pixels": int(np.sum(classes == 3)),
        "invalid_pixels": int(np.sum(classes == 0)),
        "original_C1_water_supported_pixels": int(np.sum(old_c1_class == 2)),
        "texture_class_water_supported_pixels": int(
            np.sum(old_texture_class == 2)
        ),
        "changed_vs_original_C1_pixels": int(changed_vs_c1.sum()),
        "changed_vs_texture_classification_pixels": int(
            changed_vs_texture.sum()
        ),
        "comparison_pixels": comparison_count,
        "interpretation": (
            "Per-pixel evidence availability correction. Flagged September "
            "HV-derived features are excluded from both numerator and "
            "denominator. Classes remain exploratory and unvalidated."
        ),
    }


def main():
    required = [
        AOI_DIR, STAGE_A_DIR, STAGE_B_DIR, C1_DIR, TEXTURE_DIR
    ]
    for folder in required:
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    with open(C1_DIR / "stage_C1_report.json", "r", encoding="utf-8") as f:
        c1_report = json.load(f)
    with open(TEXTURE_DIR / "texture_reference_thresholds.json",
              "r", encoding="utf-8") as f:
        texture_report = json.load(f)

    thresholds = dict(c1_report["common_thresholds"])
    thresholds.update(texture_report["thresholds"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "scoring_configuration.json",
              "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": thresholds,
            "water_support_fraction": WATER_SUPPORT_FRACTION,
            "non_water_support_fraction": NON_WATER_SUPPORT_FRACTION,
            "minimum_available_features": MIN_AVAILABLE_FEATURES,
            "feature_order": [
                "HH_low",
                "spatial_support",
                "HH_low_texture",
                "HV_low",
                "HV_HH_ratio_low",
                "HV_low_texture",
            ],
            "september_policy": (
                "September HV, HV/HH and HV texture remain diagnostic only. "
                "They are excluded from both positive and available feature "
                "counts because Stage A status is FLAG."
            ),
            "warning": (
                "Exploratory evidence classes only. Thresholds are not "
                "validated physical water boundaries."
            ),
        }, f, indent=2)

    summaries = []
    for date in DATES:
        print(f"\nProcessing {date} ...")
        result = process_date(date, thresholds)
        summaries.append(result)
        print(f"  Stage A: {result['stage_A_status']}")
        print(f"  Candidates: {result['candidate_pixels']:,}")
        print(f"  Eligible candidates: {result['eligible_candidate_pixels']:,}")
        print(f"  Water-supported: {result['water_supported_pixels']:,}")
        print(f"  Non-water-supported: {result['nonwater_supported_pixels']:,}")
        print(f"  Uncertain: {result['uncertain_pixels']:,}")
        print(f"  Changed vs original C1: "
              f"{result['changed_vs_original_C1_pixels']:,}")

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "stage_C_scoring_corrected_statistics.csv",
        index=False
    )

    report = {
        "stage": "QGPT-WM V2 - Stage C per-pixel scoring correction",
        "dates_processed": DATES,
        "thresholds": thresholds,
        "water_support_fraction": WATER_SUPPORT_FRACTION,
        "non_water_support_fraction": NON_WATER_SUPPORT_FRACTION,
        "minimum_available_features": MIN_AVAILABLE_FEATURES,
        "class_codes": {
            "0": "INVALID",
            "1": "NON_WATER_SUPPORTED",
            "2": "WATER_SUPPORTED",
            "3": "UNCERTAIN",
        },
        "september_policy": (
            "September remains included. Its HV, HV/HH and HV texture "
            "measurements are retained in diagnostic arrays but excluded "
            "from both positive and available evidence counts."
        ),
        "warning": (
            "This correction improves evidence accounting but does not "
            "establish classification accuracy. Independent reference "
            "validation is still required."
        ),
        "acquisitions": {r["date"]: r for r in summaries},
    }

    with open(OUTPUT_DIR / "stage_C_scoring_corrected_report.json",
              "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\nSTAGE C SCORING CORRECTION COMPLETE")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Inspect revised_classification_map.png and "
          "evidence_availability_diagnostics.png for each date.")
    print("These are exploratory evidence classes, not validated water maps.")


if __name__ == "__main__":
    main()

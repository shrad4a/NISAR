"""
12_stage_C_polarimetric_validation.py

Stage C: Multi-feature polarimetric validation for all three NISAR S-band dates.

September is INCLUDED in every computation and output. Its Stage A quality
status is carried as a warning/annotation; no date is skipped or silently
removed. Because September's HV-related measurements are anomalous, its
polarimetric evidence must be interpreted cautiously.

Inputs:
  aoi_outputs/<DATE>/HHHH.npy
  aoi_outputs/<DATE>/HVHV.npy
  aoi_outputs/<DATE>/HHHV.npy
  aoi_outputs/<DATE>/mask.npy
  aoi_outputs/<DATE>/inputDataExceptionMask.npy

  stage_B_candidate_output/<DATE>/water_candidate_mask.npy
  acquisition_quality_output/<DATE>_quality_flags.npy

Outputs:
  stage_C_validation_output/
      stage_C_feature_statistics.csv
      stage_C_report.json
      <DATE>/
          HH_dB.npy
          HV_dB.npy
          HV_HH_ratio_dB.npy
          covariance_gamma2.npy
          local_candidate_fraction.npy
          evidence_score.npy
          preliminary_class.npy
          feature_evidence.png
          preliminary_class_map.png

Class codes:
  0 = INVALID
  1 = NON_WATER_SUPPORTED
  2 = WATER_SUPPORTED
  3 = UNCERTAIN

This is an exploratory, unsupervised evidence layer, NOT a validated water
classifier. Thresholds are quantile-based and explicitly configurable.
No ground-truth accuracy is claimed. Stage D / validation must compare with
independent reference data.

Run:
  python 12_stage_C_polarimetric_validation.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- CONFIG -------------------------------

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
STAGE_A_DIR = BASE_DIR / "acquisition_quality_output"
STAGE_B_DIR = BASE_DIR / "stage_B_candidate_output"
OUTPUT_DIR = BASE_DIR / "stage_C_validation_output"

# All three dates are deliberately included.
DATES = ["20260819", "20260924", "20261018"]

# Exploratory within-acquisition quantile cutoffs.
# These are not physical water thresholds or validated classifier parameters.
LOW_BACKSCATTER_QUANTILE = 0.50
LOW_CROSSPOL_QUANTILE = 0.50
LOW_RATIO_QUANTILE = 0.50

# 3x3 local neighbourhood candidate support.
NEIGHBORHOOD_SIZE = 3
MIN_LOCAL_CANDIDATE_FRACTION = 0.50

# Evidence score: 1 point each for HH candidate, low HV, low HV/HH ratio,
# and local candidate support. Covariance gamma^2 is retained as a diagnostic
# and validity check, not treated as a direct water indicator.
WATER_SUPPORT_MIN_SCORE = 3
NON_WATER_MAX_SCORE = 1

EPS = 1e-12


# ------------------------------- HELPERS ------------------------------

def load_array(folder: Path, stem: str, dtype=None):
    path = folder / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def to_db(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    good = np.isfinite(values) & (values > 0)
    out[good] = (10.0 * np.log10(values[good])).astype(np.float32)
    return out


def percentile_threshold(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q * 100.0))


def local_fraction(binary_mask, valid_mask, size=3):
    """Compute valid-neighbour candidate fraction using integral images."""
    binary = (np.asarray(binary_mask, dtype=np.float64) > 0).astype(np.float64)
    valid = np.asarray(valid_mask, dtype=np.float64)
    pad = size // 2

    def box_sum(a):
        p = np.pad(a, ((pad, pad), (pad, pad)), mode="constant")
        integral = np.pad(p, ((1, 0), (1, 0)), mode="constant")
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        return (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )

    candidate_count = box_sum(binary * valid)
    valid_count = box_sum(valid)
    result = np.full(binary.shape, np.nan, dtype=np.float32)
    good = valid_count > 0
    result[good] = (candidate_count[good] / valid_count[good]).astype(np.float32)
    return result


def finite_stats(values):
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


def quality_status(date, stage_a_flags):
    # Stage A pixel flags: 0 invalid, 1 accepted, 2 flagged.
    if np.any(stage_a_flags == 2):
        return "FLAG"
    if np.any(stage_a_flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


# ----------------------------- PROCESS DATE ---------------------------

def process_date(date):
    aoi = AOI_DIR / date
    stage_b = STAGE_B_DIR / date

    hh = load_array(aoi, "HHHH", np.float32)
    hv = load_array(aoi, "HVHV", np.float32)
    hhv = load_array(aoi, "HHHV")
    mask = load_array(aoi, "mask")
    exception = load_array(aoi, "inputDataExceptionMask")
    candidate = load_array(stage_b, "water_candidate_mask").astype(bool)
    stage_a_flags = load_array(STAGE_A_DIR, f"{date}_quality_flags")

    shapes = [x.shape for x in (hh, hv, hhv, mask, exception, candidate, stage_a_flags)]
    if len(set(shapes)) != 1:
        raise ValueError(f"{date}: input shapes differ: {shapes}")

    valid = (
        (mask != 0) & (mask != 255)
        & np.isfinite(hh) & (hh > 0)
        & np.isfinite(hv) & (hv > 0)
        & np.isfinite(hhv.real) & np.isfinite(hhv.imag)
        & (stage_a_flags != 0)
    )

    hh_db = to_db(hh)
    hv_db = to_db(hv)

    ratio_db = np.full(hh.shape, np.nan, dtype=np.float32)
    ratio_valid = valid & (hh > 0) & (hv > 0)
    ratio_db[ratio_valid] = (
        10.0 * np.log10(
            np.maximum(hv[ratio_valid], EPS) / np.maximum(hh[ratio_valid], EPS)
        )
    ).astype(np.float32)

    denom = hh.astype(np.float64) * hv.astype(np.float64)
    gamma2 = np.full(hh.shape, np.nan, dtype=np.float32)
    cov_valid = valid & np.isfinite(denom) & (denom > 0)
    gamma2[cov_valid] = (
        np.abs(hhv[cov_valid].astype(np.complex128)) ** 2 / denom[cov_valid]
    ).astype(np.float32)

    # Covariance consistency is a quality diagnostic, not a water score.
    gamma_ok = cov_valid & (gamma2 >= 0) & (gamma2 <= 1.000001)

    # Restrict candidate pixels to valid Stage A pixels but retain September.
    candidate = candidate & valid
    local_support = local_fraction(candidate, valid, NEIGHBORHOOD_SIZE)

    # Per-date empirical thresholds. This permits each date to be analysed,
    # including September, without silently substituting another date's data.
    hh_threshold = percentile_threshold(hh_db[valid], LOW_BACKSCATTER_QUANTILE)
    hv_threshold = percentile_threshold(hv_db[valid], LOW_CROSSPOL_QUANTILE)
    ratio_threshold = percentile_threshold(ratio_db[ratio_valid], LOW_RATIO_QUANTILE)

    hh_low = valid & (hh_db <= hh_threshold)
    hv_low = valid & (hv_db <= hv_threshold)
    ratio_low = ratio_valid & (ratio_db <= ratio_threshold)
    spatial_supported = valid & (local_support >= MIN_LOCAL_CANDIDATE_FRACTION)

    # Score only candidate pixels. Each criterion contributes one point.
    score = np.zeros(hh.shape, dtype=np.uint8)
    score[candidate & hh_low] += 1
    score[candidate & hv_low] += 1
    score[candidate & ratio_low] += 1
    score[candidate & spatial_supported] += 1

    # Class codes: 0 invalid, 1 non-water supported, 2 water supported,
    # 3 uncertain. Non-candidates are not automatically called non-water;
    # they remain uncertain unless the evidence rule below supports otherwise.
    classes = np.zeros(hh.shape, dtype=np.uint8)
    classes[valid] = 3
    classes[candidate & (score >= WATER_SUPPORT_MIN_SCORE)] = 2
    classes[candidate & (score <= NON_WATER_MAX_SCORE)] = 1

    # Candidate pixels with intermediate score remain uncertain.
    # Non-candidate pixels remain uncertain at this stage because Stage C
    # is specifically validating Stage B candidates, not final mapping.

    status = quality_status(date, stage_a_flags)
    out_dir = OUTPUT_DIR / date
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "HH_dB.npy", hh_db)
    np.save(out_dir / "HV_dB.npy", hv_db)
    np.save(out_dir / "HV_HH_ratio_dB.npy", ratio_db)
    np.save(out_dir / "covariance_gamma2.npy", gamma2)
    np.save(out_dir / "local_candidate_fraction.npy", local_support)
    np.save(out_dir / "evidence_score.npy", score)
    np.save(out_dir / "preliminary_class.npy", classes)

    # Feature evidence visualization.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (hh_db, "HH Gamma0 (dB)", "gray", -30, 5),
        (hv_db, "HV Gamma0 (dB)", "gray", -45, 0),
        (ratio_db, "HV/HH ratio (dB)", "viridis", -50, 5),
        (gamma2, "Covariance gamma²", "viridis", 0, 1),
        (local_support, "Local candidate fraction", "viridis", 0, 1),
        (score, "Candidate evidence score (0-4)", "viridis", 0, 4),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{date} | Stage C feature evidence | Stage A status: {status}",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(out_dir / "feature_evidence.png", dpi=180)
    plt.close(fig)

    # Class map: 0 black invalid, 1 non-water supported, 2 water supported,
    # 3 uncertain. Discrete labels are shown with a fixed color scale.
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(classes, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Invalid", "Non-water supported", "Water supported", "Uncertain"])
    plt.title(f"{date}: preliminary Stage C evidence classes ({status})")
    plt.xlabel("AOI column")
    plt.ylabel("AOI row")
    plt.tight_layout()
    plt.savefig(out_dir / "preliminary_class_map.png", dpi=180)
    plt.close()

    n_valid = int(valid.sum())
    n_cand = int(candidate.sum())
    summary = {
        "date": date,
        "stage_A_status": status,
        "september_included": True,
        "valid_pixels": n_valid,
        "candidate_pixels": n_cand,
        "candidate_fraction_of_valid": float(n_cand / n_valid) if n_valid else None,
        "HH_low_threshold_dB": hh_threshold,
        "HV_low_threshold_dB": hv_threshold,
        "HV_HH_ratio_low_threshold_dB": ratio_threshold,
        "local_candidate_support_threshold": MIN_LOCAL_CANDIDATE_FRACTION,
        "covariance_gamma2_median": finite_stats(gamma2[gamma_ok]).get("median"),
        "covariance_violation_fraction": (
            float(np.mean(cov_valid & (gamma2 > 1.000001)))
            if cov_valid.any() else None
        ),
        "exception_code_64_fraction": float(np.mean(exception == 64)),
        "candidate_score_0_1_count": int(np.sum(candidate & (score <= 1))),
        "candidate_score_2_count": int(np.sum(candidate & (score == 2))),
        "candidate_score_3_4_count": int(np.sum(candidate & (score >= 3))),
        "class_0_invalid_count": int(np.sum(classes == 0)),
        "class_1_nonwater_supported_count": int(np.sum(classes == 1)),
        "class_2_water_supported_count": int(np.sum(classes == 2)),
        "class_3_uncertain_count": int(np.sum(classes == 3)),
        "HH_dB_statistics": finite_stats(hh_db[valid]),
        "HV_dB_statistics": finite_stats(hv_db[valid]),
        "HV_HH_ratio_dB_statistics": finite_stats(ratio_db[ratio_valid]),
        "interpretation": (
            "Exploratory evidence classes only. Not validated water labels. "
            "September is processed and retained, but its known HV/ratio anomaly "
            "must be considered when interpreting its polarimetric evidence."
        ),
    }
    return summary


# -------------------------------- MAIN ---------------------------------

def main():
    for folder in (AOI_DIR, STAGE_A_DIR, STAGE_B_DIR):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    for date in DATES:
        print(f"\nProcessing {date} (included)...")
        summary = process_date(date)
        summaries.append(summary)
        print(f"  Stage A status: {summary['stage_A_status']}")
        print(f"  Candidate pixels: {summary['candidate_pixels']:,}")
        print(f"  Water-supported candidates: {summary['class_2_water_supported_count']:,}")
        print(f"  Non-water-supported candidates: {summary['class_1_nonwater_supported_count']:,}")
        print(f"  Uncertain pixels: {summary['class_3_uncertain_count']:,}")

    df_rows = []
    for item in summaries:
        df_rows.append({
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in item.items()
        })

    pd.DataFrame(df_rows).to_csv(
        OUTPUT_DIR / "stage_C_feature_statistics.csv", index=False
    )

    report = {
        "stage": "C - Multi-feature polarimetric validation",
        "dates_processed": DATES,
        "all_dates_included": True,
        "september_policy": (
            "September is processed through every feature and output. Its known "
            "HV/HH anomaly is disclosed as a reliability caveat; it is not omitted."
        ),
        "class_codes": {
            "0": "INVALID",
            "1": "NON_WATER_SUPPORTED",
            "2": "WATER_SUPPORTED",
            "3": "UNCERTAIN",
        },
        "method_warning": (
            "Exploratory unsupervised evidence scoring using within-date quantiles. "
            "Not a validated classifier; thresholds require independent reference "
            "data and sensitivity analysis."
        ),
        "acquisitions": {x["date"]: x for x in summaries},
    }
    with open(OUTPUT_DIR / "stage_C_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\n========== STAGE C COMPLETE ==========")
    print(f"All dates processed: {', '.join(DATES)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Class codes: 0=INVALID, 1=NON_WATER_SUPPORTED, "
          "2=WATER_SUPPORTED, 3=UNCERTAIN")
    print("These are preliminary evidence classes, not validated water labels.")


if __name__ == "__main__":
    main()

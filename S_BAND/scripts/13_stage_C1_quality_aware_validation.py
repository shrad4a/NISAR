"""
13_stage_C1_quality_aware_validation.py

Stage C.1: Quality-aware, reference-calibrated polarimetric evidence scoring.

All three dates are processed:
  20260819, 20260924, 20261018

Reference acquisitions:
  20260819 and 20261018

September is NOT dropped. Its HH and spatial evidence are retained. Its HV
and HV/HH evidence are computed and saved, but are marked unreliable for
positive/negative scoring because Stage A flagged its acquisition-wide
polarimetric anomaly.

IMPORTANT SCIENTIFIC DESIGN:
- A common threshold is estimated from the pooled reference acquisitions.
- Quantile thresholds are screening parameters, not physical water boundaries.
- HV and HV/HH values for September remain in the outputs and diagnostics.
- September HV/ratio do not contribute to its evidence score until the anomaly
  is independently resolved.
- HHHV covariance consistency is diagnostic, not a direct water indicator.
- The score is an exploratory evidence score, not a trained/validated classifier.
- No final water accuracy is claimed.

Inputs:
  aoi_outputs/<DATE>/{HHHH,HVHV,HHHV,mask,inputDataExceptionMask}.npy
  stage_B_candidate_output/<DATE>/water_candidate_mask.npy
  acquisition_quality_output/<DATE>_quality_flags.npy

Run:
  python 13_stage_C1_quality_aware_validation.py
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
OUTPUT_DIR = BASE_DIR / "stage_C1_validation_output"

DATES = ["20260819", "20260924", "20261018"]
REFERENCE_DATES = ["20260819", "20261018"]

# Common reference thresholds: pooled reference-date percentiles.
# These are transparent initial screening cutoffs, not validated physical
# water thresholds. Sensitivity analysis and ground truth are still required.
HH_LOW_QUANTILE = 0.50
HV_LOW_QUANTILE = 0.50
RATIO_LOW_QUANTILE = 0.50

NEIGHBORHOOD_SIZE = 3
MIN_LOCAL_CANDIDATE_FRACTION = 0.50

# Scoring:
# HH low + spatial support + HV low + ratio low = up to 4 evidence points.
# For September, only HH and spatial support are score-eligible because the
# acquisition-level HV/ratio anomaly remains unresolved.
WATER_SUPPORT_MIN_SCORE = 3
NON_WATER_MAX_SCORE = 1

EPS = 1e-12


# ------------------------------ HELPERS -------------------------------

def load_array(folder: Path, stem: str):
    path = folder / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return np.asarray(np.load(path, mmap_mode="r"))


def to_db(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    good = np.isfinite(a) & (a > 0)
    out[good] = (10.0 * np.log10(a[good])).astype(np.float32)
    return out


def quantile(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.percentile(values, q * 100.0))


def local_fraction(binary_mask, valid_mask, size=3):
    binary = (np.asarray(binary_mask, dtype=np.float64) > 0).astype(np.float64)
    valid = np.asarray(valid_mask, dtype=np.float64)
    pad = size // 2

    def box_sum(a):
        p = np.pad(a, ((pad, pad), (pad, pad)), mode="constant")
        integ = np.pad(p, ((1, 0), (1, 0)), mode="constant")
        integ = integ.cumsum(axis=0).cumsum(axis=1)
        return (
            integ[size:, size:] - integ[:-size, size:]
            - integ[size:, :-size] + integ[:-size, :-size]
        )

    n_candidate = box_sum(binary * valid)
    n_valid = box_sum(valid)
    result = np.full(binary.shape, np.nan, dtype=np.float32)
    ok = n_valid > 0
    result[ok] = (n_candidate[ok] / n_valid[ok]).astype(np.float32)
    return result


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


def get_acquisition_status(flags):
    if np.any(flags == 2):
        return "FLAG"
    if np.any(flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


# ---------------------- BUILD COMMON REFERENCE -----------------------

def derive_reference_thresholds():
    pooled = {"hh": [], "hv": [], "ratio": []}

    for date in REFERENCE_DATES:
        folder = AOI_DIR / date
        hh = load_array(folder, "HHHH").astype(np.float32)
        hv = load_array(folder, "HVHV").astype(np.float32)
        mask = load_array(folder, "mask")

        valid = (
            (mask != 0) & (mask != 255)
            & np.isfinite(hh) & (hh > 0)
            & np.isfinite(hv) & (hv > 0)
        )

        hh_db = to_db(hh)
        hv_db = to_db(hv)
        ratio_db = np.full(hh.shape, np.nan, dtype=np.float32)
        ratio_db[valid] = (
            10.0 * np.log10(
                np.maximum(hv[valid], EPS) / np.maximum(hh[valid], EPS)
            )
        ).astype(np.float32)

        pooled["hh"].append(hh_db[valid])
        pooled["hv"].append(hv_db[valid])
        pooled["ratio"].append(ratio_db[valid])

    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    thresholds = {
        "HH_low_threshold_dB": quantile(pooled["hh"], HH_LOW_QUANTILE),
        "HV_low_threshold_dB": quantile(pooled["hv"], HV_LOW_QUANTILE),
        "HV_HH_ratio_low_threshold_dB": quantile(pooled["ratio"], RATIO_LOW_QUANTILE),
    }
    return thresholds, pooled


# -------------------------- PROCESS EACH DATE -------------------------

def process_date(date, thresholds):
    folder = AOI_DIR / date
    b_folder = STAGE_B_DIR / date

    hh = load_array(folder, "HHHH").astype(np.float32)
    hv = load_array(folder, "HVHV").astype(np.float32)
    hhv = load_array(folder, "HHHV")
    mask = load_array(folder, "mask")
    exception = load_array(folder, "inputDataExceptionMask")
    candidate = load_array(b_folder, "water_candidate_mask").astype(bool)
    stage_a_flags = load_array(STAGE_A_DIR, f"{date}_quality_flags")

    shapes = {x.shape for x in (hh, hv, hhv, mask, exception, candidate, stage_a_flags)}
    if len(shapes) != 1:
        raise ValueError(f"{date}: input array shapes do not match.")

    stage_status = get_acquisition_status(stage_a_flags)

    # Keep all dates in the outputs. Validity of HH and HV is recorded
    # separately so a bad HV measurement does not erase usable HH evidence.
    base_valid = (
        (mask != 0) & (mask != 255)
        & np.isfinite(hh) & (hh > 0)
        & (stage_a_flags != 0)
    )
    hv_valid = base_valid & np.isfinite(hv) & (hv > 0)
    cov_valid = hv_valid & np.isfinite(hhv.real) & np.isfinite(hhv.imag)

    hh_db = to_db(hh)
    hv_db = to_db(hv)

    ratio_db = np.full(hh.shape, np.nan, dtype=np.float32)
    ratio_db[hv_valid] = (
        10.0 * np.log10(
            np.maximum(hv[hv_valid], EPS) / np.maximum(hh[hv_valid], EPS)
        )
    ).astype(np.float32)

    denom = hh.astype(np.float64) * hv.astype(np.float64)
    gamma2 = np.full(hh.shape, np.nan, dtype=np.float32)
    good_cov = cov_valid & np.isfinite(denom) & (denom > 0)
    gamma2[good_cov] = (
        np.abs(hhv[good_cov].astype(np.complex128)) ** 2 / denom[good_cov]
    ).astype(np.float32)

    covariance_violation = good_cov & (gamma2 > 1.000001)
    gamma2_valid = good_cov & (gamma2 >= 0) & (gamma2 <= 1.000001)

    candidate = candidate & base_valid
    local_support = local_fraction(candidate, base_valid, NEIGHBORHOOD_SIZE)

    # Common thresholds derived once from August+October, then applied to all dates.
    hh_low = base_valid & (hh_db <= thresholds["HH_low_threshold_dB"])
    hv_low = hv_valid & (hv_db <= thresholds["HV_low_threshold_dB"])
    ratio_low = hv_valid & (ratio_db <= thresholds["HV_HH_ratio_low_threshold_dB"])
    spatial_support = base_valid & (
        local_support >= MIN_LOCAL_CANDIDATE_FRACTION
    )

    # Per-pixel evidence availability. September HV and ratio are retained
    # in arrays/statistics but not used in its score because Stage A flagged
    # the acquisition-wide anomaly. No date is removed.
    hv_evidence_reliable = stage_status == "ACCEPT"
    ratio_evidence_reliable = stage_status == "ACCEPT"

    score = np.zeros(hh.shape, dtype=np.uint8)
    score[candidate & hh_low] += 1
    score[candidate & spatial_support] += 1

    if hv_evidence_reliable:
        score[candidate & hv_low] += 1
    if ratio_evidence_reliable:
        score[candidate & ratio_low] += 1

    # A date with unreliable HV/ratio has only two available score dimensions.
    # Do not call it NON-WATER merely because it cannot reach a four-feature
    # score. For September, score=2 is the maximum; all such candidates remain
    # UNCERTAIN rather than being mislabeled.
    classes = np.zeros(hh.shape, dtype=np.uint8)
    classes[base_valid] = 3

    if hv_evidence_reliable and ratio_evidence_reliable:
        classes[candidate & (score >= WATER_SUPPORT_MIN_SCORE)] = 2
        classes[candidate & (score <= NON_WATER_MAX_SCORE)] = 1
    else:
        # With only HH + spatial evidence available, both features agreeing
        # yields "water-supported, low confidence"; anything else is uncertain.
        classes[candidate & (score == 2)] = 2
        # No non-water label from an incomplete feature set.

    # Diagnostic reliability map: 1=usable, 0=unreliable/unavailable.
    reliability = np.zeros(hh.shape, dtype=np.uint8)
    reliability[base_valid] = 1
    if not hv_evidence_reliable:
        reliability[base_valid] = 2  # HH usable; HV/ratio unreliable for scoring

    out = OUTPUT_DIR / date
    out.mkdir(parents=True, exist_ok=True)

    arrays = {
        "HH_dB": hh_db,
        "HV_dB": hv_db,
        "HV_HH_ratio_dB": ratio_db,
        "covariance_gamma2": gamma2,
        "local_candidate_fraction": local_support,
        "evidence_score": score,
        "preliminary_class": classes,
        "polarimetric_reliability_flag": reliability,
    }
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)

    # Multi-panel feature evidence.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (hh_db, "HH Gamma0 (dB)", "gray", -30, 5),
        (hv_db, "HV Gamma0 (dB)", "gray", -60 if date == "20260924" else -45, 0),
        (ratio_db, "HV/HH ratio (dB)", "viridis", -55 if date == "20260924" else -20, 5),
        (gamma2, "Covariance gamma²", "viridis", 0, 1),
        (local_support, "Local candidate fraction", "viridis", 0, 1),
        (score, "Quality-aware evidence score", "viridis", 0, 4),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{date} | Stage C.1 | Stage A: {stage_status} | "
        f"HV/ratio scoring reliable: {hv_evidence_reliable}"
    )
    fig.tight_layout()
    fig.savefig(out / "feature_evidence.png", dpi=180)
    plt.close(fig)

    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    plt.figure(figsize=(9, 7))
    im = plt.imshow(classes, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "Invalid", "Non-water supported", "Water supported", "Uncertain"
    ])
    plt.title(f"{date}: quality-aware preliminary evidence classes ({stage_status})")
    plt.xlabel("AOI column")
    plt.ylabel("AOI row")
    plt.tight_layout()
    plt.savefig(out / "classification_map.png", dpi=180)
    plt.close()

    n_valid = int(base_valid.sum())
    n_candidate = int(candidate.sum())

    return {
        "date": date,
        "stage_A_status": stage_status,
        "included": True,
        "hv_ratio_evidence_used_in_score": bool(hv_evidence_reliable),
        "valid_HH_pixels": n_valid,
        "valid_HV_pixels": int(hv_valid.sum()),
        "candidate_pixels": n_candidate,
        "candidate_fraction": float(n_candidate / n_valid) if n_valid else None,
        "common_thresholds": thresholds,
        "HH_low_candidate_count": int(np.sum(candidate & hh_low)),
        "HV_low_candidate_count": int(np.sum(candidate & hv_low)),
        "HV_ratio_low_candidate_count": int(np.sum(candidate & ratio_low)),
        "spatially_supported_candidate_count": int(np.sum(candidate & spatial_support)),
        "covariance_gamma2_median": stats(gamma2[gamma2_valid]).get("median"),
        "covariance_violation_fraction": (
            float(covariance_violation.sum() / good_cov.sum())
            if good_cov.any() else None
        ),
        "exception_code_64_fraction": float(np.mean(exception == 64)),
        "score_0_count": int(np.sum(candidate & (score == 0))),
        "score_1_count": int(np.sum(candidate & (score == 1))),
        "score_2_count": int(np.sum(candidate & (score == 2))),
        "score_3_count": int(np.sum(candidate & (score == 3))),
        "score_4_count": int(np.sum(candidate & (score == 4))),
        "class_invalid_count": int(np.sum(classes == 0)),
        "class_nonwater_supported_count": int(np.sum(classes == 1)),
        "class_water_supported_count": int(np.sum(classes == 2)),
        "class_uncertain_count": int(np.sum(classes == 3)),
        "HH_dB_statistics": stats(hh_db[base_valid]),
        "HV_dB_statistics": stats(hv_db[hv_valid]),
        "HV_HH_ratio_dB_statistics": stats(ratio_db[hv_valid]),
        "interpretation": (
            "Exploratory evidence only, not validated water labels. September "
            "is included; its HV and HV/HH evidence is retained in diagnostics "
            "but excluded from its score pending resolution of the known anomaly."
        ),
    }


# -------------------------------- MAIN ---------------------------------

def main():
    for folder in (AOI_DIR, STAGE_A_DIR, STAGE_B_DIR):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds, pooled_reference = derive_reference_thresholds()

    with open(OUTPUT_DIR / "reference_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({
            "reference_dates": REFERENCE_DATES,
            "thresholds": thresholds,
            "quantiles": {
                "HH": HH_LOW_QUANTILE,
                "HV": HV_LOW_QUANTILE,
                "HV_HH_ratio": RATIO_LOW_QUANTILE,
            },
            "note": "Initial pooled-reference screening thresholds, not validated physical water boundaries."
        }, f, indent=2)

    print("\nCOMMON REFERENCE THRESHOLDS")
    for key, value in thresholds.items():
        print(f"  {key}: {value:.3f} dB")

    summaries = []
    for date in DATES:
        print(f"\nProcessing {date} ...")
        summary = process_date(date, thresholds)
        summaries.append(summary)
        print(f"  Stage A: {summary['stage_A_status']}")
        print(f"  HV/ratio used in score: {summary['hv_ratio_evidence_used_in_score']}")
        print(f"  Candidates: {summary['candidate_pixels']:,}")
        print(f"  Water-supported: {summary['class_water_supported_count']:,}")
        print(f"  Non-water-supported: {summary['class_nonwater_supported_count']:,}")
        print(f"  Uncertain: {summary['class_uncertain_count']:,}")

    pd.DataFrame([
        {k: json.dumps(v) if isinstance(v, dict) else v for k, v in s.items()}
        for s in summaries
    ]).to_csv(OUTPUT_DIR / "stage_C1_statistics.csv", index=False)

    report = {
        "stage": "C.1 - Quality-aware polarimetric validation",
        "dates_processed": DATES,
        "all_dates_included": True,
        "reference_dates": REFERENCE_DATES,
        "common_thresholds": thresholds,
        "september_policy": (
            "September is processed and retained in all outputs. Its HV and "
            "HV/HH values are calculated and reported but not used in its "
            "evidence score while the Stage A anomaly remains unresolved."
        ),
        "class_codes": {
            "0": "INVALID",
            "1": "NON_WATER_SUPPORTED",
            "2": "WATER_SUPPORTED",
            "3": "UNCERTAIN"
        },
        "method_warning": (
            "Exploratory evidence scoring with pooled reference quantiles. "
            "Thresholds are not validated physical boundaries; independent "
            "reference data and sensitivity analysis are required."
        ),
        "acquisitions": {s["date"]: s for s in summaries}
    }
    with open(OUTPUT_DIR / "stage_C1_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\nSTAGE C.1 COMPLETE")
    print(f"All dates processed: {', '.join(DATES)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("These are preliminary evidence classes, not validated water maps.")


if __name__ == "__main__":
    main()

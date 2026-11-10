"""
10_acquisition_quality_gate.py

Stage A: Acquisition-level quality gate for the NISAR S-band GCOV AOI crops.

Expected folder structure:
S_BAND/
    10_acquisition_quality_gate.py
    aoi_outputs/
        20260819/
            HHHH.npy, HVHV.npy, HHHV.npy, mask.npy,
            inputDataExceptionMask.npy, rtcGammaToSigmaFactor.npy
        20260924/
            ...
        20261018/
            ...

Run from the S_BAND folder:
    python 10_acquisition_quality_gate.py

Important:
- This is a provisional, transparent screening gate, not a calibration verdict.
- Exception code 64 (SSAR sample shift) is recorded but NOT automatically rejected.
- Covariance consistency is an internal mathematical check, not proof of radiometric accuracy.
- The 6 dB temporal-ratio threshold is an initial configurable screening threshold.
  Review it against more acquisitions before treating it as a validated threshold.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- CONFIGURATION ---------------------------

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
OUTPUT_DIR = BASE_DIR / "acquisition_quality_output"
PLOT_DIR = OUTPUT_DIR / "quality_diagnostics"

# Known acquisitions. Add further date folders here when available.
DATES = ["20260819", "20260924", "20261018"]

# Provisional screening thresholds. Tune only with documented evidence.
MIN_VALID_COVERAGE = 0.98
MAX_NONFINITE_FRACTION = 0.001
MAX_COVARIANCE_VIOLATION_FRACTION = 0.001
MAX_TEMPORAL_RATIO_DEVIATION_DB = 6.0
BLOCK_ROWS = 20
BLOCK_COLS = 20

# A small positive floor prevents log10(0). It is not a noise correction.
EPS = 1e-12


# --------------------------- FILE HELPERS ----------------------------

def load_array(date_dir: Path, stem: str):
    """Load a required AOI .npy file and return its array."""
    path = date_dir / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, mmap_mode="r")


def finite_fraction(arr):
    arr = np.asarray(arr)
    return float(np.isfinite(arr).mean())


def block_medians(arr, valid, block_rows=20, block_cols=20):
    """Return block medians, skipping blocks with fewer than 20 valid pixels."""
    arr = np.asarray(arr)
    valid = np.asarray(valid, dtype=bool)
    h, w = arr.shape
    row_edges = np.linspace(0, h, block_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, block_cols + 1, dtype=int)
    result = np.full((block_rows, block_cols), np.nan, dtype=np.float64)

    for r in range(block_rows):
        for c in range(block_cols):
            r0, r1 = row_edges[r], row_edges[r + 1]
            c0, c1 = col_edges[c], col_edges[c + 1]
            block = arr[r0:r1, c0:c1]
            block_valid = valid[r0:r1, c0:c1] & np.isfinite(block)
            values = block[block_valid]
            if values.size >= 20:
                result[r, c] = np.median(values)
    return result


def safe_percentile(values, q):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


# --------------------------- LOAD & MEASURE --------------------------

def inspect_acquisition(date):
    date_dir = AOI_DIR / date

    hh = np.asarray(load_array(date_dir, "HHHH"), dtype=np.float64)
    hv = np.asarray(load_array(date_dir, "HVHV"), dtype=np.float64)
    hhv = np.asarray(load_array(date_dir, "HHHV"))
    mask = np.asarray(load_array(date_dir, "mask"))
    exception = np.asarray(load_array(date_dir, "inputDataExceptionMask"))
    rtc = np.asarray(load_array(date_dir, "rtcGammaToSigmaFactor"), dtype=np.float64)

    shapes = {
        "HHHH": hh.shape, "HVHV": hv.shape, "HHHV": hhv.shape,
        "mask": mask.shape, "exception": exception.shape, "RTC": rtc.shape
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"{date}: AOI array shapes do not match: {shapes}")

    # NISAR mask: 255 is outside acquisition; 0 is invalid/not fully focused;
    # nonzero values other than 255 identify valid subswaths.
    valid_mask = (mask != 0) & (mask != 255)
    finite_all = (
        np.isfinite(hh) & np.isfinite(hv) &
        np.isfinite(hhv.real) & np.isfinite(hhv.imag) &
        np.isfinite(rtc)
    )
    valid = valid_mask & finite_all & (hh > 0) & (hv >= 0) & (rtc > 0)

    valid_coverage = float(valid.mean())
    mask_coverage = float(valid_mask.mean())
    nonfinite_fraction = 1.0 - float(finite_all[valid_mask].mean()) if valid_mask.any() else 1.0

    # Covariance positive-semidefinite consistency:
    # gamma^2 = |HHHV|^2 / (HHHH * HVHV), expected in [0, 1].
    denom = hh * hv
    gamma2 = np.full(hh.shape, np.nan, dtype=np.float64)
    cov_valid = valid & (denom > 0)
    gamma2[cov_valid] = (np.abs(hhv[cov_valid]) ** 2) / denom[cov_valid]
    # Allow tiny floating-point tolerance.
    violations = cov_valid & (gamma2 > 1.000001)
    violation_fraction = (
        float(violations.sum() / cov_valid.sum()) if cov_valid.any() else 1.0
    )

    # Pixel-wise HV/HH ratio. A common multiplicative RTC factor cancels
    # in this ratio if applied equally to both covariance terms.
    ratio_db = np.full(hh.shape, np.nan, dtype=np.float64)
    ratio_valid = valid & (hh > 0) & (hv > 0)
    ratio_db[ratio_valid] = 10.0 * np.log10(
        np.maximum(hv[ratio_valid], EPS) / np.maximum(hh[ratio_valid], EPS)
    )

    ratio_values = ratio_db[np.isfinite(ratio_db)]
    block_ratio = block_medians(ratio_db, ratio_valid, BLOCK_ROWS, BLOCK_COLS)

    # Record exception-mask distribution without treating code 64 as invalid.
    exc_values, exc_counts = np.unique(exception, return_counts=True)
    exception_counts = {str(int(v)): int(n) for v, n in zip(exc_values, exc_counts)}

    metrics = {
        "date": date,
        "shape": f"{hh.shape[0]}x{hh.shape[1]}",
        "mask_coverage_fraction": mask_coverage,
        "valid_coverage_fraction": valid_coverage,
        "nonfinite_fraction_within_mask": nonfinite_fraction,
        "valid_pixel_count": int(valid.sum()),
        "exception_code_64_fraction": float((exception == 64).mean()),
        "exception_code_0_fraction": float((exception == 0).mean()),
        "exception_code_counts": exception_counts,
        "HH_median": safe_percentile(hh[valid], 50),
        "HV_median": safe_percentile(hv[valid], 50),
        "HV_HH_ratio_median_dB": safe_percentile(ratio_values, 50),
        "HV_HH_ratio_p02_dB": safe_percentile(ratio_values, 2),
        "HV_HH_ratio_p98_dB": safe_percentile(ratio_values, 98),
        "covariance_gamma2_median": safe_percentile(gamma2[cov_valid], 50),
        "covariance_gamma2_p99": safe_percentile(gamma2[cov_valid], 99),
        "covariance_violation_fraction": violation_fraction,
        "RTC_median": safe_percentile(rtc[valid], 50),
        "valid_pixels": int(valid.sum()),
    }

    return {
        "metrics": metrics,
        "valid": valid,
        "ratio_db": ratio_db,
        "block_ratio": block_ratio,
        "gamma2": gamma2,
    }


# --------------------------- QUALITY DECISIONS -----------------------

def apply_temporal_gate(results):
    """Compare acquisition ratio medians against the temporal median reference."""
    finite_medians = [
        r["metrics"]["HV_HH_ratio_median_dB"]
        for r in results.values()
        if np.isfinite(r["metrics"]["HV_HH_ratio_median_dB"])
    ]
    if len(finite_medians) < 2:
        raise RuntimeError("Need at least two acquisitions with valid HV/HH ratios.")

    # Median is robust for the current three-date case (two similar dates,
    # one outlier), but a larger time series is preferable.
    reference_db = float(np.median(finite_medians))

    for date, result in results.items():
        m = result["metrics"]
        deviation = abs(m["HV_HH_ratio_median_dB"] - reference_db)
        m["temporal_reference_ratio_median_dB"] = reference_db
        m["temporal_ratio_deviation_dB"] = float(deviation)

        reasons = []
        hard_failures = []

        if m["valid_coverage_fraction"] < MIN_VALID_COVERAGE:
            hard_failures.append(
                f"Valid AOI coverage below {MIN_VALID_COVERAGE:.1%}"
            )
        if m["nonfinite_fraction_within_mask"] > MAX_NONFINITE_FRACTION:
            hard_failures.append(
                f"Non-finite fraction exceeds {MAX_NONFINITE_FRACTION:.3%}"
            )
        if m["covariance_violation_fraction"] > MAX_COVARIANCE_VIOLATION_FRACTION:
            hard_failures.append(
                "Covariance consistency violations exceed threshold"
            )

        if deviation > MAX_TEMPORAL_RATIO_DEVIATION_DB:
            reasons.append(
                f"HV/HH median differs from temporal reference by {deviation:.2f} dB"
            )

        # Spatial consistency: fraction of valid blocks > threshold from
        # the same temporal reference. This is diagnostic, not a second
        # independent temporal reference.
        blocks = result["block_ratio"]
        finite_blocks = blocks[np.isfinite(blocks)]
        if finite_blocks.size:
            block_dev = np.abs(finite_blocks - reference_db)
            fraction_far = float(
                np.mean(block_dev > MAX_TEMPORAL_RATIO_DEVIATION_DB)
            )
        else:
            fraction_far = 1.0
        m["fraction_spatial_blocks_far_from_reference"] = fraction_far

        if fraction_far > 0.50:
            reasons.append(
                f"{fraction_far:.1%} of spatial blocks differ from temporal reference"
            )

        if hard_failures:
            status = "REJECT"
            reasons = hard_failures + reasons
        elif reasons:
            status = "FLAG"
        else:
            status = "ACCEPT"

        m["quality_status"] = status
        m["quality_reasons"] = reasons


# --------------------------- OUTPUTS ---------------------------------

def make_pixel_quality_flag(result):
    """
    Pixel flag values:
      0 = invalid / excluded by input validity checks
      1 = valid pixel in an ACCEPT acquisition
      2 = valid pixel in a FLAG or REJECT acquisition
    """
    status = result["metrics"]["quality_status"]
    flag = np.zeros(result["valid"].shape, dtype=np.uint8)
    flag[result["valid"]] = 1 if status == "ACCEPT" else 2
    return flag


def save_outputs(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    report = {
        "title": "NISAR S-band AOI acquisition-level quality gate",
        "note": (
            "Provisional screening only. ACCEPT does not certify absolute "
            "radiometric calibration. FLAG means investigate before quantitative "
            "temporal use. Exception code 64 is recorded, not automatically rejected."
        ),
        "thresholds": {
            "minimum_valid_coverage": MIN_VALID_COVERAGE,
            "maximum_nonfinite_fraction": MAX_NONFINITE_FRACTION,
            "maximum_covariance_violation_fraction": MAX_COVARIANCE_VIOLATION_FRACTION,
            "maximum_temporal_ratio_deviation_db": MAX_TEMPORAL_RATIO_DEVIATION_DB,
        },
        "acquisitions": {}
    }

    for date, result in results.items():
        m = result["metrics"]
        rows.append({
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in m.items()
        })
        report["acquisitions"][date] = m

        flag = make_pixel_quality_flag(result)
        np.save(OUTPUT_DIR / f"{date}_quality_flags.npy", flag)

        # Save per-date block medians for reproducibility.
        np.save(OUTPUT_DIR / f"{date}_HV_HH_block_medians_dB.npy",
                result["block_ratio"])

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "acquisition_quality_report.csv", index=False
    )
    with open(OUTPUT_DIR / "acquisition_quality_report.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    # Plot temporal median ratios.
    dates = list(results.keys())
    medians = [results[d]["metrics"]["HV_HH_ratio_median_dB"] for d in dates]
    ref = results[dates[0]]["metrics"]["temporal_reference_ratio_median_dB"]

    plt.figure(figsize=(9, 5))
    plt.plot(dates, medians, marker="o", linewidth=2, label="Acquisition median")
    plt.axhline(ref, linestyle="--", label=f"Temporal median reference ({ref:.2f} dB)")
    plt.axhspan(
        ref - MAX_TEMPORAL_RATIO_DEVIATION_DB,
        ref + MAX_TEMPORAL_RATIO_DEVIATION_DB,
        alpha=0.15,
        label=f"Provisional ±{MAX_TEMPORAL_RATIO_DEVIATION_DB:.0f} dB band"
    )
    plt.ylabel("HV/HH ratio (dB)")
    plt.xlabel("Acquisition date")
    plt.title("Acquisition-level polarimetric temporal screening")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "temporal_HV_HH_quality_gate.png", dpi=200)
    plt.close()

    # Block maps, one per date.
    for date, result in results.items():
        plt.figure(figsize=(8, 6))
        im = plt.imshow(result["block_ratio"], cmap="viridis", aspect="auto")
        plt.colorbar(im, label="HV/HH ratio (dB)")
        plt.title(f"{date}: spatial block medians of HV/HH ratio")
        plt.xlabel("Block column")
        plt.ylabel("Block row")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{date}_HV_HH_block_map.png", dpi=200)
        plt.close()


def main():
    if not AOI_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find {AOI_DIR}. Put this script in the S_BAND folder "
            "that contains aoi_outputs."
        )

    results = {}
    for date in DATES:
        print(f"\nProcessing {date} ...")
        results[date] = inspect_acquisition(date)
        m = results[date]["metrics"]
        print(f"  Valid coverage: {m['valid_coverage_fraction']:.4%}")
        print(f"  HV/HH median:   {m['HV_HH_ratio_median_dB']:.2f} dB")
        print(f"  Covariance violations: {m['covariance_violation_fraction']:.6%}")

    apply_temporal_gate(results)
    save_outputs(results)

    print("\n========== ACQUISITION QUALITY GATE ==========")
    for date, result in results.items():
        m = result["metrics"]
        print(f"\n{date}: {m['quality_status']}")
        print(f"  HV/HH deviation: {m['temporal_ratio_deviation_dB']:.2f} dB")
        for reason in m["quality_reasons"]:
            print(f"  - {reason}")
        if not m["quality_reasons"]:
            print("  - No configured screening rule was triggered.")

    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")
    print("\nPixel flag legend: 0=invalid, 1=valid ACCEPT, 2=valid FLAG/REJECT")
    print("Review the CSV/JSON and plots before using the statuses in research.")


if __name__ == "__main__":
    main()

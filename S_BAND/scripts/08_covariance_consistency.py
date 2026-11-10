"""
08_covariance_consistency.py

Checks the 2x2 HH/HV covariance-matrix consistency condition for three
NISAR S-band GCOV AOI extracts.

Expected input layout:
aoi_outputs/
  20260819/ HHHH.npy HVHV.npy HHHV.npy mask.npy inputDataExceptionMask.npy
  20260924/ ...
  20261018/ ...

The test is:
    gamma2 = abs(HHHV)**2 / (HHHH * HVHV)
For a positive-semidefinite covariance matrix, gamma2 should be <= 1
(up to numerical tolerance), where all terms are from the same pixel and
have consistent scaling/validity.

IMPORTANT:
- This is an internal covariance-consistency diagnostic, not proof of
  radiometric calibration accuracy.
- Do not apply RTC Gamma-to-Sigma factors in this test.
- Pixels with non-finite or non-positive diagonal terms are excluded.
- mask==0 and mask==255 are excluded when a mask file is available.
- Exception codes are reported separately; code 64 is NOT automatically
  discarded.
"""

from pathlib import Path
import csv
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
OUT_DIR = BASE_DIR / "covariance_consistency_output"
DATES = ["20260819", "20260924", "20261018"]
TOLERANCE = 1e-6
SAMPLE_LIMIT = 2_000_000
RNG_SEED = 42


def percentile_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def load_array(folder, name, required=True):
    path = folder / f"{name}.npy"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return np.load(path, mmap_mode="r")


def analyse_date(date):
    folder = AOI_DIR / date
    hh = load_array(folder, "HHHH")
    hv = load_array(folder, "HVHV")
    cross = load_array(folder, "HHHV")
    mask = load_array(folder, "mask", required=False)
    exception = load_array(folder, "inputDataExceptionMask", required=False)

    if not (hh.shape == hv.shape == cross.shape):
        raise ValueError(
            f"{date}: array shapes differ: HHHH={hh.shape}, "
            f"HVHV={hv.shape}, HHHV={cross.shape}"
        )
    if mask is not None and mask.shape != hh.shape:
        raise ValueError(f"{date}: mask shape {mask.shape} != data shape {hh.shape}")
    if exception is not None and exception.shape != hh.shape:
        raise ValueError(
            f"{date}: exception shape {exception.shape} != data shape {hh.shape}"
        )

    # Convert to float64/complex128 for stable arithmetic without changing
    # the stored source arrays.
    hh = np.asarray(hh, dtype=np.float64)
    hv = np.asarray(hv, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.complex128)

    valid = (
        np.isfinite(hh) & np.isfinite(hv)
        & np.isfinite(cross.real) & np.isfinite(cross.imag)
        & (hh > 0) & (hv > 0)
    )
    if mask is not None:
        valid &= (mask != 0) & (mask != 255)

    denominator = np.full(hh.shape, np.nan, dtype=np.float64)
    numerator = np.abs(cross) ** 2
    denominator[valid] = hh[valid] * hv[valid]

    gamma2 = np.full(hh.shape, np.nan, dtype=np.float64)
    gamma2[valid] = numerator[valid] / denominator[valid]

    valid_gamma = gamma2[valid]
    violations = valid_gamma > (1.0 + TOLERANCE)

    # Deterministic sample for CSV summaries and compact diagnostic files.
    flat_indices = np.flatnonzero(valid)
    if flat_indices.size > SAMPLE_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        flat_indices = np.sort(rng.choice(flat_indices, SAMPLE_LIMIT, replace=False))

    sampled_gamma = gamma2.ravel()[flat_indices]
    sample_csv = OUT_DIR / f"{date}_gamma2_sample.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flat_pixel_index", "row", "column", "gamma_squared"])
        width = hh.shape[1]
        for idx, value in zip(flat_indices, sampled_gamma):
            writer.writerow([int(idx), int(idx // width), int(idx % width), float(value)])

    # Store a full-resolution diagnostic raster for mapping if desired.
    np.save(OUT_DIR / f"{date}_gamma2.npy", gamma2.astype(np.float32))

    rows = []
    groups = [("ALL", np.ones(hh.shape, dtype=bool))]
    if exception is not None:
        for code in np.unique(exception):
            groups.append((f"EXCEPTION_{int(code)}", exception == code))

    for group_name, group_mask in groups:
        group_valid = valid & group_mask
        vals = gamma2[group_valid]
        if vals.size:
            row = {
                "date": date,
                "group": group_name,
                "pixels_in_group": int(np.count_nonzero(group_mask)),
                "valid_pixels": int(vals.size),
                "valid_fraction": float(vals.size / max(1, np.count_nonzero(group_mask))),
                "violation_count_gamma2_gt_1": int(np.count_nonzero(vals > 1.0 + TOLERANCE)),
                "violation_percent": float(
                    100.0 * np.count_nonzero(vals > 1.0 + TOLERANCE) / vals.size
                ),
                **percentile_summary(vals),
            }
        else:
            row = {
                "date": date, "group": group_name,
                "pixels_in_group": int(np.count_nonzero(group_mask)),
                "valid_pixels": 0, "valid_fraction": 0.0,
                "violation_count_gamma2_gt_1": 0, "violation_percent": 0.0,
            }
        rows.append(row)

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not AOI_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {AOI_DIR}. Put this script beside the aoi_outputs folder."
        )

    all_rows = []
    for date in DATES:
        print(f"\nAnalysing {date} ...")
        rows = analyse_date(date)
        all_rows.extend(rows)
        all_row = next(r for r in rows if r["group"] == "ALL")
        print(
            f"  valid pixels: {all_row['valid_pixels']:,}; "
            f"median gamma^2: {all_row.get('median', float('nan')):.6g}; "
            f"p99: {all_row.get('p99', float('nan')):.6g}; "
            f"violations > 1: {all_row['violation_percent']:.6f}%"
        )

    # Union of fields across rows, preserving a readable column order.
    preferred = [
        "date", "group", "pixels_in_group", "valid_pixels", "valid_fraction",
        "violation_count_gamma2_gt_1", "violation_percent",
        "count", "min", "p01", "median", "mean", "p99", "max",
    ]
    fields = preferred
    out_csv = OUT_DIR / "covariance_consistency_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nFinished.")
    print(f"Summary CSV: {out_csv}")
    print(f"Full gamma^2 arrays and sampled pixel CSVs: {OUT_DIR}")
    print("\nInterpretation reminder:")
    print("- Values <= 1 are consistent with the 2x2 covariance PSD condition.")
    print("- Values > 1 beyond tolerance are flagged for investigation.")
    print("- A pass does NOT establish radiometric calibration correctness.")
    print("- Compare exception groups, and do not automatically discard code 64.")


if __name__ == "__main__":
    main()

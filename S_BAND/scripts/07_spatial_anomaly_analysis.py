"""
07_spatial_anomaly_analysis.py

Investigates the September 2026 S-band HVHV anomaly in the extracted AOI.
Reads the .npy layers produced by 04_extract_aoi.py.

Outputs:
  spatial_anomaly_output/exception_group_stats.csv
  spatial_anomaly_output/block_statistics.csv
  spatial_anomaly_output/rtc_factor_stats.csv
  spatial_anomaly_output/hvhv_hhhv_date_summary.csv
  spatial_anomaly_output/plots/*.png

Important:
- Does not delete or modify source data.
- Does not automatically discard exception code 64.
- Reports raw GCOV-term statistics and a separate diagnostic multiplication
  by rtcGammaToSigmaFactor. It does NOT assert that multiplication is the
  correct product conversion; verify the product specification first.
- HHHV is complex covariance. Its real, imaginary and magnitude are reported
  separately; it is not treated as an intensity channel.
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
OUT_DIR = BASE_DIR / "spatial_anomaly_output"
PLOT_DIR = OUT_DIR / "plots"

DATES = ["20260819", "20260924", "20261018"]
LAYERS = [
    "HHHH",
    "HVHV",
    "HHHV",
    "inputDataExceptionMask",
    "mask",
    "rtcGammaToSigmaFactor",
]
BLOCKS = 20  # 20 x 20 spatial blocks over the AOI


def load_layer(date, layer):
    path = AOI_DIR / date / f"{layer}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path, mmap_mode="r")


def finite_positive(a):
    a = np.asarray(a)
    return np.isfinite(a) & (a > 0)


def describe(a):
    a = np.asarray(a)
    good = np.isfinite(a)
    v = a[good]
    if v.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan,
                "std": np.nan, "p02": np.nan, "p98": np.nan,
                "min": np.nan, "max": np.nan}
    return {
        "count": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p02": float(np.percentile(v, 2)),
        "p98": float(np.percentile(v, 98)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    # Confirm all dates have the expected aligned crop shape.
    arrays = {}
    shape = None
    for date in DATES:
        arrays[date] = {}
        for layer in LAYERS:
            arr = load_layer(date, layer)
            if shape is None:
                shape = arr.shape
            if arr.shape != shape:
                raise ValueError(
                    f"Shape mismatch: {date}/{layer} is {arr.shape}, expected {shape}. "
                    "Do not compare dates until grids are aligned."
                )
            arrays[date][layer] = arr
        print(f"Loaded {date}: shape={shape}")

    # 1) Per-date statistics by exception code.
    exception_rows = []
    for date in DATES:
        exc = np.asarray(arrays[date]["inputDataExceptionMask"])
        mask = np.asarray(arrays[date]["mask"])
        base = (mask == 1) & np.isfinite(exc)
        for code in np.unique(exc[base]).astype(int):
            pix = base & (exc == code)
            row = {
                "date": date,
                "exception_code": int(code),
                "pixel_count": int(pix.sum()),
                "pixel_percent_of_AOI": float(100 * pix.sum() / base.sum()),
            }
            for layer in ["HHHH", "HVHV"]:
                data = np.asarray(arrays[date][layer])
                valid = pix & finite_positive(data)
                s = describe(data[valid])
                for key, value in s.items():
                    row[f"{layer}_{key}"] = value
                row[f"{layer}_positive_fraction"] = float(valid.sum() / max(pix.sum(), 1))
            exception_rows.append(row)
    save_csv(OUT_DIR / "exception_group_stats.csv", exception_rows)

    # 2) Spatial block statistics: same row/column blocks for all dates.
    block_rows = []
    h, w = shape
    row_edges = np.linspace(0, h, BLOCKS + 1, dtype=int)
    col_edges = np.linspace(0, w, BLOCKS + 1, dtype=int)
    for date in DATES:
        exc = np.asarray(arrays[date]["inputDataExceptionMask"])
        mask = np.asarray(arrays[date]["mask"])
        hh = np.asarray(arrays[date]["HHHH"])
        hv = np.asarray(arrays[date]["HVHV"])
        rtc = np.asarray(arrays[date]["rtcGammaToSigmaFactor"])
        for ri in range(BLOCKS):
            for ci in range(BLOCKS):
                r0, r1 = row_edges[ri], row_edges[ri + 1]
                c0, c1 = col_edges[ci], col_edges[ci + 1]
                sl = np.s_[r0:r1, c0:c1]
                valid = (mask[sl] == 1)
                row = {
                    "date": date, "block_row": ri, "block_col": ci,
                    "row_start": r0, "row_end": r1,
                    "col_start": c0, "col_end": c1,
                    "valid_mask_pixels": int(valid.sum()),
                    "code0_percent": float(100 * np.mean(exc[sl][valid] == 0)) if valid.any() else np.nan,
                    "code64_percent": float(100 * np.mean(exc[sl][valid] == 64)) if valid.any() else np.nan,
                }
                for name, data in [("HHHH", hh), ("HVHV", hv)]:
                    v = data[sl]
                    good = valid & finite_positive(v)
                    s = describe(v[good])
                    row[f"{name}_positive_count"] = s["count"]
                    row[f"{name}_median"] = s["median"]
                    row[f"{name}_mean"] = s["mean"]
                    row[f"{name}_p02"] = s["p02"]
                    row[f"{name}_p98"] = s["p98"]
                rs = describe(rtc[sl][valid & np.isfinite(rtc[sl])])
                row["RTC_median"] = rs["median"]
                row["RTC_mean"] = rs["mean"]
                block_rows.append(row)
    save_csv(OUT_DIR / "block_statistics.csv", block_rows)

    # 3) Date-level summary for HH, HV, complex HHHV and RTC.
    summary_rows = []
    for date in DATES:
        hh = np.asarray(arrays[date]["HHHH"])
        hv = np.asarray(arrays[date]["HVHV"])
        cross = np.asarray(arrays[date]["HHHV"])
        rtc = np.asarray(arrays[date]["rtcGammaToSigmaFactor"])
        exc = np.asarray(arrays[date]["inputDataExceptionMask"])
        mask = np.asarray(arrays[date]["mask"]) == 1
        row = {"date": date, "code64_percent": float(100 * np.mean(exc[mask] == 64))}
        for name, data in [("HHHH_raw", hh), ("HVHV_raw", hv),
                           ("HHHV_real", cross.real), ("HHHV_imag", cross.imag),
                           ("HHHV_magnitude", np.abs(cross)),
                           ("RTC_factor", rtc)]:
            valid = mask & np.isfinite(data)
            if name in ["HHHH_raw", "HVHV_raw", "HHHV_magnitude"]:
                valid &= data > 0
            s = describe(data[valid])
            for k, v in s.items():
                row[f"{name}_{k}"] = v

        # Diagnostic only: multiplication is reported separately, not assumed
        # to be the official conversion convention.
        hv_diag = hv * rtc
        valid = mask & finite_positive(hv_diag) & np.isfinite(rtc)
        s = describe(hv_diag[valid])
        for k, v in s.items():
            row[f"HVHV_times_RTC_diagnostic_{k}"] = v
        summary_rows.append(row)
    save_csv(OUT_DIR / "hvhv_hhhv_date_summary.csv", summary_rows)

    # 4) Date-comparable spatial maps for HVHV and HHHH, using one common
    # percentile stretch per layer. Display transform only: 10*log10(raw term).
    for layer in ["HHHH", "HVHV"]:
        pooled = []
        for date in DATES:
            a = np.asarray(arrays[date][layer])
            v = a[finite_positive(a)]
            if v.size:
                stride = max(1, v.size // 300000)
                pooled.append(v[::stride])
        pooled = np.concatenate(pooled)
        lo, hi = np.percentile(10 * np.log10(pooled), [2, 98])
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        for ax, date in zip(axes, DATES):
            a = np.asarray(arrays[date][layer], dtype=np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10 * np.log10(np.where(a > 0, a, np.nan))
            im = ax.imshow(db, vmin=lo, vmax=hi, cmap="gray")
            ax.set_title(date)
            ax.set_xlabel("AOI pixel column")
            ax.set_ylabel("AOI pixel row")
        fig.colorbar(im, ax=axes, label="10 log10(raw stored GCOV term), display only")
        fig.suptitle(f"{layer}: common stretch across dates")
        fig.savefig(PLOT_DIR / f"{layer}_common_stretch.png", dpi=160)
        plt.close(fig)

    # 5) HVHV ratio/difference maps against August. These are raw-term
    # diagnostics, not calibrated physical change maps.
    aug = np.asarray(arrays["20260819"]["HVHV"], dtype=np.float64)
    for date in ["20260924", "20261018"]:
        cur = np.asarray(arrays[date]["HVHV"], dtype=np.float64)
        valid = finite_positive(aug) & finite_positive(cur)
        diff = np.full(aug.shape, np.nan, dtype=np.float32)
        diff[valid] = (10 * np.log10(cur[valid] / aug[valid])).astype(np.float32)
        finite = diff[np.isfinite(diff)]
        if finite.size:
            lim = np.percentile(np.abs(finite), 98)
            lim = max(float(lim), 1.0)
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"HVHV raw-term difference: {date} vs 20260819")
            ax.set_xlabel("AOI pixel column")
            ax.set_ylabel("AOI pixel row")
            fig.colorbar(im, ax=ax, label="10 log10(current / August), dB diagnostic")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"HVHV_difference_{date}_vs_20260819.png", dpi=160)
            plt.close(fig)

    # 6) Exception mask maps.
    for date in DATES:
        exc = np.asarray(arrays[date]["inputDataExceptionMask"])
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(exc, vmin=0, vmax=64)
        ax.set_title(f"{date}: inputDataExceptionMask")
        ax.set_xlabel("AOI pixel column")
        ax.set_ylabel("AOI pixel row")
        fig.colorbar(ax.images[0], ax=ax, label="Exception code")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"exception_mask_{date}.png", dpi=160)
        plt.close(fig)

    print("\nDONE.")
    print(f"Outputs: {OUT_DIR}")
    print("Start by opening:")
    print("  exception_group_stats.csv")
    print("  block_statistics.csv")
    print("  hvhv_hhhv_date_summary.csv")
    print(f"  plots folder: {PLOT_DIR}")
    print("\nNo source .npy files were modified or deleted.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        if e.name == "matplotlib":
            print("Install matplotlib first: python -m pip install matplotlib")
        else:
            raise

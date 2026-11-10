"""
09_polarization_ratio_analysis.py

Compares the pixel-wise HV/HH power ratio across the three NISAR S-band
GCOV AOI acquisitions.

Inputs in aoi_outputs/<date>/:
  HHHH.npy, HVHV.npy, mask.npy, inputDataExceptionMask.npy

Ratio:
  HV_HH_ratio = HVHV / HHHH
  HV_HH_dB = 10*log10(HVHV / HHHH)

Because both terms are from the same GCOV product and same pixel, the
pixel-wise RTC Gamma-to-Sigma factor is expected to cancel in this ratio
if applied identically to both diagonal covariance terms. This script
therefore compares the stored ratio and does not apply an RTC correction.

This is a diagnostic comparison, not a water classifier and not proof
that the acquisitions are radiometrically interchangeable.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
AOI = BASE / "aoi_outputs"
OUT = BASE / "polarization_ratio_output"
PLOTS = OUT / "plots"
DATES = ["20260819", "20260924", "20261018"]


def load(date, name):
    p = AOI / date / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, mmap_mode="r")


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return {}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p02": float(np.percentile(x, 2)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p98": float(np.percentile(x, 98)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    rows = []
    block_rows = []
    ratio_maps = {}

    for date in DATES:
        hh = np.asarray(load(date, "HHHH"), dtype=np.float64)
        hv = np.asarray(load(date, "HVHV"), dtype=np.float64)
        mask = np.asarray(load(date, "mask"))
        exc = np.asarray(load(date, "inputDataExceptionMask"))

        if not (hh.shape == hv.shape == mask.shape == exc.shape):
            raise ValueError(f"{date}: input array shapes do not match")

        valid = (
            (mask != 0) & (mask != 255) &
            np.isfinite(hh) & np.isfinite(hv) & (hh > 0) & (hv > 0)
        )
        ratio = np.full(hh.shape, np.nan, dtype=np.float32)
        ratio[valid] = (hv[valid] / hh[valid]).astype(np.float32)
        ratio_db = np.full(hh.shape, np.nan, dtype=np.float32)
        ratio_db[valid] = (10.0 * np.log10(ratio[valid])).astype(np.float32)
        ratio_maps[date] = ratio_db
        np.save(OUT / f"{date}_HV_HH_ratio_dB.npy", ratio_db)

        for group, gm in [
            ("ALL", np.ones(exc.shape, dtype=bool)),
            ("EXCEPTION_0", exc == 0),
            ("EXCEPTION_64", exc == 64),
        ]:
            use = valid & gm
            s_lin = stats(ratio[use])
            s_db = stats(ratio_db[use])
            rows.append({
                "date": date, "group": group,
                "valid_pixels": int(use.sum()),
                **{f"ratio_{k}": v for k, v in s_lin.items()},
                **{f"ratio_dB_{k}": v for k, v in s_db.items()},
            })

        # 20 x 20 spatial blocks, using the same grid for each acquisition.
        h, w = hh.shape
        re = np.linspace(0, h, 21, dtype=int)
        ce = np.linspace(0, w, 21, dtype=int)
        for ri in range(20):
            for ci in range(20):
                sl = np.s_[re[ri]:re[ri+1], ce[ci]:ce[ci+1]]
                v = ratio_db[sl]
                v = v[np.isfinite(v)]
                block_rows.append({
                    "date": date, "block_row": ri, "block_col": ci,
                    "valid_pixels": int(v.size),
                    "HV_HH_ratio_dB_median": float(np.median(v)) if v.size else np.nan,
                    "HV_HH_ratio_dB_mean": float(np.mean(v)) if v.size else np.nan,
                })

        print(f"{date}: valid={valid.sum():,}, "
              f"median HV/HH={np.nanmedian(ratio[valid]):.8g}, "
              f"median ratio dB={np.nanmedian(ratio_db[valid]):.4f} dB")

    def write_csv(path, data):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(data)

    write_csv(OUT / "polarization_ratio_summary.csv", rows)
    write_csv(OUT / "polarization_ratio_block_stats.csv", block_rows)

    # Common display range across dates for fair map comparison.
    all_values = np.concatenate([
        x[np.isfinite(x)][::max(1, np.isfinite(x).sum() // 250000)]
        for x in ratio_maps.values()
    ])
    lo, hi = np.percentile(all_values, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, date in zip(axes, DATES):
        im = ax.imshow(ratio_maps[date], cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(date)
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
    fig.colorbar(im, ax=axes, label="10 log10(HVHV / HHHH), dB")
    fig.suptitle("HV/HH power ratio: common scale across acquisitions")
    fig.savefig(PLOTS / "HV_HH_ratio_common_scale.png", dpi=160)
    plt.close(fig)

    print("\nDone. Outputs:", OUT)
    print("Upload polarization_ratio_summary.csv and "
          "polarization_ratio_block_stats.csv next.")


if __name__ == "__main__":
    main()

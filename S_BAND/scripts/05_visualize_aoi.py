"""
05_visualize_aoi.py

Diagnostic visualisation for the NISAR S-band AOI crops created by 04_extract_aoi.py.

What it creates:
- HH and HV backscatter-look maps in display dB (10*log10 of the stored positive
  GCOV term). This is for visual comparison only, not a declaration that the
  product is officially Sigma0/dB.
- InputDataExceptionMask maps.
- Log-value histograms split by exception code 0 and 64.
- CSV with HH/HV statistics split by exception code.

Run from the folder containing this script:
    python 05_visualize_aoi.py

Expected input:
    ./aoi_outputs/YYYYMMDD/HHHH.npy
    ./aoi_outputs/YYYYMMDD/HVHV.npy
    ./aoi_outputs/YYYYMMDD/mask.npy
    ./aoi_outputs/YYYYMMDD/inputDataExceptionMask.npy
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
PLOT_DIR = AOI_DIR / "diagnostic_plots"
STATS_CSV = AOI_DIR / "quality_split_statistics.csv"

CHANNELS = {
    "HHHH": "HH-HH (HH intensity/covariance term)",
    "HVHV": "HV-HV (HV intensity/covariance term)",
}

# Use the same display range for each date within a channel so dates can be
# compared visually. Percentiles are calculated from all positive, finite
# pixels across the three AOI crops.
DISPLAY_PERCENTILES = (2, 98)


def load_array(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def display_db(array):
    """Convert positive linear values to 10*log10 for display only."""
    result = np.full(array.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(array) & (array > 0)
    result[valid] = 10.0 * np.log10(array[valid])
    return result


def get_group_codes(exception_mask):
    """Return stable quality groups without assuming code 64 is invalid."""
    groups = {
        "Code 0 (no exception)": exception_mask == 0,
        "Code 64 (sample-shift flag)": exception_mask == 64,
    }
    other = (exception_mask != 0) & (exception_mask != 64) & (exception_mask != 255)
    if np.any(other):
        groups["Other exception codes"] = other
    return groups


def calculate_stats(values):
    valid = np.isfinite(values) & (values > 0)
    x = values[valid]
    if x.size == 0:
        return {
            "pixel_count": 0,
            "median_linear": np.nan,
            "mean_linear": np.nan,
            "p02_linear": np.nan,
            "p98_linear": np.nan,
            "median_display_db": np.nan,
        }
    median = float(np.median(x))
    return {
        "pixel_count": int(x.size),
        "median_linear": median,
        "mean_linear": float(np.mean(x)),
        "p02_linear": float(np.percentile(x, 2)),
        "p98_linear": float(np.percentile(x, 98)),
        "median_display_db": float(10.0 * np.log10(median)),
    }


def main():
    if not AOI_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find {AOI_DIR}. Put this script beside the aoi_outputs folder."
        )

    date_dirs = sorted(
        p for p in AOI_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    if not date_dirs:
        raise FileNotFoundError(f"No YYYYMMDD folders found inside {AOI_DIR}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    scenes = []
    for date_dir in date_dirs:
        required = [
            date_dir / "HHHH.npy",
            date_dir / "HVHV.npy",
            date_dir / "mask.npy",
            date_dir / "inputDataExceptionMask.npy",
        ]
        if all(p.exists() for p in required):
            scenes.append({
                "date": date_dir.name,
                "folder": date_dir,
                "HHHH": load_array(required[0]).astype(np.float32),
                "HVHV": load_array(required[1]).astype(np.float32),
                "mask": load_array(required[2]),
                "exception": load_array(required[3]),
            })
        else:
            missing = [str(p.name) for p in required if not p.exists()]
            print(f"Skipping {date_dir.name}; missing: {', '.join(missing)}")

    if not scenes:
        raise RuntimeError("No complete AOI date folders were found.")

    # Check all raster dimensions before comparing or plotting.
    for scene in scenes:
        shapes = {
            scene["HHHH"].shape,
            scene["HVHV"].shape,
            scene["mask"].shape,
            scene["exception"].shape,
        }
        if len(shapes) != 1:
            raise ValueError(
                f"Array shape mismatch for {scene['date']}: {sorted(shapes)}"
            )

    print(f"Found {len(scenes)} AOI scenes: {', '.join(s['date'] for s in scenes)}")

    # Compute common display limits per channel across all dates.
    display_limits = {}
    for channel in CHANNELS:
        all_db = []
        for scene in scenes:
            db = display_db(scene[channel])
            finite = db[np.isfinite(db)]
            if finite.size:
                # Bound memory use for percentile estimation on very large arrays.
                if finite.size > 500_000:
                    step = max(1, finite.size // 500_000)
                    finite = finite[::step]
                all_db.append(finite)
        if not all_db:
            display_limits[channel] = (-30, 0)
        else:
            combined = np.concatenate(all_db)
            lo, hi = np.percentile(combined, DISPLAY_PERCENTILES)
            display_limits[channel] = (float(lo), float(hi))

    stats_rows = []

    for scene in scenes:
        date = scene["date"]
        exception = scene["exception"]
        groups = get_group_codes(exception)

        # 1. Backscatter-look maps, with common channel scale across dates.
        for channel, description in CHANNELS.items():
            db = display_db(scene[channel])
            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(
                db,
                cmap="gray",
                vmin=display_limits[channel][0],
                vmax=display_limits[channel][1],
                interpolation="nearest",
            )
            ax.set_title(
                f"{date} | {description}\\n"
                "Display only: 10 log10(stored positive GCOV term)"
            )
            ax.set_xlabel("AOI pixel column")
            ax.set_ylabel("AOI pixel row")
            fig.colorbar(im, ax=ax, label="Display value (dB)")
            fig.tight_layout()
            out = PLOT_DIR / f"{date}_{channel}_display_db.png"
            fig.savefig(out, dpi=180)
            plt.close(fig)

        # 2. Exception-mask map.
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(exception, interpolation="nearest")
        ax.set_title(f"{date} | inputDataExceptionMask")
        ax.set_xlabel("AOI pixel column")
        ax.set_ylabel("AOI pixel row")
        fig.colorbar(im, ax=ax, label="Exception code")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{date}_exception_mask.png", dpi=180)
        plt.close(fig)

        # 3. Histograms and quality-split statistics.
        for channel, description in CHANNELS.items():
            values = scene[channel]
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted = 0

            for group_name, group_mask in groups.items():
                selected = values[group_mask]
                selected = selected[np.isfinite(selected) & (selected > 0)]
                if selected.size == 0:
                    continue

                # Plot log10(linear value). Subsample for speed, preserving
                # distribution reasonably for diagnostic visualisation.
                if selected.size > 300_000:
                    step = max(1, selected.size // 300_000)
                    selected = selected[::step]
                ax.hist(
                    np.log10(selected),
                    bins=100,
                    histtype="step",
                    density=True,
                    label=group_name,
                )
                plotted += 1

            ax.set_title(f"{date} | {channel} distribution by exception code")
            ax.set_xlabel("log10(stored positive GCOV term)")
            ax.set_ylabel("Density")
            if plotted:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No positive finite pixels", ha="center", va="center")
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f"{date}_{channel}_quality_histogram.png", dpi=180)
            plt.close(fig)

            for group_name, group_mask in groups.items():
                result = calculate_stats(values[group_mask])
                stats_rows.append({
                    "date": date,
                    "channel": channel,
                    "quality_group": group_name,
                    **result,
                })

        # Include the full-scene totals as a reference row.
        for channel in CHANNELS:
            result = calculate_stats(scene[channel])
            stats_rows.append({
                "date": date,
                "channel": channel,
                "quality_group": "All AOI pixels",
                **result,
            })

    stats = pd.DataFrame(stats_rows)
    stats.to_csv(STATS_CSV, index=False)

    # Compact console summary for the key comparison.
    print("\\nMedian values by date and quality group:")
    for channel in CHANNELS:
        print(f"\\n{channel}")
        subset = stats[
            (stats["channel"] == channel)
            & (stats["quality_group"].isin([
                "Code 0 (no exception)",
                "Code 64 (sample-shift flag)",
                "All AOI pixels",
            ]))
        ]
        if not subset.empty:
            print(subset[[
                "date", "quality_group", "pixel_count",
                "median_linear", "median_display_db"
            ]].to_string(index=False))

    print("\\nDone.")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Quality-split statistics saved to: {STATS_CSV}")
    print(
        "\\nImportant: code 64 is reported separately, not automatically removed. "
        "The plotted dB values are diagnostic transforms of the stored positive "
        "GCOV terms; do not label them official Sigma0 until the product scaling "
        "and RTC-factor convention have been verified."
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()


"""
SCRIPT 26
Full-Scene Mask Quality Analysis
QGPT-WM | NISAR S-band GCOV

Purpose:
1. Calculate joint mask distributions.
2. Identify valid pixels with SSAR sample-shift flags.
3. Identify invalid/fill pixels.
4. Verify spatial alignment of masks.
5. Generate coarse spatial quality maps.
6. Export CSV reports.

Does NOT modify source GeoTIFFs.
Does NOT automatically discard exception-64 pixels.
"""

from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "full_scene_geotiffs"

OUTPUT_DIR = BASE_DIR / "full_scene_quality_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = [
    "20260819",
    "20260924",
    "20261018"
]

MASK_FILE = "mask.tif"
EXCEPTION_FILE = "inputDataExceptionMask.tif"

# Spatial overview resolution.
# This does not affect the original pixel-level statistics.
OVERVIEW_HEIGHT = 1000
OVERVIEW_WIDTH = 1000

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def percentage(count, total):
    if total == 0:
        return 0.0
    return (count / total) * 100.0


def save_csv(path, headers, rows):

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def verify_alignment(mask_ds, exception_ds):

    if mask_ds.width != exception_ds.width:
        raise ValueError("Raster width mismatch")

    if mask_ds.height != exception_ds.height:
        raise ValueError("Raster height mismatch")

    if mask_ds.crs != exception_ds.crs:
        raise ValueError("CRS mismatch")

    if mask_ds.transform != exception_ds.transform:
        raise ValueError("Geotransform mismatch")

    return True


# ============================================================
# 3. PROCESS ONE ACQUISITION
# ============================================================

def process_date(date):

    print("\n" + "=" * 75)
    print(f"PROCESSING DATE: {date}")
    print("=" * 75)

    date_dir = INPUT_DIR / date

    mask_path = date_dir / MASK_FILE
    exception_path = date_dir / EXCEPTION_FILE

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    if not exception_path.exists():
        raise FileNotFoundError(exception_path)

    date_output = OUTPUT_DIR / date
    date_output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Open rasters
    # --------------------------------------------------------

    with rasterio.open(mask_path) as mask_ds, \
         rasterio.open(exception_path) as exception_ds:

        verify_alignment(mask_ds, exception_ds)

        height = mask_ds.height
        width = mask_ds.width

        total_pixels = height * width

        print(f"\nDimensions: {width} x {height}")
        print(f"Total pixels: {total_pixels:,}")
        print(f"CRS: {mask_ds.crs}")
        print(f"Resolution: {mask_ds.res}")

        # ----------------------------------------------------
        # Pixel counters
        # ----------------------------------------------------

        mask_counts = defaultdict(int)
        exception_counts = defaultdict(int)

        joint_counts = defaultdict(int)

        # Additional quality statistics
        valid_clean = 0
        valid_shift = 0
        valid_other_exception = 0

        valid_exception_255 = 0
        invalid_mask_zero = 0
        outside_mask_255 = 0

        other_mask_values = defaultdict(int)
        other_exception_values = defaultdict(int)

        # ----------------------------------------------------
        # Spatial overview accumulators
        # ----------------------------------------------------

        grid_h = OVERVIEW_HEIGHT
        grid_w = OVERVIEW_WIDTH

        coarse_total = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        coarse_valid = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        coarse_shift = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        # ----------------------------------------------------
        # Windowed full-scene processing
        # ----------------------------------------------------

        for _, window in mask_ds.block_windows(1):

            mask = mask_ds.read(
                1,
                window=window
            )

            exception = exception_ds.read(
                1,
                window=window
            )

            if mask.shape != exception.shape:
                raise ValueError("Window shape mismatch")

            # -----------------------------------------------
            # Individual mask counts
            # -----------------------------------------------

            values, counts = np.unique(
                mask,
                return_counts=True
            )

            for value, count in zip(values, counts):
                mask_counts[int(value)] += int(count)

            values, counts = np.unique(
                exception,
                return_counts=True
            )

            for value, count in zip(values, counts):
                exception_counts[int(value)] += int(count)

            # -----------------------------------------------
            # Joint mask distribution
            # -----------------------------------------------

            encoded = (
                mask.astype(np.uint16) * 256
                + exception.astype(np.uint16)
            )

            unique_pairs, pair_counts = np.unique(
                encoded,
                return_counts=True
            )

            for pair, count in zip(
                unique_pairs, pair_counts
            ):

                mask_value = int(pair // 256)
                exception_value = int(pair % 256)

                joint_counts[
                    (mask_value, exception_value)
                ] += int(count)

            # -----------------------------------------------
            # Main quality categories
            # -----------------------------------------------

            valid = mask == 1

            clean = valid & (exception == 0)

            shifted = valid & (exception == 64)

            exception_fill = valid & (exception == 255)

            invalid_zero = mask == 0

            outside = mask == 255

            valid_clean += int(np.count_nonzero(clean))

            valid_shift += int(np.count_nonzero(shifted))

            valid_exception_255 += int(
                np.count_nonzero(exception_fill)
            )

            invalid_mask_zero += int(
                np.count_nonzero(invalid_zero)
            )

            outside_mask_255 += int(
                np.count_nonzero(outside)
            )

            # -----------------------------------------------
            # Identify unexpected values
            # -----------------------------------------------

            for value in np.unique(mask):
                if int(value) not in (0, 1, 255):
                    other_mask_values[int(value)] += int(
                        np.count_nonzero(mask == value)
                    )

            for value in np.unique(exception):
                if int(value) not in (0, 64, 255):
                    other_exception_values[int(value)] += int(
                        np.count_nonzero(exception == value)
                    )

            # -----------------------------------------------
            # Spatial aggregation
            # -----------------------------------------------

            row_start = int(window.row_off)
            col_start = int(window.col_off)

            rows = np.arange(
                row_start,
                row_start + mask.shape[0]
            )

            cols = np.arange(
                col_start,
                col_start + mask.shape[1]
            )

            coarse_rows = (
                rows * grid_h // height
            )

            coarse_cols = (
                cols * grid_w // width
            )

            indices = (
                coarse_rows[:, None] * grid_w
                + coarse_cols[None, :]
            )

            flat_indices = indices.ravel()

            np.add.at(
                coarse_total.ravel(),
                flat_indices,
                1
            )

            np.add.at(
                coarse_valid.ravel(),
                flat_indices,
                valid.ravel().astype(np.uint64)
            )

            np.add.at(
                coarse_shift.ravel(),
                flat_indices,
                shifted.ravel().astype(np.uint64)
            )

        # ====================================================
        # 4. PRINT RESULTS
        # ====================================================

        print("\nMASK DISTRIBUTION")

        for value in sorted(mask_counts):

            count = mask_counts[value]

            print(
                f"Value {value}: "
                f"{count:,} "
                f"({percentage(count, total_pixels):.4f}%)"
            )

        print("\nEXCEPTION MASK DISTRIBUTION")

        for value in sorted(exception_counts):

            count = exception_counts[value]

            print(
                f"Value {value}: "
                f"{count:,} "
                f"({percentage(count, total_pixels):.4f}%)"
            )

        print("\nJOINT MASK DISTRIBUTION")

        print(
            f"{'MASK':<12}"
            f"{'EXCEPTION':<15}"
            f"{'PIXELS':>18}"
            f"{'PERCENT':>14}"
        )

        joint_rows = []

        for (m, e), count in sorted(joint_counts.items()):

            pct = percentage(count, total_pixels)

            print(
                f"{m:<12}"
                f"{e:<15}"
                f"{count:>18,}"
                f"{pct:>13.4f}%"
            )

            joint_rows.append([
                m,
                e,
                count,
                round(pct, 6)
            ])

        # ====================================================
        # 5. QUALITY SUMMARY
        # ====================================================

        valid_total = mask_counts.get(1, 0)

        summary = [
            ("Total pixels", total_pixels),
            ("Valid GCOV pixels", valid_total),
            ("Valid + no exception", valid_clean),
            ("Valid + SSAR shift", valid_shift),
            ("Valid + exception fill", valid_exception_255),
            ("Invalid mask zero", invalid_mask_zero),
            ("Outside acquisition", outside_mask_255),
        ]

        print("\nQUALITY SUMMARY")

        for label, count in summary:

            print(
                f"{label:<30}: "
                f"{count:>18,} "
                f"({percentage(count, total_pixels):.4f}%)"
            )

        if valid_total:

            print("\nWITHIN VALID GCOV PIXELS")

            print(
                "Clean percentage:",
                f"{percentage(valid_clean, valid_total):.4f}%"
            )

            print(
                "SSAR shift percentage:",
                f"{percentage(valid_shift, valid_total):.4f}%"
            )

            print(
                "Exception fill percentage:",
                f"{percentage(valid_exception_255, valid_total):.4f}%"
            )

        # ====================================================
        # 6. EXPORT JOINT DISTRIBUTION
        # ====================================================

        save_csv(
            date_output / "joint_mask_distribution.csv",
            [
                "mask_value",
                "exception_value",
                "pixel_count",
                "percentage_of_full_grid"
            ],
            joint_rows
        )

        # ====================================================
        # 7. EXPORT SUMMARY
        # ====================================================

        summary_rows = []

        for label, count in summary:

            summary_rows.append([
                label,
                count,
                round(percentage(count, total_pixels), 6)
            ])

        save_csv(
            date_output / "quality_summary.csv",
            ["metric", "pixel_count", "percentage_of_full_grid"],
            summary_rows
        )

        # ====================================================
        # 8. SPATIAL OVERVIEW MAPS
        # ====================================================

        valid_fraction = np.divide(
            coarse_valid,
            coarse_total,
            out=np.zeros_like(
                coarse_valid, dtype=np.float64
            ),
            where=coarse_total > 0
        )

        shift_fraction = np.divide(
            coarse_shift,
            coarse_valid,
            out=np.zeros_like(
                coarse_shift, dtype=np.float64
            ),
            where=coarse_valid > 0
        )

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6)
        )

        im1 = axes[0].imshow(
            valid_fraction * 100,
            cmap="viridis",
            vmin=0,
            vmax=100
        )

        axes[0].set_title(
            f"{date} - Valid GCOV Coverage (%)"
        )

        fig.colorbar(
            im1,
            ax=axes[0],
            label="Valid pixels (%)"
        )

        im2 = axes[1].imshow(
            shift_fraction * 100,
            cmap="magma",
            vmin=0,
            vmax=100
        )

        axes[1].set_title(
            f"{date} - SSAR Shift Among Valid (%)"
        )

        fig.colorbar(
            im2,
            ax=axes[1],
            label="Shift pixels / valid pixels (%)"
        )

        for ax in axes:
            ax.set_xlabel("Coarse column")
            ax.set_ylabel("Coarse row")

        plt.tight_layout()

        plt.savefig(
            date_output / "spatial_quality_overview.png",
            dpi=200
        )

        plt.close()

        print(
            f"\nOutputs saved to: {date_output}"
        )

        return {
            "date": date,
            "total": total_pixels,
            "valid": valid_total,
            "clean": valid_clean,
            "shift": valid_shift,
            "fill": valid_exception_255,
            "invalid_zero": invalid_mask_zero,
            "outside": outside_mask_255,
            "other_mask": dict(other_mask_values),
            "other_exception": dict(other_exception_values)
        }


# ============================================================
# 9. PROCESS ALL THREE ACQUISITIONS
# ============================================================

def main():

    results = []

    for date in DATES:

        result = process_date(date)

        results.append(result)

    # ========================================================
    # 10. CROSS-DATE COMPARISON
    # ========================================================

    print("\n" + "=" * 75)
    print("CROSS-DATE QUALITY COMPARISON")
    print("=" * 75)

    comparison_rows = []

    for result in results:

        total = result["total"]

        print(f"\n{result['date']}")

        print(
            "Valid:",
            f"{percentage(result['valid'], total):.4f}%"
        )

        print(
            "Valid + clean:",
            f"{percentage(result['clean'], total):.4f}%"
        )

        print(
            "Valid + SSAR shift:",
            f"{percentage(result['shift'], total):.4f}%"
        )

        comparison_rows.append([
            result["date"],
            total,
            result["valid"],
            percentage(result["valid"], total),
            result["clean"],
            percentage(result["clean"], total),
            result["shift"],
            percentage(result["shift"], total),
            result["fill"],
            percentage(result["fill"], total)
        ])

    save_csv(
        OUTPUT_DIR / "cross_date_quality_comparison.csv",
        [
            "date",
            "total_pixels",
            "valid_pixels",
            "valid_pct",
            "clean_pixels",
            "clean_pct",
            "shift_pixels",
            "shift_pct",
            "fill_pixels",
            "fill_pct"
        ],
        comparison_rows
    )

    print("\n" + "=" * 75)
    print("SCRIPT 26 COMPLETED")
    print("=" * 75)

    print(f"\nAll outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


"""
SCRIPT 27
Cross-Date SSAR Spatial Consistency Analysis

QGPT-WM | NISAR S-band GCOV

Acquisitions:
    August     : 20260819
    September  : 20260924
    October    : 20261018

Objectives:
1. Compare SSAR sample-shift locations across dates.
2. Calculate pairwise Jaccard similarity.
3. Identify persistent and date-specific shift pixels.
4. Calculate all eight temporal transition patterns.
5. Export full-resolution GeoTIFF products.
6. Generate CSV reports and a spatial overview.

Important:
- Uses original full-resolution masks.
- Restricts temporal comparison to common valid pixels.
- Does not modify original files.
- Does not discard SSAR-shift measurements.
"""

from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "full_scene_geotiffs"

OUTPUT_DIR = BASE_DIR / "ssar_temporal_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = [
    "20260819",
    "20260924",
    "20261018"
]

MASK_FILE = "mask.tif"
EXCEPTION_FILE = "inputDataExceptionMask.tif"

DATE_LABELS = [
    "August",
    "September",
    "October"
]

OVERVIEW_SIZE = 1000

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_csv(path, headers, rows):

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def safe_percentage(numerator, denominator):

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


def validate_datasets(datasets):

    reference = datasets[0]

    for ds in datasets[1:]:

        if ds.width != reference.width:
            raise ValueError("Raster width mismatch")

        if ds.height != reference.height:
            raise ValueError("Raster height mismatch")

        if ds.crs != reference.crs:
            raise ValueError("CRS mismatch")

        if ds.transform != reference.transform:
            raise ValueError("Geotransform mismatch")


def jaccard(intersection, union):

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# 3. OPEN ALL MASKS
# ============================================================

def main():

    print("=" * 75)
    print("SCRIPT 27: CROSS-DATE SSAR SPATIAL CONSISTENCY")
    print("=" * 75)

    mask_datasets = []
    exception_datasets = []

    for date in DATES:

        folder = INPUT_DIR / date

        mask_path = folder / MASK_FILE
        exception_path = folder / EXCEPTION_FILE

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        if not exception_path.exists():
            raise FileNotFoundError(exception_path)

        mask_datasets.append(
            rasterio.open(mask_path)
        )

        exception_datasets.append(
            rasterio.open(exception_path)
        )

    try:

        validate_datasets(mask_datasets)
        validate_datasets(exception_datasets)

        validate_datasets(
            [mask_datasets[0], exception_datasets[0]]
        )

        reference = mask_datasets[0]

        height = reference.height
        width = reference.width

        total_pixels = height * width

        print("\nSCENE INFORMATION")
        print(f"Dimensions: {width} x {height}")
        print(f"Total pixels: {total_pixels:,}")
        print(f"CRS: {reference.crs}")
        print(f"Resolution: {reference.res}")

        # ----------------------------------------------------
        # Check every mask/exception pair is aligned
        # ----------------------------------------------------

        for i in range(3):

            validate_datasets([
                mask_datasets[i],
                exception_datasets[i]
            ])

        # ====================================================
        # 4. OUTPUT GEOTIFF CONFIGURATION
        # ====================================================

        profile = reference.profile.copy()

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER"
        )

        persistent_path = (
            OUTPUT_DIR / "persistent_ssar_shift_3_dates.tif"
        )

        any_shift_path = (
            OUTPUT_DIR / "any_date_ssar_shift.tif"
        )

        transition_path = (
            OUTPUT_DIR / "ssar_temporal_transition_code.tif"
        )

        date_shift_paths = [
            OUTPUT_DIR / f"{date}_ssar_shift_mask.tif"
            for date in DATES
        ]

        # ====================================================
        # 5. STATISTICAL ACCUMULATORS
        # ====================================================

        common_valid_count = 0

        valid_counts = np.zeros(3, dtype=np.uint64)

        shift_counts = np.zeros(3, dtype=np.uint64)

        common_shift_counts = np.zeros(3, dtype=np.uint64)

        # Pairwise intersection and union
        pair_intersection = np.zeros(3, dtype=np.uint64)
        pair_union = np.zeros(3, dtype=np.uint64)

        # Pair order:
        # 0 = August / September
        # 1 = September / October
        # 2 = August / October

        pair_indices = [
            (0, 1),
            (1, 2),
            (0, 2)
        ]

        # Eight temporal patterns:
        # 0 = none
        # 1 = August only
        # 2 = September only
        # 3 = August + September
        # 4 = October only
        # 5 = August + October
        # 6 = September + October
        # 7 = all three

        transition_counts = np.zeros(8, dtype=np.uint64)

        # Validity patterns across dates
        common_validity_patterns = defaultdict(int)

        # Spatial overview accumulators
        grid_h = OVERVIEW_SIZE
        grid_w = OVERVIEW_SIZE

        coarse_valid = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        coarse_persistent = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        coarse_any = np.zeros(
            (grid_h, grid_w), dtype=np.uint64
        )

        # ====================================================
        # 6. WINDOWED FULL-SCENE PROCESSING
        # ====================================================

        print("\nPROCESSING FULL SCENE...")

        with rasterio.open(
            persistent_path, "w", **profile
        ) as persistent_ds, \
             rasterio.open(
                 any_shift_path, "w", **profile
             ) as any_ds, \
             rasterio.open(
                 transition_path, "w", **profile
             ) as transition_ds, \
             rasterio.open(
                 date_shift_paths[0], "w", **profile
             ) as aug_ds, \
             rasterio.open(
                 date_shift_paths[1], "w", **profile
             ) as sep_ds, \
             rasterio.open(
                 date_shift_paths[2], "w", **profile
             ) as oct_ds:

            output_date_datasets = [
                aug_ds, sep_ds, oct_ds
            ]

            for block_number, (_, window) in enumerate(
                reference.block_windows(1), start=1
            ):

                masks = [
                    ds.read(1, window=window)
                    for ds in mask_datasets
                ]

                exceptions = [
                    ds.read(1, window=window)
                    for ds in exception_datasets
                ]

                # --------------------------------------------
                # Validity masks
                # --------------------------------------------

                valid = [
                    masks[i] == 1
                    for i in range(3)
                ]

                # SSAR shift condition:
                # A valid GCOV pixel AND exception value 64.

                shifted = [
                    valid[i] & (exceptions[i] == 64)
                    for i in range(3)
                ]

                common_valid = (
                    valid[0] & valid[1] & valid[2]
                )

                # --------------------------------------------
                # Individual counts
                # --------------------------------------------

                for i in range(3):

                    valid_counts[i] += np.count_nonzero(
                        valid[i]
                    )

                    shift_counts[i] += np.count_nonzero(
                        shifted[i]
                    )

                    common_shift_counts[i] += np.count_nonzero(
                        shifted[i] & common_valid
                    )

                common_valid_count += np.count_nonzero(
                    common_valid
                )

                # --------------------------------------------
                # Pairwise overlap
                # Only inside common valid footprint.
                # --------------------------------------------

                for p, (i, j) in enumerate(pair_indices):

                    a = shifted[i] & common_valid
                    b = shifted[j] & common_valid

                    pair_intersection[p] += np.count_nonzero(
                        a & b
                    )

                    pair_union[p] += np.count_nonzero(
                        a | b
                    )

                # --------------------------------------------
                # Temporal transition encoding
                # --------------------------------------------

                transition = np.full(
                    masks[0].shape,
                    255,
                    dtype=np.uint8
                )

                code = (
                    shifted[0].astype(np.uint8)
                    + 2 * shifted[1].astype(np.uint8)
                    + 4 * shifted[2].astype(np.uint8)
                )

                transition[common_valid] = code[
                    common_valid
                ]

                for pattern in range(8):

                    transition_counts[pattern] += np.count_nonzero(
                        common_valid & (code == pattern)
                    )

                # --------------------------------------------
                # Persistent and any-date shift
                # --------------------------------------------

                persistent = (
                    shifted[0]
                    & shifted[1]
                    & shifted[2]
                )

                any_shift = (
                    shifted[0]
                    | shifted[1]
                    | shifted[2]
                )

                persistent_output = np.full(
                    masks[0].shape,
                    255,
                    dtype=np.uint8
                )

                any_output = np.full(
                    masks[0].shape,
                    255,
                    dtype=np.uint8
                )

                persistent_output[common_valid] = (
                    persistent[common_valid].astype(np.uint8)
                )

                any_output[common_valid] = (
                    any_shift[common_valid].astype(np.uint8)
                )

                persistent_ds.write(
                    persistent_output,
                    1,
                    window=window
                )

                any_ds.write(
                    any_output,
                    1,
                    window=window
                )

                transition_ds.write(
                    transition,
                    1,
                    window=window
                )

                # --------------------------------------------
                # Date-specific shift masks
                # --------------------------------------------

                for i in range(3):

                    date_output = np.full(
                        masks[i].shape,
                        255,
                        dtype=np.uint8
                    )

                    date_output[common_valid] = (
                        shifted[i][common_valid].astype(np.uint8)
                    )

                    output_date_datasets[i].write(
                        date_output,
                        1,
                        window=window
                    )

                # --------------------------------------------
                # Coarse spatial overview
                # --------------------------------------------

                row_start = int(window.row_off)
                col_start = int(window.col_off)

                rows = np.arange(
                    row_start,
                    row_start + masks[0].shape[0]
                )

                cols = np.arange(
                    col_start,
                    col_start + masks[0].shape[1]
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
                    coarse_valid.ravel(),
                    flat_indices,
                    common_valid.ravel().astype(np.uint64)
                )

                np.add.at(
                    coarse_persistent.ravel(),
                    flat_indices,
                    persistent.ravel().astype(np.uint64)
                )

                np.add.at(
                    coarse_any.ravel(),
                    flat_indices,
                    (any_shift & common_valid).ravel().astype(
                        np.uint64
                    )
                )

                if block_number % 500 == 0:

                    print(
                        f"Processed {block_number:,} blocks"
                    )

        # ====================================================
        # 7. PAIRWISE RESULTS
        # ====================================================

        print("\n" + "=" * 75)
        print("PAIRWISE SSAR SPATIAL AGREEMENT")
        print("=" * 75)

        pair_rows = []

        for p, (i, j) in enumerate(pair_indices):

            intersection = int(pair_intersection[p])
            union = int(pair_union[p])

            score = jaccard(intersection, union)

            print(
                f"\n{DATE_LABELS[i]} vs {DATE_LABELS[j]}"
            )

            print(f"Intersection: {intersection:,}")
            print(f"Union: {union:,}")
            print(f"Jaccard: {score:.8f}")
            print(f"Jaccard (%): {score * 100:.4f}%")

            pair_rows.append([
                DATE_LABELS[i],
                DATE_LABELS[j],
                intersection,
                union,
                score,
                score * 100
            ])

        save_csv(
            OUTPUT_DIR / "pairwise_jaccard.csv",
            [
                "date_1",
                "date_2",
                "intersection_pixels",
                "union_pixels",
                "jaccard_similarity",
                "jaccard_percentage"
            ],
            pair_rows
        )

        # ====================================================
        # 8. INDIVIDUAL AND COMMON FOOTPRINT STATISTICS
        # ====================================================

        print("\n" + "=" * 75)
        print("VALID FOOTPRINT AND SHIFT STATISTICS")
        print("=" * 75)

        footprint_rows = []

        for i in range(3):

            shift_percentage = safe_percentage(
                shift_counts[i],
                valid_counts[i]
            )

            common_shift_percentage = safe_percentage(
                common_shift_counts[i],
                common_valid_count
            )

            print(f"\n{DATE_LABELS[i]}")

            print(
                f"Valid pixels: {valid_counts[i]:,}"
            )

            print(
                f"Shift pixels: {shift_counts[i]:,}"
            )

            print(
                f"Shift among valid: {shift_percentage:.4f}%"
            )

            print(
                "Shift pixels inside common footprint:",
                f"{common_shift_counts[i]:,}"
            )

            footprint_rows.append([
                DATES[i],
                int(valid_counts[i]),
                int(shift_counts[i]),
                shift_percentage,
                int(common_shift_counts[i]),
                common_shift_percentage
            ])

        save_csv(
            OUTPUT_DIR / "footprint_shift_statistics.csv",
            [
                "date",
                "valid_pixels",
                "shift_pixels",
                "shift_pct_of_date_valid",
                "shift_pixels_common_footprint",
                "shift_pct_of_common_footprint"
            ],
            footprint_rows
        )

        print(
            "\nCommon valid footprint:",
            f"{common_valid_count:,}"
        )

        # ====================================================
        # 9. TEMPORAL TRANSITION STATISTICS
        # ====================================================

        print("\n" + "=" * 75)
        print("TEMPORAL TRANSITION ANALYSIS")
        print("=" * 75)

        pattern_names = {
            0: "No shift in any date",
            1: "August only",
            2: "September only",
            3: "August + September",
            4: "October only",
            5: "August + October",
            6: "September + October",
            7: "Persistent in all three dates"
        }

        transition_rows = []

        for pattern in range(8):

            count = int(transition_counts[pattern])

            pct = safe_percentage(
                count,
                common_valid_count
            )

            print(
                f"{pattern}: {pattern_names[pattern]:<35}"
                f"{count:>18,} ({pct:.6f}%)"
            )

            transition_rows.append([
                pattern,
                pattern_names[pattern],
                count,
                pct
            ])

        save_csv(
            OUTPUT_DIR / "temporal_transition_counts.csv",
            [
                "transition_code",
                "transition_name",
                "pixel_count",
                "percentage_of_common_valid"
            ],
            transition_rows
        )

        # ====================================================
        # 10. PERSISTENT AND ANY-DATE SUMMARY
        # ====================================================

        persistent_count = int(transition_counts[7])

        any_count = int(
            sum(transition_counts[1:])
        )

        print("\nPERSISTENT SHIFT PIXELS:")
        print(f"{persistent_count:,}")

        print(
            "Percentage of common valid footprint:",
            f"{safe_percentage(persistent_count, common_valid_count):.4f}%"
        )

        print("\nSHIFT IN AT LEAST ONE DATE:")
        print(f"{any_count:,}")

        # ====================================================
        # 11. SPATIAL OVERVIEW
        # ====================================================

        persistent_fraction = np.divide(
            coarse_persistent,
            coarse_valid,
            out=np.zeros_like(
                coarse_persistent, dtype=np.float64
            ),
            where=coarse_valid > 0
        )

        any_fraction = np.divide(
            coarse_any,
            coarse_valid,
            out=np.zeros_like(
                coarse_any, dtype=np.float64
            ),
            where=coarse_valid > 0
        )

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6)
        )

        im1 = axes[0].imshow(
            persistent_fraction * 100,
            cmap="magma",
            vmin=0,
            vmax=100
        )

        axes[0].set_title(
            "Persistent SSAR Shift - All Three Dates"
        )

        fig.colorbar(
            im1,
            ax=axes[0],
            label="Persistent shift (%)"
        )

        im2 = axes[1].imshow(
            any_fraction * 100,
            cmap="viridis",
            vmin=0,
            vmax=100
        )

        axes[1].set_title(
            "SSAR Shift - At Least One Date"
        )

        fig.colorbar(
            im2,
            ax=axes[1],
            label="Shift occurrence (%)"
        )

        for ax in axes:

            ax.set_xlabel("Coarse column")
            ax.set_ylabel("Coarse row")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "ssar_temporal_overview.png",
            dpi=200
        )

        plt.close()

        # ====================================================
        # 12. FINAL SUMMARY
        # ====================================================

        print("\n" + "=" * 75)
        print("SCRIPT 27 COMPLETED")
        print("=" * 75)

        print("\nOUTPUT DIRECTORY:")
        print(OUTPUT_DIR)

        print("\nGENERATED FILES:")

        for path in sorted(OUTPUT_DIR.iterdir()):

            print(path.name)

    finally:

        for ds in mask_datasets + exception_datasets:
            ds.close()


if __name__ == "__main__":
    main()

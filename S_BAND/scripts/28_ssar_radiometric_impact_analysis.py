
"""
SCRIPT 28
SSAR Radiometric Impact Analysis
QGPT-WM | NISAR S-band GCOV

Compares clean and SSAR-shift flagged pixels using:
    HHHH.tif
    HVHV.tif
    mask.tif
    inputDataExceptionMask.tif

Three acquisitions:
    20260819
    20260924
    20261018

Method:
- Full-scene windowed processing.
- Deterministic random sampling.
- Spatially stratified descriptive statistics.
- No source modification.
- No automatic pixel rejection or downweighting.

IMPORTANT:
The script does not assume that exception value 64
means radiometric corruption.
"""

from pathlib import Path
from collections import defaultdict

import csv
import json

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "full_scene_geotiffs"

OUTPUT_DIR = BASE_DIR / "ssar_radiometric_impact"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = [
    "20260819",
    "20260924",
    "20261018"
]

HH_FILE = "HHHH.tif"
HV_FILE = "HVHV.tif"
MASK_FILE = "mask.tif"
EXCEPTION_FILE = "inputDataExceptionMask.tif"

# Random sample probability.
# Increase only if runtime and memory permit.
SAMPLE_RATE = 0.001

RANDOM_SEED = 42

# Spatial stratification tile size in original pixels.
# At 10 m spacing, 1024 pixels = 10.24 km.
TILE_SIZE = 1024

# Minimum sampled observations for tile statistics.
MIN_TILE_SAMPLES = 20

# Ratio interpretation:
# "auto" = use raster metadata to identify units.
# "linear" = explicitly confirm values are linear power.
# "db" = explicitly confirm values are already in dB.
# "unknown" = do not calculate a ratio.
RATIO_MODE = "auto"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def metadata_text(dataset):

    parts = []

    for key, value in dataset.tags().items():
        parts.append(f"{key}={value}")

    for key, value in dataset.tags(1).items():
        parts.append(f"band_{key}={value}")

    return " | ".join(parts)


def determine_ratio_mode(text):

    mode = RATIO_MODE.lower()

    if mode in ("linear", "db", "unknown"):
        return mode

    lower = text.lower()

    if "decibel" in lower or "db" in lower:
        return "db"

    if (
        "linear" in lower
        or "gamma0" in lower
        or "gamma_0" in lower
        or "power" in lower
    ):
        return "linear"

    return "unknown"


def describe(values):

    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan
        }

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95))
    }


def add_statistics_rows(rows, date, feature, group, values):

    result = describe(values)

    rows.append([
        date,
        feature,
        group,
        result["count"],
        result["mean"],
        result["std"],
        result["median"],
        result["p05"],
        result["p25"],
        result["p75"],
        result["p95"]
    ])


def write_csv(path, headers, rows):

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def raster_valid(values, nodata):

    valid = np.isfinite(values)

    if nodata is not None:

        if np.isnan(nodata):
            valid &= ~np.isnan(values)
        else:
            valid &= values != nodata

    return valid


# ============================================================
# 3. PROCESS ONE ACQUISITION
# ============================================================

def process_date(date):

    print("\n" + "=" * 75)
    print(f"PROCESSING {date}")
    print("=" * 75)

    folder = INPUT_DIR / date

    paths = {
        "HH": folder / HH_FILE,
        "HV": folder / HV_FILE,
        "MASK": folder / MASK_FILE,
        "EXCEPTION": folder / EXCEPTION_FILE
    }

    for name, path in paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} file missing: {path}"
            )

    date_output = OUTPUT_DIR / date
    date_output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(
        RANDOM_SEED + int(date[-2:])
    )

    # Sample arrays by quality group.
    samples = {
        "clean": {
            "HH": [],
            "HV": []
        },
        "shift": {
            "HH": [],
            "HV": []
        }
    }

    # Spatially stratified samples:
    # key = (tile_row, tile_col, quality_group)
    tile_samples = defaultdict(
        lambda: {
            "HH": [],
            "HV": []
        }
    )

    total_valid = 0
    total_clean = 0
    total_shift = 0

    sampled_clean = 0
    sampled_shift = 0

    # --------------------------------------------------------
    # Open all four rasters
    # --------------------------------------------------------

    with rasterio.open(paths["HH"]) as hh_ds, \
         rasterio.open(paths["HV"]) as hv_ds, \
         rasterio.open(paths["MASK"]) as mask_ds, \
         rasterio.open(paths["EXCEPTION"]) as exception_ds:

        datasets = [
            hh_ds, hv_ds, mask_ds, exception_ds
        ]

        reference = hh_ds

        for ds in datasets[1:]:

            if ds.width != reference.width:
                raise ValueError("Raster width mismatch")

            if ds.height != reference.height:
                raise ValueError("Raster height mismatch")

            if ds.crs != reference.crs:
                raise ValueError("CRS mismatch")

            if ds.transform != reference.transform:
                raise ValueError("Geotransform mismatch")

        height = hh_ds.height
        width = hh_ds.width

        print(f"Dimensions: {width} x {height}")
        print(f"CRS: {hh_ds.crs}")
        print(f"Resolution: {hh_ds.res}")

        # ----------------------------------------------------
        # Inspect raster metadata
        # ----------------------------------------------------

        hh_metadata = metadata_text(hh_ds)
        hv_metadata = metadata_text(hv_ds)

        metadata = {
            "date": date,
            "HH_dtype": hh_ds.dtypes[0],
            "HV_dtype": hv_ds.dtypes[0],
            "HH_nodata": hh_ds.nodata,
            "HV_nodata": hv_ds.nodata,
            "HH_units": hh_ds.units,
            "HV_units": hv_ds.units,
            "HH_metadata": hh_metadata,
            "HV_metadata": hv_metadata,
            "CRS": str(hh_ds.crs),
            "resolution": list(hh_ds.res),
            "width": width,
            "height": height
        }

        with open(
            date_output / "raster_metadata.json", "w"
        ) as f:

            json.dump(metadata, f, indent=4, default=str)

        print("\nHH metadata:")
        print(hh_metadata or "No band tags available")

        print("\nHV metadata:")
        print(hv_metadata or "No band tags available")

        # Determine whether ratio calculation is justified.
        combined_metadata = (
            hh_metadata + " " + hv_metadata
            + " " + str(hh_ds.units)
            + " " + str(hv_ds.units)
        )

        ratio_mode = determine_ratio_mode(
            combined_metadata
        )

        print("\nRatio interpretation:", ratio_mode)

        # ----------------------------------------------------
        # Windowed processing
        # ----------------------------------------------------

        for block_number, (_, window) in enumerate(
            hh_ds.block_windows(1), start=1
        ):

            hh = hh_ds.read(1, window=window)
            hv = hv_ds.read(1, window=window)

            mask = mask_ds.read(1, window=window)

            exception = exception_ds.read(
                1, window=window
            )

            # Only fully valid GCOV samples.
            valid = (
                (mask == 1)
                & raster_valid(hh, hh_ds.nodata)
                & raster_valid(hv, hv_ds.nodata)
            )

            # Quality groups.
            clean = valid & (exception == 0)
            shifted = valid & (exception == 64)

            total_valid += np.count_nonzero(valid)
            total_clean += np.count_nonzero(clean)
            total_shift += np.count_nonzero(shifted)

            # ------------------------------------------------
            # Random sample selection
            # ------------------------------------------------

            random_values = rng.random(hh.shape)

            clean_sample = (
                clean & (random_values < SAMPLE_RATE)
            )

            shift_sample = (
                shifted & (random_values < SAMPLE_RATE)
            )

            # ------------------------------------------------
            # Global sample storage
            # ------------------------------------------------

            for group, selection in [
                ("clean", clean_sample),
                ("shift", shift_sample)
            ]:

                indices = np.flatnonzero(selection)

                if len(indices) == 0:
                    continue

                hh_values = hh.ravel()[indices].astype(
                    np.float64
                )

                hv_values = hv.ravel()[indices].astype(
                    np.float64
                )

                finite = (
                    np.isfinite(hh_values)
                    & np.isfinite(hv_values)
                )

                hh_values = hh_values[finite]
                hv_values = hv_values[finite]

                samples[group]["HH"].append(hh_values)
                samples[group]["HV"].append(hv_values)

                if group == "clean":
                    sampled_clean += len(hh_values)
                else:
                    sampled_shift += len(hh_values)

                # --------------------------------------------
                # Spatial tile assignment
                # --------------------------------------------

                local_rows = indices // hh.shape[1]
                local_cols = indices % hh.shape[1]

                global_rows = (
                    int(window.row_off) + local_rows
                )

                global_cols = (
                    int(window.col_off) + local_cols
                )

                tile_rows = global_rows // TILE_SIZE
                tile_cols = global_cols // TILE_SIZE

                tile_ids = (
                    tile_rows * ((width + TILE_SIZE - 1)
                                 // TILE_SIZE)
                    + tile_cols
                )

                tile_ids = tile_ids[finite]

                unique_tiles = np.unique(tile_ids)

                for tile_id in unique_tiles:

                    tile_selection = tile_ids == tile_id

                    tr = int(
                        tile_id // (
                            (width + TILE_SIZE - 1)
                            // TILE_SIZE
                        )
                    )

                    tc = int(
                        tile_id % (
                            (width + TILE_SIZE - 1)
                            // TILE_SIZE
                        )
                    )

                    key = (tr, tc, group)

                    tile_samples[key]["HH"].append(
                        hh_values[tile_selection]
                    )

                    tile_samples[key]["HV"].append(
                        hv_values[tile_selection]
                    )

            if block_number % 500 == 0:

                print(
                    f"Processed {block_number:,} blocks | "
                    f"Clean samples: {sampled_clean:,} | "
                    f"Shift samples: {sampled_shift:,}"
                )

    # ========================================================
    # 4. COMBINE SAMPLES
    # ========================================================

    for group in ("clean", "shift"):

        for feature in ("HH", "HV"):

            if samples[group][feature]:

                samples[group][feature] = np.concatenate(
                    samples[group][feature]
                )

            else:

                samples[group][feature] = np.array(
                    [], dtype=np.float64
                )

    print("\nEXACT VALID PIXEL COUNTS")

    print(f"Valid: {total_valid:,}")
    print(f"Clean: {total_clean:,}")
    print(f"SSAR shift: {total_shift:,}")

    print("\nRANDOM SAMPLE COUNTS")

    print(f"Clean samples: {sampled_clean:,}")
    print(f"Shift samples: {sampled_shift:,}")

    # ========================================================
    # 5. RADIOMETRIC FEATURE EXTRACTION
    # ========================================================

    clean_hh = samples["clean"]["HH"]
    shift_hh = samples["shift"]["HH"]

    clean_hv = samples["clean"]["HV"]
    shift_hv = samples["shift"]["HV"]

    feature_arrays = {
        "HH": {
            "clean": clean_hh,
            "shift": shift_hh
        },
        "HV": {
            "clean": clean_hv,
            "shift": shift_hv
        }
    }

    # Calculate HV/HH ratio only when units are known.
    if ratio_mode == "linear":

        clean_ratio_valid = (
            (clean_hh > 0) & (clean_hv > 0)
        )

        shift_ratio_valid = (
            (shift_hh > 0) & (shift_hv > 0)
        )

        clean_ratio = (
            clean_hv[clean_ratio_valid]
            / clean_hh[clean_ratio_valid]
        )

        shift_ratio = (
            shift_hv[shift_ratio_valid]
            / shift_hh[shift_ratio_valid]
        )

        feature_arrays["HV_HH_RATIO"] = {
            "clean": clean_ratio,
            "shift": shift_ratio
        }

        feature_arrays["HV_HH_RATIO_dB"] = {
            "clean": 10 * np.log10(clean_ratio),
            "shift": 10 * np.log10(shift_ratio)
        }

    elif ratio_mode == "db":

        feature_arrays["HV_minus_HH_dB"] = {
            "clean": clean_hv - clean_hh,
            "shift": shift_hv - shift_hh
        }

    else:

        print(
            "\nWARNING: Ratio skipped because raster units "
            "could not be established from metadata."
        )

    # ========================================================
    # 6. GLOBAL DESCRIPTIVE STATISTICS
    # ========================================================

    statistics_rows = []

    for feature, groups in feature_arrays.items():

        for group, values in groups.items():

            add_statistics_rows(
                statistics_rows,
                date,
                feature,
                group,
                values
            )

    write_csv(
        date_output / "global_radiometric_statistics.csv",
        [
            "date",
            "feature",
            "quality_group",
            "sample_count",
            "mean",
            "std",
            "median",
            "p05",
            "p25",
            "p75",
            "p95"
        ],
        statistics_rows
    )

    # ========================================================
    # 7. SPATIAL TILE STATISTICS
    # ========================================================

    tile_rows = []

    tile_keys = sorted({
        (key[0], key[1])
        for key in tile_samples
    })

    for tr, tc in tile_keys:

        for feature in ("HH", "HV"):

            group_values = {}

            for group in ("clean", "shift"):

                key = (tr, tc, group)

                arrays = tile_samples[key][feature]

                if arrays:
                    values = np.concatenate(arrays)
                else:
                    values = np.array([])

                group_values[group] = values

            clean_values = group_values["clean"]
            shift_values = group_values["shift"]

            clean_stats = describe(clean_values)
            shift_stats = describe(shift_values)

            if (
                clean_stats["count"] < MIN_TILE_SAMPLES
                or shift_stats["count"] < MIN_TILE_SAMPLES
            ):
                continue

            clean_median = clean_stats["median"]
            shift_median = shift_stats["median"]

            difference = shift_median - clean_median

            relative_difference = (
                difference / abs(clean_median) * 100
                if clean_median != 0 else np.nan
            )

            tile_rows.append([
                date,
                tr,
                tc,
                feature,
                clean_stats["count"],
                shift_stats["count"],
                clean_median,
                shift_median,
                difference,
                relative_difference
            ])

    write_csv(
        date_output / "spatial_tile_comparison.csv",
        [
            "date",
            "tile_row",
            "tile_col",
            "feature",
            "clean_sample_count",
            "shift_sample_count",
            "clean_median",
            "shift_median",
            "median_difference",
            "relative_median_difference_pct"
        ],
        tile_rows
    )

    # ========================================================
    # 8. HISTOGRAMS
    # ========================================================

    for feature, groups in feature_arrays.items():

        clean_values = groups["clean"]
        shift_values = groups["shift"]

        if (
            len(clean_values) == 0
            or len(shift_values) == 0
        ):
            continue

        combined = np.concatenate([
            clean_values,
            shift_values
        ])

        low, high = np.percentile(
            combined[np.isfinite(combined)],
            [1, 99]
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            clean_values,
            bins=100,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="Clean (exception 0)"
        )

        ax.hist(
            shift_values,
            bins=100,
            range=(low, high),
            density=True,
            alpha=0.55,
            label="SSAR shift (exception 64)"
        )

        ax.set_title(
            f"{date} - {feature}: Clean vs SSAR shift"
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()

        plt.savefig(
            date_output / f"{feature}_distribution.png",
            dpi=180
        )

        plt.close()

    print(f"\nCompleted {date}")
    print(f"Outputs: {date_output}")

    return {
        "date": date,
        "valid": total_valid,
        "clean": total_clean,
        "shift": total_shift,
        "sampled_clean": sampled_clean,
        "sampled_shift": sampled_shift,
        "ratio_mode": ratio_mode
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    results = []

    for date in DATES:

        results.append(
            process_date(date)
        )

    comparison_rows = []

    print("\n" + "=" * 75)
    print("CROSS-DATE RADIOMETRIC ANALYSIS SUMMARY")
    print("=" * 75)

    for result in results:

        print(f"\n{result['date']}")

        print(f"Valid: {result['valid']:,}")
        print(f"Clean: {result['clean']:,}")
        print(f"SSAR shift: {result['shift']:,}")

        print(
            f"Sampled clean: {result['sampled_clean']:,}"
        )

        print(
            f"Sampled shift: {result['sampled_shift']:,}"
        )

        print(
            f"Ratio mode: {result['ratio_mode']}"
        )

        comparison_rows.append([
            result["date"],
            result["valid"],
            result["clean"],
            result["shift"],
            result["sampled_clean"],
            result["sampled_shift"],
            result["ratio_mode"]
        ])

    write_csv(
        OUTPUT_DIR / "cross_date_radiometric_summary.csv",
        [
            "date",
            "valid_pixels",
            "clean_pixels",
            "shift_pixels",
            "sampled_clean",
            "sampled_shift",
            "ratio_mode"
        ],
        comparison_rows
    )

    print("\n" + "=" * 75)
    print("SCRIPT 28 COMPLETED")
    print("=" * 75)

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

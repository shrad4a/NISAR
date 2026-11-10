"""
SCRIPT 30
Calibrated Gamma0 / Sigma0 comparison and HHHV complex-term inspection

NISAR S-band L2 GCOV
Acquisitions:
    20260819
    20260924
    20261018

Processing:
    1. Read original HDF5 GCOV datasets.
    2. Calculate Sigma0 = Gamma0 * rtcGammaToSigmaFactor.
    3. Calculate HV/HH power ratio and dB ratio.
    4. Inspect HHHV real, imaginary, magnitude and phase.
    5. Compare clean and SSAR-shift-flagged pixels.
    6. Generate spatial tile statistics.
    7. Write full-scene GeoTIFF outputs using windowed processing.

IMPORTANT:
    - Does not discard SSAR-shift pixels.
    - Does not apply an arbitrary SSAR reliability penalty.
    - Gamma0 inputs are retained as the primary measurements.
    - Sigma0 is an additional representation.
"""

import os
import glob
import csv
import json
import warnings

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND"

HDF5_DIR = os.path.join(BASE_DIR, "RAW_HDF5")

GEOTIFF_DIR = os.path.join(BASE_DIR, "full_scene_geotiffs")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "calibrated_gamma_sigma_analysis"
)

DATES = [
    "20260819",
    "20260924",
    "20261018"
]

BLOCK_SIZE = 512

NODATA_VALUE = -9999.0

# Original NISAR HDF5 dataset paths
DATASET_PATHS = {
    "HHHH": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/HHHH"
    ),
    "HVHV": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/HVHV"
    ),
    "HHHV": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/HHHV"
    ),
    "RTC": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/rtcGammaToSigmaFactor"
    ),
    "MASK": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/mask"
    ),
    "EXCEPTION": (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/inputDataExceptionMask"
    )
}


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def find_hdf5(date):

    patterns = [
        os.path.join(HDF5_DIR, f"*{date}*.h5"),
        os.path.join(HDF5_DIR, f"*{date}*.hdf5")
    ]

    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No HDF5 file found for {date} in {HDF5_DIR}"
        )

    if len(matches) > 1:
        print(f"\nMultiple HDF5 files found for {date}:")
        for item in matches:
            print(" ", item)

        # Use the first matching product.
        # Change this manually if the folder contains duplicates.
        print("Using:", matches[0])

    return matches[0]


def get_template_raster(date):

    folder = os.path.join(GEOTIFF_DIR, date)

    candidates = [
        os.path.join(folder, "HHHH.tif"),
        os.path.join(folder, "HHHH.tiff")
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"HHHH GeoTIFF template not found for {date}: {folder}"
    )


def safe_median(array):

    if array.size == 0:
        return np.nan

    return float(np.median(array))


def write_output_raster(path, profile, description):

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA_VALUE,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER"
    )

    dataset = rasterio.open(
        path,
        "w",
        **output_profile
    )

    dataset.update_tags(
        DESCRIPTION=description,
        SOURCE="NISAR S-band L2 GCOV",
        PROCESSING_SCRIPT="30_calibrated_gamma_sigma_comparison.py"
    )

    return dataset


class RunningStats:

    def __init__(self):

        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf

    def update(self, values):

        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return

        self.count += values.size
        self.total += np.sum(values, dtype=np.float64)
        self.total_sq += np.sum(
            values.astype(np.float64) ** 2
        )

        self.minimum = min(
            self.minimum,
            float(np.min(values))
        )

        self.maximum = max(
            self.maximum,
            float(np.max(values))
        )

    def result(self):

        if self.count == 0:
            return {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan
            }

        mean = self.total / self.count

        variance = max(
            (self.total_sq / self.count) - mean ** 2,
            0.0
        )

        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "min": float(self.minimum),
            "max": float(self.maximum)
        }


# ============================================================
# 4. PROCESS EACH ACQUISITION
# ============================================================

global_summary = []
tile_summary = []
metadata_summary = []

for date in DATES:

    print("\n" + "=" * 70)
    print(f"PROCESSING ACQUISITION: {date}")
    print("=" * 70)

    date_output = os.path.join(OUTPUT_DIR, date)
    os.makedirs(date_output, exist_ok=True)

    hdf5_path = find_hdf5(date)
    template_path = get_template_raster(date)

    print("HDF5:", hdf5_path)
    print("GeoTIFF template:", template_path)

    # --------------------------------------------------------
    # Open the template GeoTIFF for geospatial metadata
    # --------------------------------------------------------

    with rasterio.open(template_path) as template:

        profile = template.profile.copy()

        height = template.height
        width = template.width

        transform = template.transform
        crs = template.crs

    print("Dimensions:", width, "x", height)
    print("CRS:", crs)
    print("Transform:", transform)

    # --------------------------------------------------------
    # Open original HDF5
    # --------------------------------------------------------

    h5 = h5py.File(hdf5_path, "r")

    for key, path in DATASET_PATHS.items():

        if path not in h5:
            h5.close()
            raise KeyError(
                f"Dataset {key} not found: {path}"
            )

    ds_hh = h5[DATASET_PATHS["HHHH"]]
    ds_hv = h5[DATASET_PATHS["HVHV"]]
    ds_hhv = h5[DATASET_PATHS["HHHV"]]
    ds_rtc = h5[DATASET_PATHS["RTC"]]
    ds_mask = h5[DATASET_PATHS["MASK"]]
    ds_exception = h5[DATASET_PATHS["EXCEPTION"]]

    expected_shape = (height, width)

    for name, dataset in [
        ("HHHH", ds_hh),
        ("HVHV", ds_hv),
        ("HHHV", ds_hhv),
        ("RTC", ds_rtc),
        ("MASK", ds_mask),
        ("EXCEPTION", ds_exception)
    ]:

        if dataset.shape != expected_shape:
            h5.close()
            raise ValueError(
                f"{name} shape mismatch: "
                f"{dataset.shape} != {expected_shape}"
            )

    # --------------------------------------------------------
    # Create output rasters
    # --------------------------------------------------------

    output_definitions = {
        "sigma0_HH.tif": (
            "Sigma0 HH = Gamma0 HH multiplied by RTC factor"
        ),
        "sigma0_HV.tif": (
            "Sigma0 HV = Gamma0 HV multiplied by RTC factor"
        ),
        "HV_HH_ratio_gamma_linear.tif": (
            "Linear HVHV / HHHH power ratio"
        ),
        "HV_HH_ratio_gamma_db.tif": (
            "10 * log10(HVHV / HHHH)"
        ),
        "HHHV_real.tif": (
            "Real component of complex HHHV covariance"
        ),
        "HHHV_imaginary.tif": (
            "Imaginary component of complex HHHV covariance"
        ),
        "HHHV_magnitude.tif": (
            "Magnitude of complex HHHV covariance"
        ),
        "HHHV_phase_radians.tif": (
            "Phase angle of complex HHHV covariance in radians"
        )
    }

    output_datasets = {}

    for filename, description in output_definitions.items():

        path = os.path.join(date_output, filename)

        output_datasets[filename] = write_output_raster(
            path,
            profile,
            description
        )

    # --------------------------------------------------------
    # Running statistics
    # --------------------------------------------------------

    metrics = [
        "Gamma0_HH",
        "Gamma0_HV",
        "Sigma0_HH",
        "Sigma0_HV",
        "HV_HH_ratio_linear",
        "HV_HH_ratio_dB",
        "HHHV_real",
        "HHHV_imaginary",
        "HHHV_magnitude",
        "HHHV_phase"
    ]

    stats = {
        group: {
            metric: RunningStats()
            for metric in metrics
        }
        for group in ["CLEAN", "SSAR_SHIFT"]
    }

    total_valid = 0
    total_clean = 0
    total_shift = 0
    total_invalid_radiometry = 0

    # --------------------------------------------------------
    # Windowed full-scene processing
    # --------------------------------------------------------

    for row in range(0, height, BLOCK_SIZE):

        for col in range(0, width, BLOCK_SIZE):

            block_height = min(
                BLOCK_SIZE,
                height - row
            )

            block_width = min(
                BLOCK_SIZE,
                width - col
            )

            window = Window(
                col,
                row,
                block_width,
                block_height
            )

            slices = (
                slice(row, row + block_height),
                slice(col, col + block_width)
            )

            # Read original HDF5 arrays
            hh = np.asarray(
                ds_hh[slices],
                dtype=np.float32
            )

            hv = np.asarray(
                ds_hv[slices],
                dtype=np.float32
            )

            rtc = np.asarray(
                ds_rtc[slices],
                dtype=np.float32
            )

            hhv = np.asarray(
                ds_hhv[slices],
                dtype=np.complex64
            )

            mask = np.asarray(
                ds_mask[slices]
            )

            exception = np.asarray(
                ds_exception[slices]
            )

            # ------------------------------------------------
            # Quality masks
            # ------------------------------------------------

            valid_mask = mask == 1

            clean_mask = (
                valid_mask &
                (exception == 0)
            )

            shift_mask = (
                valid_mask &
                (exception == 64)
            )

            finite_hh_hv = (
                np.isfinite(hh) &
                np.isfinite(hv)
            )

            valid_rtc = (
                np.isfinite(rtc) &
                (rtc > 0)
            )

            radiometric_valid = (
                valid_mask &
                finite_hh_hv
            )

            sigma_valid = (
                radiometric_valid &
                valid_rtc
            )

            ratio_valid = (
                radiometric_valid &
                (hh > 0) &
                (hv > 0)
            )

            complex_valid = (
                valid_mask &
                np.isfinite(hhv.real) &
                np.isfinite(hhv.imag)
            )

            total_valid += int(np.count_nonzero(valid_mask))

            total_clean += int(np.count_nonzero(clean_mask))
            total_shift += int(np.count_nonzero(shift_mask))

            total_invalid_radiometry += int(
                np.count_nonzero(
                    valid_mask & ~finite_hh_hv
                )
            )

            # ------------------------------------------------
            # Sigma0 conversion
            # ------------------------------------------------

            sigma_hh = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            sigma_hv = np.full(
                hv.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            sigma_hh[sigma_valid] = (
                hh[sigma_valid] * rtc[sigma_valid]
            )

            sigma_hv[sigma_valid] = (
                hv[sigma_valid] * rtc[sigma_valid]
            )

            # ------------------------------------------------
            # HV/HH ratio
            # ------------------------------------------------

            ratio = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            ratio_db = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            ratio[ratio_valid] = (
                hv[ratio_valid] / hh[ratio_valid]
            )

            positive_ratio = (
                ratio_valid &
                (ratio > 0)
            )

            ratio_db[positive_ratio] = (
                10.0 * np.log10(
                    ratio[positive_ratio]
                )
            )

            # ------------------------------------------------
            # HHHV complex-term decomposition
            # ------------------------------------------------

            hhv_real = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            hhv_imag = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            hhv_magnitude = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            hhv_phase = np.full(
                hh.shape,
                NODATA_VALUE,
                dtype=np.float32
            )

            hhv_real[complex_valid] = (
                hhv.real[complex_valid]
            )

            hhv_imag[complex_valid] = (
                hhv.imag[complex_valid]
            )

            hhv_magnitude[complex_valid] = (
                np.abs(hhv[complex_valid])
            )

            hhv_phase[complex_valid] = (
                np.angle(hhv[complex_valid])
            )

            # ------------------------------------------------
            # Write output windows
            # ------------------------------------------------

            output_arrays = {
                "sigma0_HH.tif": sigma_hh,
                "sigma0_HV.tif": sigma_hv,
                "HV_HH_ratio_gamma_linear.tif": ratio,
                "HV_HH_ratio_gamma_db.tif": ratio_db,
                "HHHV_real.tif": hhv_real,
                "HHHV_imaginary.tif": hhv_imag,
                "HHHV_magnitude.tif": hhv_magnitude,
                "HHHV_phase_radians.tif": hhv_phase
            }

            for filename, array in output_arrays.items():

                output_datasets[filename].write(
                    array,
                    1,
                    window=window
                )

            # ------------------------------------------------
            # Statistics by quality group
            # ------------------------------------------------

            group_masks = {
                "CLEAN": clean_mask,
                "SSAR_SHIFT": shift_mask
            }

            metric_arrays = {
                "Gamma0_HH": hh,
                "Gamma0_HV": hv,
                "Sigma0_HH": sigma_hh,
                "Sigma0_HV": sigma_hv,
                "HV_HH_ratio_linear": ratio,
                "HV_HH_ratio_dB": ratio_db,
                "HHHV_real": hhv_real,
                "HHHV_imaginary": hhv_imag,
                "HHHV_magnitude": hhv_magnitude,
                "HHHV_phase": hhv_phase
            }

            for group, group_mask in group_masks.items():

                for metric, array in metric_arrays.items():

                    if metric.startswith("Gamma0"):
                        selected_mask = (
                            group_mask &
                            np.isfinite(array)
                        )

                    elif metric.startswith("Sigma0"):
                        selected_mask = (
                            group_mask &
                            sigma_valid &
                            (array != NODATA_VALUE)
                        )

                    elif metric.startswith("HV_HH_ratio"):
                        selected_mask = (
                            group_mask &
                            (array != NODATA_VALUE) &
                            np.isfinite(array)
                        )

                    else:
                        selected_mask = (
                            group_mask &
                            complex_valid &
                            (array != NODATA_VALUE)
                        )

                    stats[group][metric].update(
                        array[selected_mask]
                    )

            # ------------------------------------------------
            # Spatial tile statistics
            # Each processing block is one spatial tile.
            # ------------------------------------------------

            for group, group_mask in group_masks.items():

                tile_record = {
                    "date": date,
                    "tile_row": row // BLOCK_SIZE,
                    "tile_col": col // BLOCK_SIZE,
                    "group": group,
                    "pixel_count": int(
                        np.count_nonzero(group_mask)
                    )
                }

                tile_metrics = {
                    "Gamma0_HH": hh,
                    "Gamma0_HV": hv,
                    "Sigma0_HH": sigma_hh,
                    "Sigma0_HV": sigma_hv,
                    "HV_HH_ratio_linear": ratio,
                    "HV_HH_ratio_dB": ratio_db
                }

                for metric, array in tile_metrics.items():

                    if metric.startswith("Sigma0"):
                        selected_mask = (
                            group_mask &
                            sigma_valid &
                            (array != NODATA_VALUE)
                        )

                    elif metric.startswith("HV_HH_ratio"):
                        selected_mask = (
                            group_mask &
                            (array != NODATA_VALUE) &
                            np.isfinite(array)
                        )

                    else:
                        selected_mask = (
                            group_mask &
                            np.isfinite(array)
                        )

                    tile_record[
                        metric + "_median"
                    ] = safe_median(
                        array[selected_mask]
                    )

                tile_summary.append(tile_record)

        print(
            f"{date}: processed rows "
            f"{min(row + BLOCK_SIZE, height)} / {height}"
        )

    # --------------------------------------------------------
    # Close output datasets and HDF5
    # --------------------------------------------------------

    for dataset in output_datasets.values():
        dataset.close()

    h5.close()

    # --------------------------------------------------------
    # Save global statistics
    # --------------------------------------------------------

    for group in ["CLEAN", "SSAR_SHIFT"]:

        for metric in metrics:

            result = stats[group][metric].result()

            global_summary.append({
                "date": date,
                "group": group,
                "metric": metric,
                **result
            })

    # --------------------------------------------------------
    # Metadata summary
    # --------------------------------------------------------

    metadata_summary.append({
        "date": date,
        "hdf5_file": os.path.basename(hdf5_path),
        "width": width,
        "height": height,
        "total_pixels": int(width * height),
        "valid_mask_pixels": total_valid,
        "clean_pixels": total_clean,
        "ssar_shift_pixels": total_shift,
        "valid_pixels_with_nonfinite_HH_or_HV": (
            total_invalid_radiometry
        ),
        "crs": str(crs),
        "resolution_x": float(transform.a),
        "resolution_y": float(abs(transform.e))
    })

    print("\nDATE SUMMARY:", date)
    print("Valid mask pixels:", f"{total_valid:,}")
    print("Clean pixels:", f"{total_clean:,}")
    print("SSAR-shift pixels:", f"{total_shift:,}")
    print(
        "Valid pixels with non-finite HH/HV:",
        f"{total_invalid_radiometry:,}"
    )

    print("Outputs written to:", date_output)


# ============================================================
# 5. EXPORT CSV FILES
# ============================================================

def write_csv(path, records):

    if not records:
        return

    fieldnames = []

    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "global_radiometric_statistics.csv"
    ),
    global_summary
)

write_csv(
    os.path.join(
        OUTPUT_DIR,
        "spatial_tile_statistics.csv"
    ),
    tile_summary
)

write_csv(
    os.path.join(
        OUTPUT_DIR,
        "acquisition_metadata_summary.csv"
    ),
    metadata_summary
)


# ============================================================
# 6. SAVE PROCESSING SUMMARY
# ============================================================

summary = {
    "script": "30_calibrated_gamma_sigma_comparison.py",
    "dates": DATES,
    "processing": {
        "sigma0_formula": (
            "Sigma0 = Gamma0 * rtcGammaToSigmaFactor"
        ),
        "ratio_formula": (
            "HV_HH_ratio = HVHV / HHHH"
        ),
        "ratio_db_formula": (
            "10 * log10(HVHV / HHHH)"
        ),
        "clean_definition": (
            "mask == 1 and inputDataExceptionMask == 0"
        ),
        "ssar_shift_definition": (
            "mask == 1 and inputDataExceptionMask == 64"
        ),
        "nodata_value": NODATA_VALUE,
        "block_size": BLOCK_SIZE
    },
    "important_notes": [
        "SSAR-shift pixels were retained.",
        "No arbitrary SSAR penalty was applied.",
        "Gamma0 was retained as the primary measurement.",
        "Sigma0 was calculated using multiplication by RTC factor.",
        "HHHV real and imaginary components were inspected separately.",
        "Phase is reported in radians."
    ]
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "processing_summary.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# 7. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("SCRIPT 30 COMPLETED")
print("=" * 70)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nGENERATED FILES:")

for root, dirs, files in os.walk(OUTPUT_DIR):

    for filename in files:
        print(os.path.relpath(
            os.path.join(root, filename),
            OUTPUT_DIR
        ))

print("\nMAIN CSV FILES:")
print("1. global_radiometric_statistics.csv")
print("2. spatial_tile_statistics.csv")
print("3. acquisition_metadata_summary.csv")

print("\nIMPORTANT:")
print(
    "Do not interpret differences between clean and SSAR-shift "
    "groups as causal evidence of SSAR corruption."
)

print("\nNEXT:")
print(
    "Review radiometric results before implementing "
    "Stage D local adaptive HH thresholding."
)
"""
04_extract_aoi.py

Extract the Bihar AOI from NISAR S-band GCOV HDF5 products.
Searches this script's folder and all subfolders for .h5 files.

Outputs:
  aoi_outputs/<acquisition>/<dataset>.npy
  aoi_outputs/<acquisition>/<dataset>.tif   (if rasterio is installed; real-valued layers)
  aoi_outputs/aoi_extraction_summary.csv
  aoi_outputs/aoi_extraction_summary.json

Install:
  python -m pip install h5py numpy pandas pyproj rasterio
"""

from pathlib import Path
from datetime import datetime
import json
import re
import warnings

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

# ------------------------------ SETTINGS -------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "aoi_outputs"

# Bihar AOI: west, south, east, north (WGS84 longitude/latitude)
WEST, SOUTH, EAST, NORTH = 86.3946, 25.4098, 86.6054, 25.5902
EPSG = 32645

GRID = "/science/SSAR/GCOV/grids/frequencyA"

# Extract these datasets. HHHV is complex and will be saved as NPY.
LAYERS = {
    "HHHH": f"{GRID}/HHHH",
    "HVHV": f"{GRID}/HVHV",
    "HHHV": f"{GRID}/HHHV",
    "mask": f"{GRID}/mask",
    "inputDataExceptionMask": f"{GRID}/inputDataExceptionMask",
    "numberOfLooks": f"{GRID}/numberOfLooks",
    "rtcGammaToSigmaFactor": f"{GRID}/rtcGammaToSigmaFactor",
}

# ------------------------------- HELPERS --------------------------------

def get_acquisition_name(filename):
    match = re.search(r"_(\d{8}T\d{6})_", filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
            return dt.strftime("%Y%m%d")
        except ValueError:
            pass
    return Path(filename).stem[:60]


def projected_aoi_bounds():
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{EPSG}", always_xy=True
    )
    # Densified transformation keeps bounds robust if the AOI is not tiny.
    xmin, ymin, xmax, ymax = transformer.transform_bounds(
        WEST, SOUTH, EAST, NORTH, densify_pts=21
    )
    return xmin, ymin, xmax, ymax


def select_coordinate_indices(coords, lower, upper):
    """Return indices whose pixel centres fall inside [lower, upper]."""
    coords = np.asarray(coords)
    selected = np.flatnonzero((coords >= lower) & (coords <= upper))
    if selected.size == 0:
        raise ValueError(
            f"No coordinate centres fall inside requested range {lower} to {upper}."
        )
    return int(selected.min()), int(selected.max()) + 1


def summarize_array(name, arr):
    result = {
        "dataset": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }

    if name in ("mask", "inputDataExceptionMask"):
        values, counts = np.unique(arr, return_counts=True)
        distribution = {str(int(v)): int(c) for v, c in zip(values, counts)}
        result["value_counts"] = distribution
        total = int(arr.size)

        if name == "mask":
            # Mask 255 means outside acquisition; mask 0 is not valid focused data.
            valid = int(np.count_nonzero((arr != 0) & (arr != 255)))
            result["valid_mask_pixels"] = valid
            result["valid_mask_percent_of_AOI"] = round(100 * valid / total, 3)
            result["outside_coverage_pixels"] = distribution.get("255", 0)
            result["mask_zero_pixels"] = distribution.get("0", 0)

        if name == "inputDataExceptionMask":
            in_coverage = arr != 255
            denominator = int(np.count_nonzero(in_coverage))
            code64 = int(np.count_nonzero(arr == 64))
            code0 = int(np.count_nonzero(arr == 0))
            result["code_0_no_anomaly_pixels"] = code0
            result["code_64_sample_shift_pixels"] = code64
            result["code_64_percent_of_in_coverage"] = (
                round(100 * code64 / denominator, 3) if denominator else None
            )
        return result

    if np.issubdtype(arr.dtype, np.complexfloating):
        values = np.abs(arr)
        result["note"] = "Statistics are for magnitude of complex values."
    else:
        values = arr

    finite = np.isfinite(values)
    finite_values = values[finite]
    result["finite_pixels"] = int(finite_values.size)
    result["nonfinite_pixels"] = int(values.size - finite_values.size)

    if finite_values.size:
        result["minimum"] = float(np.min(finite_values))
        result["maximum"] = float(np.max(finite_values))
        result["mean"] = float(np.mean(finite_values, dtype=np.float64))
        result["median"] = float(np.median(finite_values))
        result["p02"] = float(np.quantile(finite_values, 0.02))
        result["p98"] = float(np.quantile(finite_values, 0.98))

    return result


def save_geotiff_if_available(path, array, x_first_center, y_first_center,
                              x_spacing, y_spacing):
    """Write real-valued arrays as GeoTIFF when rasterio is installed."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return False

    if np.iscomplexobj(array):
        return False

    # Coordinates are pixel centres. Convert to upper-left pixel corner.
    pixel_width = abs(float(x_spacing))
    pixel_height = abs(float(y_spacing))
    left = float(x_first_center) - pixel_width / 2
    top = float(y_first_center) + pixel_height / 2
    transform = from_origin(left, top, pixel_width, pixel_height)

    output = np.asarray(array)
    if np.issubdtype(output.dtype, np.floating):
        output = output.astype(np.float32, copy=False)
        nodata = np.nan
    else:
        nodata = None

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype=output.dtype,
        crs=f"EPSG:{EPSG}",
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3 if np.issubdtype(output.dtype, np.floating) else 2,
        tiled=True,
    ) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, path.stem)

    return True


# -------------------------------- MAIN -----------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recursive search supports both:
    # ...\S_BAND\*.h5 and ...\S_BAND\S_BAND\*.h5
    h5_files = sorted(BASE_DIR.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 products found in {BASE_DIR} or its subfolders."
        )

    xmin, ymin, xmax, ymax = projected_aoi_bounds()
    print("AOI projected to EPSG:32645", flush=True)
    print(f"X: {xmin:.3f} to {xmax:.3f}", flush=True)
    print(f"Y: {ymin:.3f} to {ymax:.3f}", flush=True)
    print(f"Found {len(h5_files)} HDF5 file(s).", flush=True)

    summaries = []

    for h5_path in h5_files:
        print(f"\nProcessing: {h5_path.name}", flush=True)
        acquisition = get_acquisition_name(h5_path.name)
        product_dir = OUTPUT_DIR / acquisition
        product_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, "r") as h5:
            x_path = f"{GRID}/xCoordinates"
            y_path = f"{GRID}/yCoordinates"

            if x_path not in h5 or y_path not in h5:
                raise KeyError("xCoordinates or yCoordinates dataset is missing.")

            x = np.asarray(h5[x_path][()], dtype=np.float64)
            y = np.asarray(h5[y_path][()], dtype=np.float64)

            # Coordinate spacing and direction are read from actual arrays.
            dx_values = np.diff(x)
            dy_values = np.diff(y)
            dx = float(np.median(dx_values))
            dy = float(np.median(dy_values))

            col_start, col_end = select_coordinate_indices(x, xmin, xmax)
            row_start, row_end = select_coordinate_indices(y, ymin, ymax)

            # Ensure coordinate selections correspond to the raster dimensions.
            expected_rows = row_end - row_start
            expected_cols = col_end - col_start

            print(
                f"Window: rows {row_start}:{row_end}, "
                f"columns {col_start}:{col_end} "
                f"({expected_rows} x {expected_cols})",
                flush=True,
            )

            file_summary = {
                "filename": h5_path.name,
                "acquisition_date": acquisition,
                "source_file_size_GB": round(h5_path.stat().st_size / (1024 ** 3), 4),
                "crs_epsg": EPSG,
                "aoi_wgs84": [WEST, SOUTH, EAST, NORTH],
                "aoi_projected_bounds": [xmin, ymin, xmax, ymax],
                "source_raster_shape": [
                    int(y.size), int(x.size)
                ],
                "crop_window": {
                    "row_start": row_start,
                    "row_end_exclusive": row_end,
                    "col_start": col_start,
                    "col_end_exclusive": col_end,
                },
                "crop_shape": [expected_rows, expected_cols],
                "x_spacing_m": dx,
                "y_spacing_m": dy,
                "datasets": [],
            }

            for layer_name, dataset_path in LAYERS.items():
                if dataset_path not in h5:
                    warnings.warn(f"{layer_name} missing in {h5_path.name}; skipping.")
                    continue

                ds = h5[dataset_path]
                if ds.ndim != 2:
                    warnings.warn(f"{layer_name} is not 2D; skipping.")
                    continue

                if ds.shape != (y.size, x.size):
                    raise ValueError(
                        f"{layer_name} shape {ds.shape} does not match "
                        f"coordinate lengths {(y.size, x.size)}."
                    )

                # Read only the AOI window, not the full raster.
                crop = ds[row_start:row_end, col_start:col_end]

                npy_path = product_dir / f"{layer_name}.npy"
                np.save(npy_path, crop)

                # First selected coordinates are pixel centres.
                x_first = float(x[col_start])
                y_first = float(y[row_start])
                tif_path = product_dir / f"{layer_name}.tif"
                geotiff_written = save_geotiff_if_available(
                    tif_path, crop, x_first, y_first, dx, dy
                )

                layer_summary = summarize_array(layer_name, crop)
                layer_summary["npy_file"] = str(npy_path)
                layer_summary["geotiff_file"] = (
                    str(tif_path) if geotiff_written else None
                )
                file_summary["datasets"].append(layer_summary)

                print(
                    f"  Saved {layer_name}: {crop.shape}, "
                    f"GeoTIFF={'yes' if geotiff_written else 'no'}",
                    flush=True,
                )

        summaries.append(file_summary)

    json_path = OUTPUT_DIR / "aoi_extraction_summary.json"
    csv_path = OUTPUT_DIR / "aoi_extraction_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    rows = []
    for product in summaries:
        for dataset in product["datasets"]:
            rows.append({
                "filename": product["filename"],
                "acquisition_date": product["acquisition_date"],
                "dataset": dataset["dataset"],
                "crop_rows": product["crop_shape"][0],
                "crop_cols": product["crop_shape"][1],
                "dtype": dataset["dtype"],
                "finite_pixels": dataset.get("finite_pixels"),
                "nonfinite_pixels": dataset.get("nonfinite_pixels"),
                "valid_mask_pixels": dataset.get("valid_mask_pixels"),
                "valid_mask_percent_of_AOI": dataset.get("valid_mask_percent_of_AOI"),
                "code_0_no_anomaly_pixels": dataset.get("code_0_no_anomaly_pixels"),
                "code_64_sample_shift_pixels": dataset.get("code_64_sample_shift_pixels"),
                "code_64_percent_of_in_coverage": dataset.get("code_64_percent_of_in_coverage"),
                "minimum": dataset.get("minimum"),
                "maximum": dataset.get("maximum"),
                "mean": dataset.get("mean"),
                "median": dataset.get("median"),
                "p02": dataset.get("p02"),
                "p98": dataset.get("p98"),
                "npy_file": dataset.get("npy_file"),
                "geotiff_file": dataset.get("geotiff_file"),
                "value_counts": json.dumps(dataset.get("value_counts", {})),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nAOI extraction complete.", flush=True)
    print(f"Output folder: {OUTPUT_DIR}", flush=True)
    print(f"Summary JSON: {json_path}", flush=True)
    print(f"Summary CSV:  {csv_path}", flush=True)
    print(
        "\nIf GeoTIFF files were not created, install rasterio with:\n"
        "python -m pip install rasterio",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""
03_extract_sband_metadata.py

Extract metadata and chunked quality/backscatter statistics from NISAR S-band
GCOV HDF5 products. Does not load full raster datasets into memory.

Expected layout:
NISAR_Project/
    03_extract_sband_metadata.py
    S_BAND/
        *.h5

Outputs:
    sband_metadata_report.json
    sband_metadata_report.csv
    sband_dataset_statistics.csv
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

# ---------------------------- CONFIGURATION ----------------------------

BASE_DIR = Path(__file__).resolve().parent
H5_DIR = BASE_DIR  # Search this folder and its subfolders for the HDF5 products.
OUTPUT_DIR = BASE_DIR / "sband_metadata_output"

# AOI used in the project: west, south, east, north (longitude/latitude)
AOI_WGS84 = (86.3946, 25.4098, 86.6054, 25.5902)
TARGET_EPSG = 32645

# Raster processing controls. Each chunk is small relative to a full image.
CHUNK_ROWS = 512
SAMPLE_LIMIT_PER_DATASET = 2_000_000  # deterministic sampling cap for quantiles
RANDOM_SEED = 42

GRID = "/science/SSAR/GCOV/grids/frequencyA"
PARAMS = "/science/SSAR/GCOV/metadata/processingInformation/parameters"

DATASET_PATHS = {
    "HHHH": f"{GRID}/HHHH",
    "HHHV": f"{GRID}/HHHV",
    "HVHV": f"{GRID}/HVHV",
    "mask": f"{GRID}/mask",
    "inputDataExceptionMask": f"{GRID}/inputDataExceptionMask",
    "numberOfLooks": f"{GRID}/numberOfLooks",
    "rtcGammaToSigmaFactor": f"{GRID}/rtcGammaToSigmaFactor",
}

SCALAR_PATHS = {
    "x_spacing_dataset": f"{GRID}/xCoordinateSpacing",
    "y_spacing_dataset": f"{GRID}/yCoordinateSpacing",
    "projection_dataset": f"{GRID}/projection",
    "noiseCorrectionApplied": f"{PARAMS}/noiseCorrectionApplied",
    "radiometricTerrainCorrectionApplied": f"{PARAMS}/radiometricTerrainCorrectionApplied",
    "isFullCovariance": f"{PARAMS}/isFullCovariance",
    "polarimetricSymmetrizationApplied": f"{PARAMS}/polarimetricSymmetrizationApplied",
    "rfiMitigationApplied": f"{PARAMS}/rfiMitigationApplied",
    "backscatterConversionFormula":
        "/science/SSAR/GCOV/metadata/ceosAnalysisReadyData/outputBackscatterDecibelConversionFormula",
    "orbitType": "/science/SSAR/GCOV/metadata/orbit/orbitType",
}

# ------------------------------- HELPERS --------------------------------

def json_safe(value):
    """Convert HDF5 / NumPy values into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return json_safe(value.reshape(-1)[0])
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return str(value)


def read_scalar_or_missing(h5, path):
    if path not in h5:
        return None
    try:
        return json_safe(h5[path][()])
    except Exception as exc:
        return f"READ_ERROR: {exc}"


def get_attr(dataset, attr_name):
    if attr_name not in dataset.attrs:
        return None
    return json_safe(dataset.attrs[attr_name])


def parse_filename(filename):
    """Extract product identifiers and acquisition timestamp from NISAR name."""
    name = Path(filename).name
    # Example timestamp: 20260819T233810
    match = re.search(r"_(\d{8}T\d{6})_", name)
    acquisition = None
    if match:
        try:
            acquisition = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").isoformat()
        except ValueError:
            acquisition = match.group(1)

    parts = name.split("_")
    return {
        "filename": name,
        "acquisition_datetime_from_filename": acquisition,
        "mission": parts[0] if parts else None,
        "frequency": parts[1] if len(parts) > 1 else None,
        "product_type": parts[2] if len(parts) > 2 else None,
        "production_type": parts[3] if len(parts) > 3 else None,
        "track": parts[4] if len(parts) > 4 else None,
        "frame": parts[6] if len(parts) > 6 else None,
        "polarization_or_mode_tokens": "_".join(parts[7:]) if len(parts) > 7 else None,
    }


def coordinate_summary(h5, coordinate_name):
    path = f"{GRID}/{coordinate_name}"
    if path not in h5:
        return {"present": False}

    arr = np.asarray(h5[path][()], dtype=np.float64)
    if arr.size == 0:
        return {"present": True, "count": 0}

    diffs = np.diff(arr)
    finite_diffs = diffs[np.isfinite(diffs)]
    return {
        "present": True,
        "count": int(arr.size),
        "minimum": float(np.nanmin(arr)),
        "maximum": float(np.nanmax(arr)),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "monotonic_increasing": bool(np.all(finite_diffs > 0)) if finite_diffs.size else None,
        "monotonic_decreasing": bool(np.all(finite_diffs < 0)) if finite_diffs.size else None,
        "median_spacing": float(np.nanmedian(finite_diffs)) if finite_diffs.size else None,
        "minimum_spacing": float(np.nanmin(finite_diffs)) if finite_diffs.size else None,
        "maximum_spacing": float(np.nanmax(finite_diffs)) if finite_diffs.size else None,
    }


def transform_aoi_to_projected_bounds(epsg):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    west, south, east, north = AOI_WGS84
    corners = [
        transformer.transform(west, south),
        transformer.transform(west, north),
        transformer.transform(east, south),
        transformer.transform(east, north),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return {
        "xmin": float(min(xs)), "xmax": float(max(xs)),
        "ymin": float(min(ys)), "ymax": float(max(ys)),
    }


def dataset_chunk_statistics(ds, kind="numeric"):
    """
    Read a raster in row chunks.
    For numeric rasters, compute finite counts/min/max/mean and an approximate
    quantile sample. For categorical masks, count every encountered value.
    """
    if ds.ndim != 2:
        return {"status": "SKIPPED_NOT_2D", "shape": list(ds.shape)}

    rows, cols = ds.shape
    rng = np.random.default_rng(RANDOM_SEED)
    sample_parts = []
    sampled = 0
    total = 0
    finite_count = 0
    nonfinite_count = 0
    min_value = None
    max_value = None
    sum_value = 0.0
    value_counts = {}

    # Complex HHHV is not an intensity channel; summarize magnitude and components.
    complex_data = np.issubdtype(ds.dtype, np.complexfloating)

    for row_start in range(0, rows, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, rows)
        block = ds[row_start:row_end, :]
        total += int(block.size)

        if kind == "categorical":
            vals, counts = np.unique(block, return_counts=True)
            for val, count in zip(vals, counts):
                key = str(json_safe(val))
                value_counts[key] = value_counts.get(key, 0) + int(count)
            continue

        if complex_data:
            block_values = np.abs(block).astype(np.float64, copy=False)
        else:
            block_values = np.asarray(block, dtype=np.float64)

        finite = np.isfinite(block_values)
        finite_vals = block_values[finite]
        finite_count += int(finite_vals.size)
        nonfinite_count += int(block_values.size - finite_vals.size)

        if finite_vals.size:
            bmin = float(np.min(finite_vals))
            bmax = float(np.max(finite_vals))
            min_value = bmin if min_value is None else min(min_value, bmin)
            max_value = bmax if max_value is None else max(max_value, bmax)
            sum_value += float(np.sum(finite_vals, dtype=np.float64))

            # Reservoir-like random subsample, bounded in memory.
            remaining = SAMPLE_LIMIT_PER_DATASET - sampled
            if remaining > 0:
                if finite_vals.size <= remaining:
                    chosen = finite_vals
                else:
                    chosen = rng.choice(finite_vals, size=remaining, replace=False)
                sample_parts.append(chosen.astype(np.float32, copy=False))
                sampled += int(chosen.size)

    result = {
        "shape": [int(rows), int(cols)],
        "dtype": str(ds.dtype),
        "total_pixels": total,
        "finite_pixels": finite_count,
        "nonfinite_pixels": nonfinite_count,
    }

    if kind == "categorical":
        result["value_counts"] = value_counts
        result["valid_pixel_count_excluding_255"] = total - value_counts.get("255", 0)
        return result

    result["minimum"] = min_value
    result["maximum"] = max_value
    result["mean"] = (sum_value / finite_count) if finite_count else None
    result["quantile_sample_count"] = sampled
    if sample_parts:
        sample = np.concatenate(sample_parts)
        result["sample_quantiles"] = {
            "p02": float(np.quantile(sample, 0.02)),
            "p25": float(np.quantile(sample, 0.25)),
            "p50": float(np.quantile(sample, 0.50)),
            "p75": float(np.quantile(sample, 0.75)),
            "p98": float(np.quantile(sample, 0.98)),
        }
    if complex_data:
        result["note"] = "Statistics are for absolute magnitude of complex HHHV values."
    return result


def check_aoi_coverage(x_summary, y_summary, epsg):
    if not x_summary.get("present") or not y_summary.get("present"):
        return {"status": "COORDINATES_MISSING"}

    try:
        bounds = transform_aoi_to_projected_bounds(epsg)
    except Exception as exc:
        return {"status": "TRANSFORM_ERROR", "error": str(exc)}

    covers = (
        x_summary["minimum"] <= bounds["xmin"] and x_summary["maximum"] >= bounds["xmax"]
        and y_summary["minimum"] <= bounds["ymin"] and y_summary["maximum"] >= bounds["ymax"]
    )
    intersects = not (
        x_summary["maximum"] < bounds["xmin"] or x_summary["minimum"] > bounds["xmax"]
        or y_summary["maximum"] < bounds["ymin"] or y_summary["minimum"] > bounds["ymax"]
    )
    return {
        "aoi_projected_bounds": bounds,
        "grid_intersects_aoi": bool(intersects),
        "grid_fully_covers_aoi_bbox": bool(covers),
    }


# -------------------------------- MAIN -----------------------------------

def inspect_file(file_path):
    print(f"\nInspecting: {file_path.name}", flush=True)
    report = parse_filename(file_path.name)
    report["file_size_GB"] = round(file_path.stat().st_size / (1024 ** 3), 4)

    with h5py.File(file_path, "r") as h5:
        # Root-level identification attributes
        report["root_attributes"] = {
            key: json_safe(value) for key, value in h5.attrs.items()
        }

        # Scalar metadata and processing flags
        report["scalar_metadata"] = {
            key: read_scalar_or_missing(h5, path)
            for key, path in SCALAR_PATHS.items()
        }

        epsg_value = report["scalar_metadata"].get("projection_dataset")
        try:
            epsg = int(epsg_value)
        except (TypeError, ValueError):
            epsg = TARGET_EPSG  # fallback only for AOI diagnostic; report the actual missing value
        report["crs"] = {
            "epsg_code_from_product": epsg_value,
            "fallback_epsg_used_for_aoi_check": epsg if epsg_value is None else None,
            "expected_crs_name_if_epsg_32645": "WGS 84 / UTM zone 45N" if epsg == 32645 else None,
        }

        # Coordinate arrays are small compared with the raster grids.
        report["x_coordinates"] = coordinate_summary(h5, "xCoordinates")
        report["y_coordinates"] = coordinate_summary(h5, "yCoordinates")
        report["aoi_coverage_check"] = check_aoi_coverage(
            report["x_coordinates"], report["y_coordinates"], epsg
        )

        # Compare coordinate-derived spacing with scalar spacing datasets.
        report["coordinate_spacing_comparison"] = {
            "x_spacing_scalar": report["scalar_metadata"].get("x_spacing_dataset"),
            "x_spacing_from_coordinates": report["x_coordinates"].get("median_spacing"),
            "y_spacing_scalar": report["scalar_metadata"].get("y_spacing_dataset"),
            "y_spacing_from_coordinates": report["y_coordinates"].get("median_spacing"),
        }

        # Dataset attributes for interpretation.
        report["raster_dataset_attributes"] = {}
        for key in ("HHHH", "HHHV", "HVHV", "mask", "inputDataExceptionMask",
                    "numberOfLooks", "rtcGammaToSigmaFactor"):
            path = DATASET_PATHS[key]
            if path in h5:
                ds = h5[path]
                report["raster_dataset_attributes"][key] = {
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                    "units": get_attr(ds, "units"),
                    "description": get_attr(ds, "description"),
                    "fill_value": get_attr(ds, "_FillValue"),
                }
            else:
                report["raster_dataset_attributes"][key] = {"missing": True}

        # Chunked raster summaries.
        report["raster_statistics"] = {}
        for key in ("mask", "inputDataExceptionMask"):
            path = DATASET_PATHS[key]
            if path in h5:
                report["raster_statistics"][key] = dataset_chunk_statistics(
                    h5[path], kind="categorical"
                )
            else:
                report["raster_statistics"][key] = {"status": "MISSING"}

        for key in ("HHHH", "HVHV", "HHHV", "numberOfLooks", "rtcGammaToSigmaFactor"):
            path = DATASET_PATHS[key]
            if path in h5:
                report["raster_statistics"][key] = dataset_chunk_statistics(h5[path])
            else:
                report["raster_statistics"][key] = {"status": "MISSING"}

    print(f"Finished: {file_path.name}", flush=True)
    return report


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(H5_DIR.rglob("*.h5"))

    if not files:
        raise FileNotFoundError(
            f"No .h5 files found in: {H5_DIR} or any of its subfolders.\n"
            "Put the three NISAR S-band GCOV files in the S_BAND folder."
        )

    print(f"Found {len(files)} HDF5 file(s).", flush=True)
    reports = []

    for file_path in files:
        try:
            reports.append(inspect_file(file_path))
        except Exception as exc:
            warnings.warn(f"Could not fully inspect {file_path.name}: {exc}")
            reports.append({
                **parse_filename(file_path.name),
                "error": str(exc),
            })

    json_path = OUTPUT_DIR / "sband_metadata_report.json"
    csv_path = OUTPUT_DIR / "sband_metadata_report.csv"
    stats_path = OUTPUT_DIR / "sband_dataset_statistics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    # Compact one-row-per-product CSV; nested objects are JSON strings.
    flat_rows = []
    stats_rows = []
    for report in reports:
        flat = {
            "filename": report.get("filename"),
            "acquisition_datetime": report.get("acquisition_datetime_from_filename"),
            "frequency": report.get("frequency"),
            "product_type": report.get("product_type"),
            "production_type": report.get("production_type"),
            "track": report.get("track"),
            "frame": report.get("frame"),
            "file_size_GB": report.get("file_size_GB"),
            "projection_EPSG": report.get("crs", {}).get("epsg_code_from_product"),
            "x_min": report.get("x_coordinates", {}).get("minimum"),
            "x_max": report.get("x_coordinates", {}).get("maximum"),
            "x_spacing": report.get("coordinate_spacing_comparison", {}).get("x_spacing_from_coordinates"),
            "y_min": report.get("y_coordinates", {}).get("minimum"),
            "y_max": report.get("y_coordinates", {}).get("maximum"),
            "y_spacing": report.get("coordinate_spacing_comparison", {}).get("y_spacing_from_coordinates"),
            "grid_intersects_AOI": report.get("aoi_coverage_check", {}).get("grid_intersects_aoi"),
            "grid_covers_AOI": report.get("aoi_coverage_check", {}).get("grid_fully_covers_aoi_bbox"),
            "noiseCorrectionApplied": report.get("scalar_metadata", {}).get("noiseCorrectionApplied"),
            "radiometricTerrainCorrectionApplied": report.get("scalar_metadata", {}).get("radiometricTerrainCorrectionApplied"),
            "isFullCovariance": report.get("scalar_metadata", {}).get("isFullCovariance"),
            "polarimetricSymmetrizationApplied": report.get("scalar_metadata", {}).get("polarimetricSymmetrizationApplied"),
            "rfiMitigationApplied": report.get("scalar_metadata", {}).get("rfiMitigationApplied"),
            "backscatterConversionFormula": report.get("scalar_metadata", {}).get("backscatterConversionFormula"),
            "orbitType": report.get("scalar_metadata", {}).get("orbitType"),
            "error": report.get("error"),
        }
        flat_rows.append(flat)

        for dataset_name, stats in report.get("raster_statistics", {}).items():
            stats_rows.append({
                "filename": report.get("filename"),
                "acquisition_datetime": report.get("acquisition_datetime_from_filename"),
                "dataset": dataset_name,
                **{
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in stats.items()
                },
            })

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    print("\nExtraction complete.", flush=True)
    print(f"Metadata JSON: {json_path}", flush=True)
    print(f"Metadata CSV:  {csv_path}", flush=True)
    print(f"Dataset stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()

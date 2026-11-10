"""
06_compare_product_metadata.py

Compare metadata and storage properties for the three NISAR S-band GCOV HDF5
products used in the AOI analysis.

The script:
- Finds .h5 files recursively below the folder containing this script.
- Identifies the three AOI acquisition dates from filenames.
- Compares important GCOV datasets and their attributes.
- Compares product-level attributes and relevant metadata values.
- Writes CSV and JSON reports. It does not modify the HDF5 products.

Run:
    python 06_compare_product_metadata.py

Dependencies:
    python -m pip install h5py numpy pandas
"""

from pathlib import Path
from datetime import datetime, date
import json
import re

import h5py
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "product_comparison_output"

# The three acquisition dates already extracted into aoi_outputs.
TARGET_DATES = {"20260819", "20260924", "20261018"}

# Core datasets relevant to the current HV anomaly investigation.
DATASET_PATHS = [
    "/science/SSAR/GCOV/grids/frequencyA/HHHH",
    "/science/SSAR/GCOV/grids/frequencyA/HVHV",
    "/science/SSAR/GCOV/grids/frequencyA/HHHV",
    "/science/SSAR/GCOV/grids/frequencyA/mask",
    "/science/SSAR/GCOV/grids/frequencyA/inputDataExceptionMask",
    "/science/SSAR/GCOV/grids/frequencyA/numberOfLooks",
    "/science/SSAR/GCOV/grids/frequencyA/rtcGammaToSigmaFactor",
    "/science/SSAR/GCOV/grids/frequencyA/xCoordinates",
    "/science/SSAR/GCOV/grids/frequencyA/yCoordinates",
    "/science/SSAR/GCOV/grids/frequencyA/projection",
]

# Attributes worth highlighting if present at product or dataset level.
IMPORTANT_ATTRIBUTE_TERMS = (
    "unit", "scale", "offset", "fill", "valid", "missing",
    "calibr", "radiometric", "terrain", "noise", "gamma", "sigma",
    "polar", "frequency", "look", "exception", "rfi", "orbit",
    "product", "processing", "algorithm", "version", "conversion",
    "decibel", "dn", "covariance",
)


def decode_value(value):
    """Convert HDF5/numpy values into JSON-safe Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 100:
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample": [decode_value(x) for x in value.flat[:10]],
            }
        return [decode_value(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (list, tuple)):
        return [decode_value(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def attr_dict(obj):
    return {str(k): decode_value(v) for k, v in obj.attrs.items()}


def filename_date(path):
    match = re.search(r"_(20\d{6})T", path.name)
    return match.group(1) if match else None


def find_target_files():
    found = {}
    for path in BASE_DIR.rglob("*.h5"):
        d = filename_date(path)
        if d in TARGET_DATES:
            found.setdefault(d, []).append(path)

    selected = {}
    for d, paths in found.items():
        # If duplicate copies exist, prefer the shortest path/name and report
        # all candidates so the selection is transparent.
        paths = sorted(paths, key=lambda p: (len(str(p)), p.name))
        selected[d] = paths[0]
        if len(paths) > 1:
            print(f"WARNING: {d}: found {len(paths)} matching HDF5 files:")
            for p in paths:
                print(f"  {p}")
            print(f"  Using: {paths[0]}")
    return selected


def relevant_attributes(attrs):
    result = {}
    for key, value in attrs.items():
        key_lower = key.lower()
        if any(term in key_lower for term in IMPORTANT_ATTRIBUTE_TERMS):
            result[key] = value
    return result


def summarize_dataset(h5, dataset_path, scene_date):
    if dataset_path not in h5:
        return {
            "date": scene_date,
            "dataset_path": dataset_path,
            "exists": False,
        }

    ds = h5[dataset_path]
    attrs = attr_dict(ds)
    item = {
        "date": scene_date,
        "dataset_path": dataset_path,
        "exists": True,
        "shape": list(ds.shape),
        "dtype": str(ds.dtype),
        "ndim": int(ds.ndim),
        "size": int(ds.size),
        "chunks": list(ds.chunks) if ds.chunks else None,
        "compression": ds.compression,
        "compression_opts": decode_value(ds.compression_opts),
        "shuffle": bool(ds.shuffle),
        "fletcher32": bool(ds.fletcher32),
        "scaleoffset": decode_value(ds.scaleoffset),
        "fillvalue": decode_value(ds.fillvalue),
        "attributes": attrs,
        "relevant_attributes": relevant_attributes(attrs),
    }

    # Coordinate datasets are small enough to summarize directly.
    if dataset_path.endswith(("xCoordinates", "yCoordinates")):
        arr = np.asarray(ds[()])
        if arr.size:
            item["coordinate_summary"] = {
                "first": float(arr.flat[0]),
                "last": float(arr.flat[-1]),
                "count": int(arr.size),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "median_step": float(np.nanmedian(np.diff(arr.astype(np.float64))))
                if arr.size > 1 else None,
            }

    # Projection is typically a scalar dataset.
    if dataset_path.endswith("/projection"):
        try:
            item["projection_value"] = decode_value(ds[()])
        except Exception as exc:
            item["projection_read_error"] = str(exc)

    return item


def collect_file_report(scene_date, path):
    report = {
        "date": scene_date,
        "file_name": path.name,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_size_gb": round(path.stat().st_size / (1024 ** 3), 4),
        "root_attributes": {},
        "metadata_attributes": {},
        "datasets": [],
        "groups": [],
    }

    with h5py.File(path, "r") as h5:
        report["root_attributes"] = attr_dict(h5)

        # Capture attributes on metadata groups as well as root attributes.
        metadata_candidates = [
            "/science/SSAR/GCOV",
            "/science/SSAR/GCOV/grids",
            "/science/SSAR/GCOV/grids/frequencyA",
            "/science/LSAR",
            "/science",
        ]
        for group_path in metadata_candidates:
            if group_path in h5 and isinstance(h5[group_path], h5py.Group):
                report["metadata_attributes"][group_path] = attr_dict(h5[group_path])

        for dataset_path in DATASET_PATHS:
            report["datasets"].append(
                summarize_dataset(h5, dataset_path, scene_date)
            )

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                # Record group paths and only non-empty group attributes.
                attrs = attr_dict(obj)
                if attrs:
                    report["groups"].append({
                        "path": "/" + name if name else "/",
                        "attributes": attrs,
                    })

        h5.visititems(visit)

    return report


def flatten_attribute_rows(reports):
    rows = []

    for report in reports:
        d = report["date"]

        for key, value in report["root_attributes"].items():
            rows.append({
                "date": d,
                "scope": "ROOT",
                "object_path": "/",
                "attribute": key,
                "value": json.dumps(value, ensure_ascii=False, sort_keys=True),
            })

        for group_path, attrs in report["metadata_attributes"].items():
            for key, value in attrs.items():
                rows.append({
                    "date": d,
                    "scope": "GROUP",
                    "object_path": group_path,
                    "attribute": key,
                    "value": json.dumps(value, ensure_ascii=False, sort_keys=True),
                })

        for ds in report["datasets"]:
            if not ds.get("exists"):
                continue
            for key, value in ds["attributes"].items():
                rows.append({
                    "date": d,
                    "scope": "DATASET",
                    "object_path": ds["dataset_path"],
                    "attribute": key,
                    "value": json.dumps(value, ensure_ascii=False, sort_keys=True),
                })

    return rows


def flatten_dataset_rows(reports):
    rows = []
    for report in reports:
        for ds in report["datasets"]:
            row = {
                "date": report["date"],
                "file_name": report["file_name"],
                "file_size_gb": report["file_size_gb"],
                **{k: v for k, v in ds.items()
                   if k not in ("attributes", "relevant_attributes")},
            }
            row["relevant_attributes_json"] = json.dumps(
                ds.get("relevant_attributes", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            rows.append(row)
    return rows



def json_fallback(value):
    """Final safety net for values not normalized by decode_value()."""
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return decode_value(value)
    if isinstance(value, np.ndarray):
        return decode_value(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = find_target_files()
    missing = sorted(TARGET_DATES - set(selected))
    if missing:
        print("WARNING: Could not find HDF5 products for:", ", ".join(missing))

    if not selected:
        raise FileNotFoundError(
            "No target HDF5 products found. Check that the three .h5 files are "
            "inside this script's folder or one of its subfolders."
        )

    reports = []
    for scene_date in sorted(selected):
        path = selected[scene_date]
        print(f"Inspecting {scene_date}: {path.name}")
        reports.append(collect_file_report(scene_date, path))

    json_path = OUTPUT_DIR / "product_metadata_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=json_fallback)

    attr_df = pd.DataFrame(flatten_attribute_rows(reports))
    attr_path = OUTPUT_DIR / "all_product_attributes.csv"
    attr_df.to_csv(attr_path, index=False)

    ds_df = pd.DataFrame(flatten_dataset_rows(reports))
    ds_path = OUTPUT_DIR / "dataset_storage_comparison.csv"
    ds_df.to_csv(ds_path, index=False)

    # Build a compact table of important root/group/dataset attributes.
    # Values remain literal strings so differences are easy to filter in Excel.
    important_rows = []
    for row in flatten_attribute_rows(reports):
        name = row["attribute"].lower()
        if any(term in name for term in IMPORTANT_ATTRIBUTE_TERMS):
            important_rows.append(row)

    important_df = pd.DataFrame(important_rows)
    important_path = OUTPUT_DIR / "important_attributes_comparison.csv"
    important_df.to_csv(important_path, index=False)

    print("\\n=== DATASET STORAGE SUMMARY ===")
    columns = [
        "date", "dataset_path", "exists", "shape", "dtype",
        "fillvalue", "compression", "chunks",
    ]
    available_columns = [c for c in columns if c in ds_df.columns]
    print(ds_df[available_columns].to_string(index=False))

    print("\\n=== PRODUCT-LEVEL ATTRIBUTES ===")
    root_df = attr_df[attr_df["scope"] == "ROOT"]
    if root_df.empty:
        print("No root attributes found.")
    else:
        pivot = root_df.pivot_table(
            index="attribute", columns="date", values="value",
            aggfunc="first", dropna=False
        )
        print(pivot.to_string())

    print("\\nReports saved:")
    print(json_path)
    print(attr_path)
    print(ds_path)
    print(important_path)
    print(
        "\\nThis is a metadata comparison only. It does not establish that the "
        "September HVHV values are incorrectly calibrated. We will use the "
        "reported differences to decide the next diagnostic."
    )


if __name__ == "__main__":
    main()

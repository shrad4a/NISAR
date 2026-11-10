
"""
SCRIPT 29
Original NISAR S-band GCOV Radiometric Metadata Inspection

QGPT-WM Research Pipeline

Purpose:
1. Inspect original HDF5 GCOV datasets.
2. Identify HH, HV and covariance elements.
3. Extract original metadata and calibration information.
4. Identify radiometric quantities and units.
5. Inspect fill values and valid ranges.
6. Calculate sampled numerical statistics.
7. Export complete metadata reports.

IMPORTANT:
- Original HDF5 files are read-only.
- Large datasets are never loaded entirely into RAM.
- Numerical statistics are sampled, not exact full-scene statistics.
- Gamma0/Sigma0 classification is based on metadata evidence,
  not assumptions based on dataset names.
"""

from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import json
import csv
import re
import traceback


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "RAW_HDF5"

OUTPUT_DIR = BASE_DIR / "original_gcov_metadata_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling configuration
SAMPLE_MAX_DIM = 1500
SAMPLE_SEED = 42

# Acquisition identifiers
DATES = [
    "20260819",
    "20260924",
    "20261018"
]

# Search terms for radiometric and calibration metadata
CALIBRATION_TERMS = [
    "gamma",
    "sigma",
    "beta",
    "calib",
    "radiometric",
    "backscatter",
    "normalization",
    "incidence",
    "noise",
    "power",
    "scaling",
    "scale_factor",
    "offset",
    "units",
    "linear",
    "decibel",
    "db",
    "reference"
]

# ============================================================
# 2. SERIALIZATION HELPERS
# ============================================================

def convert_value(value):

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):

        if value.size > 1000:
            return {
                "array_shape": list(value.shape),
                "dtype": str(value.dtype),
                "note": "Array omitted because it is large"
            }

        return [
            convert_value(x)
            for x in value.tolist()
        ]

    if isinstance(value, np.generic):
        return convert_value(value.item())

    if isinstance(value, (list, tuple)):
        return [convert_value(x) for x in value]

    if isinstance(value, dict):
        return {
            str(k): convert_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def extract_attributes(obj):

    result = {}

    for key, value in obj.attrs.items():

        try:
            result[str(key)] = convert_value(value)

        except Exception as exc:

            result[str(key)] = (
                f"Could not decode attribute: {exc}"
            )

    return result


# ============================================================
# 3. DATASET IDENTIFICATION
# ============================================================

def normalize_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        name.lower()
    )


def identify_dataset(path):

    name = normalize_name(
        path.split("/")[-1]
    )

    # Exact GCOV polarization layer names
    known = {
        "hhhh": "HH_POWER_OR_COVARIANCE_DIAGONAL",
        "hvhv": "HV_POWER_OR_COVARIANCE_DIAGONAL",
        "hhhv": "HH_HV_CROSS_COVARIANCE"
    }

    if name in known:
        return known[name]

    if name in ("hh", "hhpower", "hhbackscatter"):
        return "HH_MEASUREMENT"

    if name in ("hv", "hvpower", "hvbackscatter"):
        return "HV_MEASUREMENT"

    if "covariance" in name or "cov" in name:
        return "OTHER_COVARIANCE_DATASET"

    if "gamma" in name:
        return "GAMMA_RELATED_DATASET"

    if "sigma" in name:
        return "SIGMA_RELATED_DATASET"

    return None


# ============================================================
# 4. METADATA KEYWORD SEARCH
# ============================================================

def search_metadata_keywords(path, attributes):

    matches = {}

    combined = (
        path + " " +
        " ".join(attributes.keys()) + " " +
        " ".join(str(v) for v in attributes.values())
    ).lower()

    for term in CALIBRATION_TERMS:

        if term in combined:
            matches[term] = True

    return matches


# ============================================================
# 5. DATASET NUMERICAL SAMPLING
# ============================================================

def sample_dataset(dataset):

    result = {
        "sampled": False,
        "sample_method": "None",
        "sample_count": 0,
        "finite_count": 0,
        "nan_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
        "std": None,
        "p01": None,
        "p05": None,
        "p25": None,
        "p75": None,
        "p95": None,
        "p99": None,
        "positive_fraction": None,
        "zero_fraction": None,
        "negative_fraction": None
    }

    if dataset.ndim != 2:
        result["sample_method"] = (
            "Skipped: dataset is not two-dimensional"
        )
        return result

    height, width = dataset.shape

    if height == 0 or width == 0:
        return result

    stride_y = max(
        1,
        int(np.ceil(height / SAMPLE_MAX_DIM))
    )

    stride_x = max(
        1,
        int(np.ceil(width / SAMPLE_MAX_DIM))
    )

    try:

        # Deterministic regular spatial sampling.
        values = dataset[
            ::stride_y,
            ::stride_x
        ]

        values = np.asarray(
            values,
            dtype=np.float64
        ).ravel()

        result["sampled"] = True

        result["sample_method"] = (
            f"Regular grid sampling; "
            f"row stride={stride_y}, "
            f"column stride={stride_x}"
        )

        result["sample_count"] = int(values.size)

        result["nan_count"] = int(
            np.count_nonzero(np.isnan(values))
        )

        result["positive_infinity_count"] = int(
            np.count_nonzero(np.isposinf(values))
        )

        result["negative_infinity_count"] = int(
            np.count_nonzero(np.isneginf(values))
        )

        finite = values[np.isfinite(values)]

        result["finite_count"] = int(finite.size)

        if finite.size == 0:
            return result

        result["minimum"] = float(np.min(finite))
        result["maximum"] = float(np.max(finite))
        result["mean"] = float(np.mean(finite))
        result["median"] = float(np.median(finite))
        result["std"] = float(np.std(finite))

        percentiles = np.percentile(
            finite,
            [1, 5, 25, 75, 95, 99]
        )

        result["p01"] = float(percentiles[0])
        result["p05"] = float(percentiles[1])
        result["p25"] = float(percentiles[2])
        result["p75"] = float(percentiles[3])
        result["p95"] = float(percentiles[4])
        result["p99"] = float(percentiles[5])

        result["positive_fraction"] = float(
            np.mean(finite > 0)
        )

        result["zero_fraction"] = float(
            np.mean(finite == 0)
        )

        result["negative_fraction"] = float(
            np.mean(finite < 0)
        )

    except Exception as exc:

        result["sample_method"] = (
            f"Sampling failed: {exc}"
        )

    return result


# ============================================================
# 6. ATTRIBUTE INTERPRETATION
# ============================================================

def extract_radiometric_evidence(dataset_info):

    attrs = dataset_info["attributes"]

    text = json.dumps(
        attrs,
        default=str
    ).lower()

    evidence = {
        "gamma0_mentioned": (
            "gamma0" in text
            or "gamma_0" in text
            or "gamma nought" in text
        ),
        "sigma0_mentioned": (
            "sigma0" in text
            or "sigma_0" in text
            or "sigma nought" in text
        ),
        "beta0_mentioned": (
            "beta0" in text
            or "beta_0" in text
        ),
        "linear_mentioned": (
            "linear" in text
        ),
        "db_mentioned": (
            "decibel" in text
            or " db" in text
            or "db " in text
        ),
        "scale_factor": attrs.get("scale_factor"),
        "add_offset": attrs.get("add_offset"),
        "units": attrs.get("units"),
        "fill_value": attrs.get("_FillValue"),
        "valid_min": attrs.get("valid_min"),
        "valid_max": attrs.get("valid_max"),
        "valid_range": attrs.get("valid_range")
    }

    if evidence["gamma0_mentioned"]:
        evidence["quantity_identification"] = (
            "Gamma0 explicitly mentioned in dataset attributes"
        )

    elif evidence["sigma0_mentioned"]:
        evidence["quantity_identification"] = (
            "Sigma0 explicitly mentioned in dataset attributes"
        )

    elif evidence["beta0_mentioned"]:
        evidence["quantity_identification"] = (
            "Beta0 explicitly mentioned in dataset attributes"
        )

    else:
        evidence["quantity_identification"] = (
            "Not established from this dataset's attributes"
        )

    return evidence


# ============================================================
# 7. DISCOVER ORIGINAL HDF5 FILES
# ============================================================

def find_files(date):

    files = list(
        RAW_DIR.glob(f"*{date}*.h5")
    )

    if not files:

        files = list(
            RAW_DIR.glob(f"*{date}*.hdf5")
        )

    return files


# ============================================================
# 8. RECURSIVE HDF5 INSPECTION
# ============================================================

def inspect_file(file_path, date):

    print("\n" + "=" * 80)
    print("PRODUCT:", file_path.name)
    print("=" * 80)

    report = {
        "date": date,
        "filename": file_path.name,
        "file_path": str(file_path),
        "root_attributes": {},
        "groups": [],
        "datasets": [],
        "calibration_related_metadata": [],
        "identified_measurements": {},
        "errors": []
    }

    with h5py.File(file_path, "r") as h5:

        # ----------------------------------------------------
        # Root attributes
        # ----------------------------------------------------

        report["root_attributes"] = extract_attributes(h5)

        # ----------------------------------------------------
        # Recursive group and dataset traversal
        # ----------------------------------------------------

        def visitor(name, obj):

            full_path = "/" + name

            if isinstance(obj, h5py.Group):

                group_info = {
                    "path": full_path,
                    "attributes": extract_attributes(obj)
                }

                report["groups"].append(group_info)

                keyword_matches = search_metadata_keywords(
                    full_path,
                    group_info["attributes"]
                )

                if keyword_matches:

                    report[
                        "calibration_related_metadata"
                    ].append({
                        "path": full_path,
                        "type": "GROUP",
                        "attributes": group_info["attributes"],
                        "keyword_matches": list(
                            keyword_matches.keys()
                        )
                    })

            elif isinstance(obj, h5py.Dataset):

                attrs = extract_attributes(obj)

                info = {
                    "path": full_path,
                    "name": obj.name,
                    "shape": list(obj.shape),
                    "ndim": obj.ndim,
                    "dtype": str(obj.dtype),
                    "chunks": (
                        list(obj.chunks)
                        if obj.chunks else None
                    ),
                    "compression": obj.compression,
                    "attributes": attrs,
                    "identified_type": identify_dataset(full_path),
                    "metadata_keyword_matches": list(
                        search_metadata_keywords(
                            full_path,
                            attrs
                        ).keys()
                    )
                }

                info["radiometric_evidence"] = (
                    extract_radiometric_evidence(info)
                )

                # Sample only likely measurement datasets
                if info["identified_type"] is not None:

                    print("\nDATASET:", full_path)
                    print("Type:", info["identified_type"])
                    print("Shape:", obj.shape)
                    print("Dtype:", obj.dtype)

                    print("Attributes:")

                    for key, value in attrs.items():
                        print(f"  {key}: {value}")

                    print("Sampling numerical values...")

                    info["numerical_statistics"] = (
                        sample_dataset(obj)
                    )

                    print(
                        "Sample min:",
                        info["numerical_statistics"]["minimum"]
                    )

                    print(
                        "Sample max:",
                        info["numerical_statistics"]["maximum"]
                    )

                    report[
                        "identified_measurements"
                    ].setdefault(
                        info["identified_type"], []
                    ).append(full_path)

                # Save calibration-related metadata
                if info["metadata_keyword_matches"]:

                    report[
                        "calibration_related_metadata"
                    ].append({
                        "path": full_path,
                        "type": "DATASET",
                        "attributes": attrs,
                        "keyword_matches": (
                            info["metadata_keyword_matches"]
                        )
                    })

                report["datasets"].append(info)

        h5.visititems(visitor)

    return report


# ============================================================
# 9. MAIN PROCESSING
# ============================================================

def main():

    all_reports = []

    dataset_rows = []

    calibration_rows = []

    print("=" * 80)
    print("SCRIPT 29: ORIGINAL GCOV RADIOMETRIC METADATA INSPECTION")
    print("=" * 80)

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Original HDF5 directory not found: {RAW_DIR}"
        )

    for date in DATES:

        files = find_files(date)

        if not files:

            print(
                f"\nWARNING: No original HDF5 found for {date}"
            )

            continue

        if len(files) > 1:

            print(
                f"\nWARNING: Found {len(files)} HDF5 files "
                f"for {date}. Inspecting all."
            )

        for file_path in files:

            try:

                report = inspect_file(
                    file_path,
                    date
                )

                all_reports.append(report)

                output_json = (
                    OUTPUT_DIR
                    / f"{date}_original_metadata.json"
                )

                # If multiple files exist for one date,
                # use a distinct filename.
                if output_json.exists():

                    output_json = (
                        OUTPUT_DIR
                        / f"{date}_{len(all_reports)}_original_metadata.json"
                    )

                with open(output_json, "w") as f:

                    json.dump(
                        report,
                        f,
                        indent=4,
                        default=str
                    )

                print(
                    "\nMetadata JSON saved:",
                    output_json
                )

                # Dataset summary
                for dataset in report["datasets"]:

                    stats = dataset.get(
                        "numerical_statistics", {}
                    )

                    evidence = dataset[
                        "radiometric_evidence"
                    ]

                    dataset_rows.append([
                        date,
                        file_path.name,
                        dataset["path"],
                        dataset["identified_type"],
                        dataset["dtype"],
                        dataset["shape"],
                        json.dumps(dataset["attributes"]),
                        evidence["quantity_identification"],
                        evidence["units"],
                        evidence["fill_value"],
                        evidence["valid_min"],
                        evidence["valid_max"],
                        evidence["scale_factor"],
                        evidence["add_offset"],
                        stats.get("minimum"),
                        stats.get("maximum"),
                        stats.get("mean"),
                        stats.get("median"),
                        stats.get("std"),
                        stats.get("finite_count"),
                        stats.get("sample_method")
                    ])

                # Calibration summary
                for item in report[
                    "calibration_related_metadata"
                ]:

                    calibration_rows.append([
                        date,
                        file_path.name,
                        item["path"],
                        item["type"],
                        json.dumps(item["attributes"]),
                        ";".join(item["keyword_matches"])
                    ])

            except Exception as exc:

                print("\nERROR inspecting:", file_path)
                print(exc)

                traceback.print_exc()

    # ========================================================
    # 10. EXPORT CSV REPORTS
    # ========================================================

    write_headers = [
        "date",
        "filename",
        "dataset_path",
        "identified_type",
        "dtype",
        "shape",
        "all_attributes",
        "quantity_identification",
        "units",
        "fill_value",
        "valid_min",
        "valid_max",
        "scale_factor",
        "add_offset",
        "sample_min",
        "sample_max",
        "sample_mean",
        "sample_median",
        "sample_std",
        "sample_finite_count",
        "sample_method"
    ]

    with open(
        OUTPUT_DIR / "dataset_metadata_summary.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(write_headers)
        writer.writerows(dataset_rows)

    with open(
        OUTPUT_DIR / "calibration_metadata_summary.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "date",
            "filename",
            "metadata_path",
            "object_type",
            "attributes",
            "matched_keywords"
        ])

        writer.writerows(calibration_rows)

    with open(
        OUTPUT_DIR / "all_acquisition_metadata.json",
        "w"
    ) as f:

        json.dump(
            all_reports,
            f,
            indent=4,
            default=str
        )

    # ========================================================
    # 11. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 80)
    print("SCRIPT 29 COMPLETED")
    print("=" * 80)

    print("\nAcquisitions inspected:", len(all_reports))

    for report in all_reports:

        print("\nDate:", report["date"])

        print("File:", report["filename"])

        print(
            "Total datasets:",
            len(report["datasets"])
        )

        print(
            "Identified measurements:"
        )

        for kind, paths in report[
            "identified_measurements"
        ].items():

            print(f"  {kind}:")

            for path in paths:
                print(f"    {path}")

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)

    print("\nGENERATED FILES:")

    for path in sorted(OUTPUT_DIR.iterdir()):

        print(path.name)


if __name__ == "__main__":
    main()

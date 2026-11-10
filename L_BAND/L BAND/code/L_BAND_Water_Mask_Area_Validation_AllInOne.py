r"""
L_BAND_Water_Mask_Area_Validation_AllInOne.py

ONE-SCRIPT L-BAND WORKFLOW
==========================
Creates, for each of the three currently prepared L-band AOI dates:
  1) Water mask (binary + diagnostic 3-class raster)
  2) Per-date HH/HV classification thresholds
  3) Water-area statistics (candidate water, uncertain, land)
  4) Validation metrics (precision, recall, F1, IoU, overall accuracy)
     when a reference label raster is available.

PROJECT INPUTS expected:
  D:\NISAR_Project\L_BAND\PROCESSED\AOI_CROPS\YYYYMMDD\
    HHHH_gamma0_linear.tif
    HVHV_gamma0_linear.tif
    HHHH_valid_mask.tif        (optional but recommended)
    HVHV_valid_mask.tif        (optional but recommended)
    frequencyA_valid_mask.tif  (optional)

OPTIONAL REFERENCE MASKS:
  D:\NISAR_Project\L_BAND\REFERENCE\VALIDATION_LABELS\YYYYMMDD\reference_water_mask.tif

Reference raster encoding MUST be:
  0 = reference non-water / land
  1 = reference water
  255 = unknown, cloud, shadow, or excluded pixel (NoData)

The script also automatically uses the existing 20260819 Sentinel-2
optical_reference_seed_20m.tif if no manual reference raster is present:
  D:\NISAR_Project\L_BAND\REFERENCE\SENTINEL2\20260819\optical_reference_seed_20m.tif
That Sentinel-2 seed is a SCREENING REFERENCE, not ground truth. Any metrics
against it are explicitly marked PRELIMINARY and must not be presented as
final independent accuracy validation without visual review / better labels.

OUTPUTS:
  D:\NISAR_Project\L_BAND\RESULTS\WATER_MASK_AREA_VALIDATION\
    YYYYMMDD\water_mask_binary.tif
    YYYYMMDD\water_mask_3class.tif
    YYYYMMDD\dark_candidate_consensus.tif
    YYYYMMDD\water_evidence_score.tif
    YYYYMMDD\thresholds_and_scene_report.json
    classification_thresholds.csv
    water_area_statistics.csv
    validation_metrics.csv
    run_summary.json
    README.txt

INSTALL ONCE:
  py -m pip install numpy pandas rasterio scipy

RUN:
  py L_BAND_Water_Mask_Area_Validation_AllInOne.py

IMPORTANT SCIENTIFIC NOTES:
- Inputs are linear Gamma0 (gamma-zero) power. Conversion to dB uses
  10*log10(Gamma0). No Gamma0-to-Sigma0 conversion is performed.
- Otsu thresholds are per-date histogram split points, not learned water
  thresholds. This is an unsupervised baseline, not a Random Forest model.
- The five-part score is an evidence count, NOT a calibrated probability.
- Binary mask: 0=non-water-like, 1=high-confidence open-water candidate,
  255=uncertain/invalid. Do not silently count 255 as land.
- Area figures are candidate-class areas from the current AOI crops, not
  verified ground-truth water area. Current AOI crops are not full-scene
  final challenge coverage.
- Validation excludes uncertain/invalid predictions and unknown reference
  pixels. The report includes validation coverage so this restriction is
  visible. Metrics from the Sentinel-2 seed are preliminary only.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter, convolve


# ----------------------------- USER SETTINGS -----------------------------
PROJECT_DIR = Path(r"D:\NISAR_Project")
L_BAND_DIR = PROJECT_DIR / "L_BAND"
INPUT_ROOT = L_BAND_DIR / "PROCESSED" / "AOI_CROPS"
OUTPUT_ROOT = L_BAND_DIR / "RESULTS" / "WATER_MASK_AREA_VALIDATION"
REFERENCE_ROOT = L_BAND_DIR / "REFERENCE" / "VALIDATION_LABELS"
S2_SEED_20260819 = (
    L_BAND_DIR / "REFERENCE" / "SENTINEL2" / "20260819" /
    "optical_reference_seed_20m.tif"
)

DATES = ("20260819", "20260924", "20261018")

OTSU_BINS = 512
OTSU_SAMPLE_SIZE = 500_000
TEXTURE_WINDOW = 5
TEXTURE_CANDIDATE_PERCENTILE = 60.0
NEIGHBOURHOOD_WINDOW = 3
MIN_NEIGHBOUR_CANDIDATES = 3
RANDOM_SEED = 42
NODATA_BYTE = 255

# -------------------------------------------------------------------------


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        valid = src.read_masks(1) > 0
        if src.nodata is not None:
            valid &= arr != src.nodata
    return arr, profile, valid


def check_same_grid(ref_profile, other_profile, label):
    same = (
        ref_profile["width"] == other_profile["width"]
        and ref_profile["height"] == other_profile["height"]
        and ref_profile["crs"] == other_profile["crs"]
        and ref_profile["transform"] == other_profile["transform"]
    )
    if not same:
        raise ValueError(
            f"Grid mismatch for {label}. Do not classify unaligned rasters; "
            "co-register them first."
        )


def load_validity_mask(path, shape, reference_profile):
    arr, profile, internal_valid = read_raster(path)
    check_same_grid(reference_profile, profile, path.name)
    if arr.shape != shape:
        raise ValueError(f"Mask shape mismatch: {path}")
    # Assumption used by the existing AOI extraction workflow: 0=invalid,
    # non-zero=valid. The unique values are recorded in the report.
    valid = internal_valid & np.isfinite(arr) & (arr != 0)
    values = np.unique(arr)
    return valid, [float(v) for v in values[:50]], int(valid.sum())


def deterministic_sample(values, max_count=OTSU_SAMPLE_SIZE):
    flat = values[np.isfinite(values)]
    if flat.size <= max_count:
        return flat
    rng = np.random.default_rng(RANDOM_SEED)
    return flat[rng.choice(flat.size, size=max_count, replace=False)]


def otsu_threshold(values):
    sample = deterministic_sample(values)
    if sample.size < 1000:
        raise ValueError("Fewer than 1,000 valid samples: cannot calculate Otsu threshold.")
    low, high = np.percentile(sample, [0.5, 99.5])
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("Invalid histogram range for Otsu threshold.")
    clipped = np.clip(sample, low, high)
    counts, edges = np.histogram(clipped, bins=OTSU_BINS, range=(low, high))
    counts = counts.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w_left = np.cumsum(counts)[:-1]
    w_right = np.cumsum(counts[::-1])[::-1][1:]
    sum_left = np.cumsum(counts * centres)[:-1]
    sum_total = np.sum(counts * centres)
    mu_left = sum_left / np.maximum(w_left, 1e-12)
    mu_right = (sum_total - sum_left) / np.maximum(w_right, 1e-12)
    between = w_left * w_right * (mu_left - mu_right) ** 2
    idx = int(np.nanargmax(between))
    threshold = float(edges[idx + 1])
    total_mean = np.average(centres, weights=np.maximum(counts, 1e-12))
    total_var_sum = np.sum(counts * (centres - total_mean) ** 2)
    separability = float(
        np.max(between) / max(total_var_sum * counts.sum(), 1e-12)
    )
    return threshold, separability, int(sample.size), [float(low), float(high)]


def local_std(values_db, valid):
    good = valid & np.isfinite(values_db)
    if not np.any(good):
        raise ValueError("No valid pixels available for texture calculation.")
    fill = float(np.median(values_db[good]))
    work = np.where(good, values_db, fill).astype(np.float32)
    mean = uniform_filter(work, size=TEXTURE_WINDOW, mode="nearest")
    mean_sq = uniform_filter(work * work, size=TEXTURE_WINDOW, mode="nearest")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)


def write_byte_tif(path, array, ref_profile, description, colormap=None, tags=None):
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=NODATA_BYTE,
        compress="DEFLATE", predictor=2, tiled=True, BIGTIFF="IF_SAFER"
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.uint8, copy=False), 1)
        dst.set_band_description(1, description)
        if colormap:
            dst.write_colormap(1, colormap)
        tag_dict = {"classification_status": "UNSUPERVISED_PROVISIONAL_BASELINE"}
        if tags:
            tag_dict.update(tags)
        dst.update_tags(**{str(k): str(v) for k, v in tag_dict.items()})


def pixel_area_km2(profile):
    transform = profile["transform"]
    area_map_units = abs(transform.a * transform.e - transform.b * transform.d)
    crs = profile.get("crs")
    if crs is None or not crs.is_projected:
        raise ValueError("Area calculation requires a projected CRS (e.g. EPSG:32645).")
    # Convert projected CRS axis units to metres if needed.
    factor = 1.0
    try:
        if crs.axis_info:
            factor = float(crs.axis_info[0].unit_conversion_factor)
    except Exception:
        factor = 1.0
    return (area_map_units * factor * factor) / 1_000_000.0


def find_reference(date):
    manual = REFERENCE_ROOT / date / "reference_water_mask.tif"
    if manual.exists():
        return manual, "MANUAL_REFERENCE_LABELS", "Reference raster supplied by user"
    if date == "20260819" and S2_SEED_20260819.exists():
        return (
            S2_SEED_20260819,
            "PRELIMINARY_SENTINEL2_SCREENING_SEED",
            "SCL+NDWI+MNDWI screening seed; not ground truth; visually review before reporting"
        )
    return None, "NOT_AVAILABLE", "No reference raster supplied for this date"


def read_reference_on_target_grid(path, target_profile, target_shape):
    """Read 0/1/255 reference labels; align to NISAR grid using nearest neighbour."""
    with rasterio.open(path) as src:
        ref = src.read(1)
        src_valid = src.read_masks(1) > 0
        if src.nodata is not None:
            src_valid &= ref != src.nodata
        # Only 0, 1 and 255 are accepted. Unknown classes become 255.
        src_out = np.where(src_valid & np.isin(ref, [0, 1]), ref, 255).astype(np.uint8)
        aligned = np.full(target_shape, 255, dtype=np.uint8)
        reproject(
            source=src_out,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=255,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=255,
            resampling=Resampling.nearest,
        )
    return aligned


def compute_metrics(prediction, reference):
    # Only hard predictions (0/1) against known reference labels (0/1).
    used = np.isin(prediction, [0, 1]) & np.isin(reference, [0, 1])
    n_ref = int(np.isin(reference, [0, 1]).sum())
    n_used = int(used.sum())
    if n_used == 0:
        return {
            "status": "NO_OVERLAP_AFTER_MASKING", "TP": 0, "FP": 0, "TN": 0,
            "FN": 0, "precision": None, "recall": None, "F1": None,
            "IoU": None, "overall_accuracy": None,
            "reference_valid_pixels": n_ref, "evaluated_pixels": n_used,
            "validation_coverage_percent": 0.0,
        }
    p = prediction[used].astype(np.uint8)
    r = reference[used].astype(np.uint8)
    tp = int(np.count_nonzero((p == 1) & (r == 1)))
    fp = int(np.count_nonzero((p == 1) & (r == 0)))
    tn = int(np.count_nonzero((p == 0) & (r == 0)))
    fn = int(np.count_nonzero((p == 0) & (r == 1)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision == 0 and recall == 0 else None)
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    oa = (tp + tn) / n_used
    return {
        "status": "COMPUTED", "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall, "F1": f1, "IoU": iou,
        "overall_accuracy": oa, "reference_valid_pixels": n_ref,
        "evaluated_pixels": n_used,
        "validation_coverage_percent": 100.0 * n_used / n_ref if n_ref else 0.0,
    }


def process_date(date):
    folder = INPUT_ROOT / date
    if not folder.exists():
        raise FileNotFoundError(f"Missing L-band input folder: {folder}")

    hh_path = folder / "HHHH_gamma0_linear.tif"
    hv_path = folder / "HVHV_gamma0_linear.tif"
    for p in (hh_path, hv_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    print(f"\nProcessing L-band {date} ...")
    hh_raw, hh_profile, hh_internal = read_raster(hh_path)
    hv_raw, hv_profile, hv_internal = read_raster(hv_path)
    check_same_grid(hh_profile, hv_profile, "HHHH and HVHV Gamma0 rasters")
    shape = hh_raw.shape

    hh = hh_raw.astype(np.float32, copy=False)
    hv = hv_raw.astype(np.float32, copy=False)
    valid = (
        hh_internal & hv_internal
        & np.isfinite(hh) & (hh > 0)
        & np.isfinite(hv) & (hv > 0)
    )

    mask_details = {}
    for mask_name in (
        "HHHH_valid_mask.tif", "HVHV_valid_mask.tif", "frequencyA_valid_mask.tif"
    ):
        mask_path = folder / mask_name
        if mask_path.exists():
            mask_valid, unique, mask_count = load_validity_mask(
                mask_path, shape, hh_profile
            )
            valid &= mask_valid
            mask_details[mask_name] = {
                "unique_values": unique,
                "mask_valid_pixel_count": mask_count,
            }
        else:
            mask_details[mask_name] = "not present; not applied"

    valid_count = int(valid.sum())
    if valid_count < 1000:
        raise ValueError(f"{date}: only {valid_count} valid pixels remain.")

    # Gamma0 linear power to dB, once. No Gamma0-to-Sigma0 conversion.
    hh_db = np.full(shape, np.nan, dtype=np.float32)
    hv_db = np.full(shape, np.nan, dtype=np.float32)
    hh_db[valid] = 10.0 * np.log10(hh[valid])
    hv_db[valid] = 10.0 * np.log10(hv[valid])

    hh_thr, hh_sep, hh_n, hh_range = otsu_threshold(hh_db[valid])
    hv_thr, hv_sep, hv_n, hv_range = otsu_threshold(hv_db[valid])
    hh_low = valid & (hh_db <= hh_thr)
    hv_low = valid & (hv_db <= hv_thr)
    consensus = hh_low & hv_low

    hh_texture = local_std(hh_db, valid)
    hv_texture = local_std(hv_db, valid)
    texture_population = consensus if int(consensus.sum()) >= 1000 else valid
    texture_source = (
        "HH-and-HV low-backscatter consensus pixels"
        if int(consensus.sum()) >= 1000
        else "all valid pixels (fallback: consensus had fewer than 1,000 pixels)"
    )
    hh_texture_thr = float(np.percentile(
        hh_texture[texture_population], TEXTURE_CANDIDATE_PERCENTILE
    ))
    hv_texture_thr = float(np.percentile(
        hv_texture[texture_population], TEXTURE_CANDIDATE_PERCENTILE
    ))
    hh_smooth = valid & (hh_texture <= hh_texture_thr)
    hv_smooth = valid & (hv_texture <= hv_texture_thr)

    kernel = np.ones((NEIGHBOURHOOD_WINDOW, NEIGHBOURHOOD_WINDOW), dtype=np.uint8)
    support_count = convolve(consensus.astype(np.uint8), kernel, mode="constant", cval=0)
    spatial_support = valid & (support_count >= MIN_NEIGHBOUR_CANDIDATES)

    # Evidence count 0..5. This is not a probability.
    score = (
        hh_low.astype(np.uint8) + hv_low.astype(np.uint8)
        + hh_smooth.astype(np.uint8) + hv_smooth.astype(np.uint8)
        + spatial_support.astype(np.uint8)
    )
    high_confidence = (
        valid & consensus & hh_smooth & hv_smooth & spatial_support & (score == 5)
    )
    candidate_evidence = consensus | (hh_low & hh_smooth) | (hv_low & hv_smooth)
    uncertain = valid & ~high_confidence & candidate_evidence
    nonwater_like = valid & ~high_confidence & ~uncertain

    binary = np.full(shape, 255, dtype=np.uint8)
    binary[nonwater_like] = 0
    binary[high_confidence] = 1
    three_class = np.full(shape, 255, dtype=np.uint8)
    three_class[nonwater_like] = 0
    three_class[high_confidence] = 1
    three_class[uncertain] = 2
    consensus_map = np.full(shape, 255, dtype=np.uint8)
    consensus_map[valid] = consensus[valid].astype(np.uint8)
    score_map = np.full(shape, 255, dtype=np.uint8)
    score_map[valid] = score[valid]

    out_dir = OUTPUT_ROOT / date
    out_dir.mkdir(parents=True, exist_ok=True)
    write_byte_tif(
        out_dir / "water_mask_binary.tif", binary, hh_profile,
        "0=non-water-like; 1=high-confidence water candidate; 255=uncertain/invalid",
        {0: (0, 0, 0, 255), 1: (255, 255, 255, 255), 255: (0, 0, 0, 0)},
        {"HH_Otsu_dB": hh_thr, "HV_Otsu_dB": hv_thr}
    )
    write_byte_tif(
        out_dir / "water_mask_3class.tif", three_class, hh_profile,
        "0=non-water-like; 1=high-confidence water candidate; 2=uncertain; 255=invalid",
        {0: (0, 0, 0, 255), 1: (255, 255, 255, 255),
         2: (255, 190, 0, 255), 255: (0, 0, 0, 0)}
    )
    write_byte_tif(
        out_dir / "dark_candidate_consensus.tif", consensus_map, hh_profile,
        "1=both HH and HV are below their Otsu thresholds; 0=otherwise; 255=invalid"
    )
    write_byte_tif(
        out_dir / "water_evidence_score.tif", score_map, hh_profile,
        "Evidence count 0-5; NOT a calibrated probability; 255=invalid"
    )

    area_per_pixel = pixel_area_km2(hh_profile)
    high_n = int(high_confidence.sum())
    uncertain_n = int(uncertain.sum())
    land_n = int(nonwater_like.sum())
    total_area = valid_count * area_per_pixel
    area_row = {
        "date": date,
        "valid_pixels": valid_count,
        "high_confidence_water_candidate_pixels": high_n,
        "uncertain_pixels": uncertain_n,
        "nonwater_like_pixels": land_n,
        "high_confidence_candidate_area_km2": high_n * area_per_pixel,
        "uncertain_area_km2": uncertain_n * area_per_pixel,
        "nonwater_like_area_km2": land_n * area_per_pixel,
        "valid_AOI_area_km2": total_area,
        "candidate_water_percent_of_valid_area": 100.0 * high_n / valid_count,
        "uncertain_percent_of_valid_area": 100.0 * uncertain_n / valid_count,
        "pixel_area_km2": area_per_pixel,
        "area_interpretation": "candidate class area, not verified ground-truth water area",
    }

    threshold_row = {
        "date": date,
        "HH_Otsu_threshold_dB": hh_thr,
        "HV_Otsu_threshold_dB": hv_thr,
        "HH_texture_threshold_dB_std": hh_texture_thr,
        "HV_texture_threshold_dB_std": hv_texture_thr,
        "HH_Otsu_separability_diagnostic": hh_sep,
        "HV_Otsu_separability_diagnostic": hv_sep,
        "HH_Otsu_sample_size": hh_n,
        "HV_Otsu_sample_size": hv_n,
        "HH_histogram_range_low_dB": hh_range[0],
        "HH_histogram_range_high_dB": hh_range[1],
        "HV_histogram_range_low_dB": hv_range[0],
        "HV_histogram_range_high_dB": hv_range[1],
        "method": "per-date Otsu HH/HV + dual-channel consensus + 5x5 texture + 3x3 support",
    }

    ref_path, ref_type, ref_note = find_reference(date)
    if ref_path is not None:
        reference = read_reference_on_target_grid(ref_path, hh_profile, shape)
        metrics = compute_metrics(binary, reference)
        metrics_row = {
            "date": date, "reference_file": str(ref_path),
            "reference_type": ref_type, "reference_note": ref_note,
            **metrics,
        }
    else:
        metrics_row = {
            "date": date, "reference_file": "",
            "reference_type": ref_type, "reference_note": ref_note,
            "status": "NOT_COMPUTED_NO_REFERENCE_LABELS",
            "TP": None, "FP": None, "TN": None, "FN": None,
            "precision": None, "recall": None, "F1": None,
            "IoU": None, "overall_accuracy": None,
            "reference_valid_pixels": None, "evaluated_pixels": 0,
            "validation_coverage_percent": None,
        }

    report = {
        "date": date,
        "band": "NISAR L-band",
        "input_backscatter": "linear Gamma0 (gamma-zero) power",
        "dB_conversion": "10*log10(Gamma0)",
        "gamma0_to_sigma0_conversion": "NOT PERFORMED",
        "method": threshold_row["method"],
        "classification_thresholds": threshold_row,
        "mask_usage": mask_details,
        "mask_semantics_assumption": "0=invalid and nonzero=valid for *_valid_mask.tif",
        "valid_pixels": valid_count,
        "water_area_statistics": area_row,
        "binary_mask_encoding": "0=non-water-like; 1=high-confidence open-water candidate; 255=uncertain/invalid",
        "three_class_encoding": "0=non-water-like; 1=high-confidence open-water candidate; 2=uncertain; 255=invalid",
        "validation": metrics_row,
        "limitations": [
            "Otsu thresholds are unsupervised histogram split points, not trained water thresholds.",
            "The five-part evidence count is not a calibrated probability.",
            "High-confidence pixels are water candidates, not independently confirmed water.",
            "Water area is candidate-class area; uncertain pixels are reported separately, not forced into land or water.",
            "Validation metrics exclude uncertain/invalid predictions and unknown reference pixels; validation coverage is reported.",
            "Sentinel-2 screening seed metrics, if used, are preliminary and not authoritative ground truth.",
            "Outputs cover the current AOI crops, not the full selected NISAR scene footprint.",
        ],
    }
    (out_dir / "thresholds_and_scene_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"  Valid pixels: {valid_count:,}")
    print(f"  HH threshold: {hh_thr:.3f} dB | HV threshold: {hv_thr:.3f} dB")
    print(f"  Candidate water area: {area_row['high_confidence_candidate_area_km2']:.3f} km2")
    print(f"  Uncertain area: {area_row['uncertain_area_km2']:.3f} km2")
    if metrics_row.get("status") == "COMPUTED":
        print(
            "  Validation (" + ref_type + "): "
            f"F1={metrics_row['F1'] if metrics_row['F1'] is not None else 'NA'}; "
            f"coverage={metrics_row['validation_coverage_percent']:.1f}%"
        )
    else:
        print("  Validation: not computed; no reference labels found.")

    return threshold_row, area_row, metrics_row


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Cannot find inputs: {INPUT_ROOT}\n"
            "Check PROJECT_DIR at the top of this script."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    threshold_rows, area_rows, metric_rows = [], [], []
    for date in DATES:
        threshold, area, metrics = process_date(date)
        threshold_rows.append(threshold)
        area_rows.append(area)
        metric_rows.append(metrics)

    pd.DataFrame(threshold_rows).to_csv(
        OUTPUT_ROOT / "classification_thresholds.csv", index=False
    )
    pd.DataFrame(area_rows).to_csv(
        OUTPUT_ROOT / "water_area_statistics.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_ROOT / "validation_metrics.csv", index=False
    )

    summary = {
        "band": "L-band only",
        "dates_processed": list(DATES),
        "output_directory": str(OUTPUT_ROOT),
        "gamma0_to_sigma0_conversion": False,
        "classification": "unsupervised Otsu-based provisional baseline; no Random Forest",
        "validation_status_by_date": {
            row["date"]: row["status"] for row in metric_rows
        },
        "area_warning": "Candidate-class area, not verified true water area; uncertain pixels are separate.",
        "coverage_warning": "Current input AOIs only; full-scene output requires full-scene processing.",
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "README.txt").write_text(
        "L-BAND WATER MASK / THRESHOLD / AREA / VALIDATION OUTPUTS\n"
        "========================================================\n\n"
        "Open each date folder and load water_mask_3class.tif in QGIS:\n"
        "  0 = non-water-like\n  1 = high-confidence open-water candidate\n"
        "  2 = uncertain\n  255 = invalid\n\n"
        "water_mask_binary.tif uses 0=non-water-like, 1=high-confidence candidate,\n"
        "255=uncertain/invalid NoData. Do not interpret 255 as land.\n\n"
        "Root CSV files:\n"
        "  classification_thresholds.csv: HH/HV Otsu and texture thresholds\n"
        "  water_area_statistics.csv: candidate, uncertain and land areas\n"
        "  validation_metrics.csv: TP/FP/TN/FN, precision, recall, F1, IoU, OA\n\n"
        "Reference labels must use 0=land, 1=water, 255=unknown and be placed at:\n"
        "  REFERENCE/VALIDATION_LABELS/YYYYMMDD/reference_water_mask.tif\n"
        "The 20260819 Sentinel-2 optical_reference_seed_20m.tif is used only as a\n"
        "preliminary screening reference when no manual reference exists.\n\n"
        "This workflow is an unsupervised provisional baseline, not a validated\n"
        "final classifier. Candidate area is not ground-truth water area. Current\n"
        "outputs cover the supplied AOI crops only.\n",
        encoding="utf-8",
    )

    print("\nALL L-BAND PROCESSING FINISHED.")
    print("Outputs:", OUTPUT_ROOT)
    print("Thresholds:", OUTPUT_ROOT / "classification_thresholds.csv")
    print("Area stats:", OUTPUT_ROOT / "water_area_statistics.csv")
    print("Validation:", OUTPUT_ROOT / "validation_metrics.csv")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

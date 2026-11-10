
import csv
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES\S_BAND"
)

REFERENCE_DIR = BASE / "shared"

ALIGNED_DIR = (
    BASE / "validation" / "aligned_classification"
)

VALIDATION_DIR = BASE / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = VALIDATION_DIR / "metrics.csv"
CONFUSION_FILE = VALIDATION_DIR / "confusion_matrix.csv"

DATES = ["20260819", "20260924", "20261018"]

# ============================================================
# BACKUP EXISTING CSV FILES
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path in [METRICS_FILE, CONFUSION_FILE]:

    if path.exists():

        backup = path.with_name(
            f"{path.stem}_backup_{timestamp}{path.suffix}"
        )

        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")

# ============================================================
# METRIC CALCULATION
# ============================================================

metrics_records = []
confusion_records = []

for date in DATES:

    print("\n" + "=" * 60)
    print(f"VALIDATING DATE: {date}")
    print("=" * 60)

    reference_path = (
        REFERENCE_DIR / f"reference_labels_{date}.tif"
    )

    classification_path = (
        ALIGNED_DIR / date / "aligned_water_mask.tif"
    )

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    if not classification_path.exists():
        raise FileNotFoundError(classification_path)

    with rasterio.open(reference_path) as ref_src, \
         rasterio.open(classification_path) as pred_src:

        # ----------------------------------------------------
        # VERIFY GRID ALIGNMENT
        # ----------------------------------------------------

        if (
            ref_src.crs != pred_src.crs
            or ref_src.transform != pred_src.transform
            or ref_src.width != pred_src.width
            or ref_src.height != pred_src.height
        ):
            raise ValueError(
                f"Grid mismatch detected for {date}"
            )

        reference = ref_src.read(1)
        prediction = pred_src.read(1)

    # --------------------------------------------------------
    # CHECK LABEL VALUES
    # --------------------------------------------------------

    ref_values, ref_counts = np.unique(
        reference, return_counts=True
    )

    pred_values, pred_counts = np.unique(
        prediction, return_counts=True
    )

    print("\nReference label distribution:")

    for value, count in zip(ref_values, ref_counts):
        print(f"  {value}: {count:,} pixels")

    print("\nClassification label distribution:")

    for value, count in zip(pred_values, pred_counts):
        print(f"  {value}: {count:,} pixels")

    # --------------------------------------------------------
    # VALID PIXELS
    # Reference:
    # 0 = Non-water
    # 1 = Water
    # 255 = Invalid; excluded
    # --------------------------------------------------------

    valid_reference = np.isin(reference, [0, 1])

    # Classification must contain binary labels
    # on every valid reference pixel.
    unexpected = np.unique(
        prediction[valid_reference]
    )

    if not np.isin(unexpected, [0, 1]).all():

        raise ValueError(
            f"Unexpected classification labels for {date}: "
            f"{unexpected}. Inspect before calculating metrics."
        )

    y_true = reference[valid_reference]
    y_pred = prediction[valid_reference]

    if y_true.size == 0:
        raise ValueError(
            f"No valid reference pixels for {date}"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = TP + TN + FP + FN

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = (TP + TN) / total

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0 else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    iou = (
        TP / (TP + FP + FN)
        if (TP + FP + FN) > 0 else 0.0
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    metrics_records.append({
        "date": date,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "valid_pixels": total,
        "reference_water_pixels": int(np.sum(y_true == 1)),
        "predicted_water_pixels": int(np.sum(y_pred == 1))
    })

    confusion_records.append({
        "date": date,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "total_valid_pixels": total
    })

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("------------------------------")
    print(f"TP: {TP:,}")
    print(f"FP: {FP:,}")
    print(f"FN: {FN:,}")
    print(f"TN: {TN:,}")

    print("\nVALIDATION METRICS")
    print("------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")

    print(f"\nValid reference pixels: {total:,}")

# ============================================================
# WRITE metrics.csv
# ============================================================

with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "date",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "iou",
            "valid_pixels",
            "reference_water_pixels",
            "predicted_water_pixels"
        ]
    )

    writer.writeheader()
    writer.writerows(metrics_records)

# ============================================================
# WRITE confusion_matrix.csv
# ============================================================

with open(CONFUSION_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "date",
            "TP",
            "FP",
            "FN",
            "TN",
            "total_valid_pixels"
        ]
    )

    writer.writeheader()
    writer.writerows(confusion_records)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nMetrics CSV:")
print(METRICS_FILE)

print(f"\nConfusion Matrix CSV:")
print(CONFUSION_FILE)

print("\nAll metrics were calculated from aligned rasters.")
print("Reference pixels labelled 255 were excluded.")
print("Original classification masks were not modified.")

print("\n" + "=" * 60)

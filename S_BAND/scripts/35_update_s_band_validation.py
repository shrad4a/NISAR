
import csv
import shutil
from pathlib import Path
from datetime import datetime

# CHANGE THIS ONLY IF YOUR PROJECT IS IN A DIFFERENT LOCATION
DELIVERABLES = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES\S_BAND"
)

VALIDATION_DIR = DELIVERABLES / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Date, Accuracy, Precision, Recall, F1, IoU, Reference samples
metrics = [
    ["20261018", 0.7622, 0.3242, 0.1841, 0.2349, 0.1331, 3242655],
    ["20260924", 0.5858, 0.6343, 0.1338, 0.2210, 0.1242, 2098371],
    ["20260819", 0.7600, 0.3786, 0.1705, 0.2351, 0.1332, 856131]
]

# Date, True Water, False Water, Missed Water, True Non-water
confusion = [
    ["20261018", 118327, 246610, 524343, 2353375],
    ["20260924", 123309, 71090, 798143, 1105829],
    ["20260819", 31581, 51829, 153658, 619063]
]

files = {
    "metrics.csv": (
        ["date", "accuracy", "precision", "recall",
         "f1_score", "iou", "reference_sample_count"],
        metrics
    ),
    "confusion_matrix.csv": (
        ["date", "true_water", "false_water",
         "missed_water", "true_non_water"],
        confusion
    )
}

# Back up existing CSVs before replacing them
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for filename, (headers, rows) in files.items():
    output_path = VALIDATION_DIR / filename

    if output_path.exists():
        backup_path = VALIDATION_DIR / f"{filename}.{timestamp}.bak"
        shutil.copy2(output_path, backup_path)
        print(f"Backup created: {backup_path.name}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Updated: {output_path}")

print("\nVALIDATION CSV UPDATE COMPLETE")
print("Updated files:")
print("1. validation/metrics.csv")
print("2. validation/confusion_matrix.csv")
print("\nOriginal TIFF classification files were not modified.")

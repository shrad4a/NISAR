
import os
import csv

BASE = r"C:\Users\HP\Downloads\NISAR\S_BAND"

VALID_DIR = os.path.join(
    BASE,
    "TEAM_DELIVERABLES",
    "S_BAND",
    "validation"
)

os.makedirs(VALID_DIR, exist_ok=True)

# Empty templates only. Populate after independent
# reference labels and predictions are available.

metrics_path = os.path.join(
    VALID_DIR, "metrics.csv"
)

confusion_path = os.path.join(
    VALID_DIR, "confusion_matrix.csv"
)

with open(metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "date",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "reference_sample_count"
    ])

with open(confusion_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "date",
        "true_water",
        "false_water",
        "missed_water",
        "true_non_water"
    ])

print("SCRIPT 34 COMPLETED")
print("Validation folder prepared:")
print(VALID_DIR)
print("Metric templates created.")


from pathlib import Path

root = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES"
)

required = [
    "S_BAND/raw_data",
    "S_BAND/extracted_features",
    "S_BAND/classification",
    "S_BAND/validation/metrics.csv",
    "S_BAND/validation/confusion_matrix.csv",
    "S_BAND/metadata/acquisition_details.json",
    "S_BAND/metadata/preprocessing.txt",
    "S_BAND/notebook",
    "SHARED"
]

print("FINAL DELIVERABLE AUDIT\n")

for item in required:
    path = root / item
    status = "FOUND" if path.exists() else "MISSING"
    print(f"[{status}] {item}")

print("\nAUDIT COMPLETE")

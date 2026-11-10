
from pathlib import Path

ROOT = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND"
    r"\TEAM_DELIVERABLES\S_BAND"
)

REQUIRED = {
    "raw_data": [
    "NISAR_S2_PR_GCOV_028_098_A_014_3700_DHNA_A_20260819T233810_20260819T233847_P00500_P_F_I_001 (1).h5",
    "NISAR_S2_UR_GCOV_031_098_A_014_3700_DHNA_A_20260924T233810_20260924T233846_P00500_M_F_I_001.h5",
    "NISAR_S2_PR_GCOV_033_098_A_014_3700_DHNA_A_20261018T233810_20261018T233847_P00500_M_F_I_001 (1).h5"
],
    "extracted_features": [
        "available_feature_layers_20260819.tif",
        "available_feature_layers_20260924.tif",
        "available_feature_layers_20261018.tif"
    ],
    "classification": [
        "20260819/water_mask.tif",
        "20260819/water_probability.tif",
        "20260819/backscatter_map.png",
        "20260924/water_mask.tif",
        "20260924/water_probability.tif",
        "20260924/backscatter_map.png",
        "20261018/water_mask.tif",
        "20261018/water_probability.tif",
        "20261018/backscatter_map.png"
    ],
    "validation": [
        "metrics.csv",
        "confusion_matrix.csv"
    ],
    "metadata": [
        "acquisition_details.json",
        "preprocessing.txt"
    ],
    "notebook": [
        "S_band_analysis.ipynb"
    ]
}

print("\n" + "=" * 60)
print("S-BAND DELIVERABLES CHECK")
print("=" * 60)

missing = []

for folder, files in REQUIRED.items():
    print(f"\n[{folder.upper()}]")

    for filename in files:
        path = ROOT / folder / filename

        if path.is_file():
            print(f"  [OK]      {filename}")
        else:
            print(f"  [MISSING] {filename}")
            missing.append(str(path))

print("\n" + "=" * 60)

if missing:
    print(f"MISSING FILES: {len(missing)}")
    for item in missing:
        print(item)
else:
    print("ALL CHECKED FILES ARE PRESENT")

print("=" * 60)

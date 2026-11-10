import os
import json

# =====================================================
# SCRIPT 32: S-BAND METADATA
# Lightweight; safe alongside Script 31
# =====================================================

BASE = r"C:\Users\HP\Downloads\NISAR\S_BAND"

OUTPUT_DIR = os.path.join(
    BASE, "TEAM_DELIVERABLES", "S_BAND", "metadata"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

acquisitions = {
    "20260819": {
        "date": "2026-08-19",
        "product": "NISAR S-band L2 GCOV",
        "source_file": "NISAR_S2_PR_GCOV_028_098_A_014_3700_DHNA_A_20260819T233810_20260819T233847_P00500_P_F_I_001 (1).h5"
    },
    "20260924": {
        "date": "2026-09-24",
        "product": "NISAR S-band L2 GCOV",
        "source_file": "NISAR_S2_UR_GCOV_031_098_A_014_3700_DHNA_A_20260924T233810_20260924T233846_P00500_M_F_I_001.h5"
    },
    "20261018": {
        "date": "2026-10-18",
        "product": "NISAR S-band L2 GCOV",
        "source_file": "NISAR_S2_PR_GCOV_033_098_A_014_3700_DHNA_A_20261018T233810_20261018T233847_P00500_M_F_I_001 (1).h5"
    }
}

metadata = {
    "sensor": "NISAR S-band",
    "product_level": "L2 GCOV",
    "crs": "EPSG:32645",
    "resolution_m": 10,
    "width_pixels": 34200,
    "height_pixels": 33120,
    "acquisitions": acquisitions,
    "radiometric_information": {
        "primary_measurement": "Gamma0",
        "sigma0_conversion": "Sigma0 = Gamma0 * rtcGammaToSigmaFactor",
        "cross_pol_ratio": "HVHV / HHHH",
        "cross_pol_ratio_db": "10 * log10(HVHV / HHHH)"
    },
    "quality_flags": {
        "mask_1": "Valid fully focused GCOV sample",
        "mask_255": "Outside radar acquisition extent",
        "exception_0": "No input data anomaly",
        "exception_64": "NISAR SSAR sample shift"
    }
}

json_path = os.path.join(
    OUTPUT_DIR, "acquisition_details.json"
)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

preprocessing = """
NISAR S-BAND WATER MAPPING
PREPROCESSING RECORD

Input:
Three NISAR S-band L2 GCOV acquisitions:
2026-08-19
2026-09-24
2026-10-18

Spatial reference:
CRS: EPSG:32645
Nominal resolution: 10 m
Dimensions: 34200 x 33120 pixels

Radiometric processing:
- HH and HV diagonal covariance terms used as Gamma0 measurements.
- Sigma0 calculated by multiplying Gamma0 by rtcGammaToSigmaFactor.
- HV/HH power ratio calculated from HVHV / HHHH.
- Ratio in dB calculated using 10 * log10(HVHV / HHHH).
- HHHV complex term inspected through real, imaginary,
  magnitude and phase components.

Quality handling:
- mask == 1 identifies valid GCOV samples.
- inputDataExceptionMask == 0 identifies clean samples.
- inputDataExceptionMask == 64 identifies SSAR-shift-flagged samples.
- SSAR-shift pixels were retained for analysis and not
  automatically classified as corrupted.
- Non-finite measurements are treated as invalid.

Feature extraction:
- HH Gamma0
- HV Gamma0
- HH dB
- HV dB
- HV/HH ratio in dB

Final classification and validation details must be added
after those outputs have actually been generated.
"""

txt_path = os.path.join(
    OUTPUT_DIR, "preprocessing.txt"
)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(preprocessing.strip() + "\n")

print("SCRIPT 32 COMPLETED")
print("Created:")
print(json_path)
print(txt_path)
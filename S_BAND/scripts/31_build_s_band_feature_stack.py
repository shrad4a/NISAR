import os
import numpy as np
import rasterio
from rasterio.windows import Window

# =====================================================
# SCRIPT 31: BUILD S-BAND FEATURE STACKS
# =====================================================

BASE = r"C:\Users\HP\Downloads\NISAR\S_BAND"

INPUT_DIR = os.path.join(BASE, "full_scene_geotiffs")
SCRIPT30_DIR = os.path.join(BASE, "calibrated_gamma_sigma_analysis")

OUTPUT_DIR = os.path.join(
    BASE, "TEAM_DELIVERABLES", "S_BAND", "extracted_features"
)

DATES = ["20260819", "20260924", "20261018"]

NODATA = -9999.0
BLOCK = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_db(power):
    result = np.full(power.shape, NODATA, dtype=np.float32)
    valid = np.isfinite(power) & (power > 0)
    result[valid] = 10.0 * np.log10(power[valid])
    return result


for date in DATES:

    print(f"\nProcessing {date}")

    hh_path = os.path.join(INPUT_DIR, date, "HHHH.tif")
    hv_path = os.path.join(INPUT_DIR, date, "HVHV.tif")

    ratio_path = os.path.join(
        SCRIPT30_DIR, date, "HV_HH_ratio_gamma_db.tif"
    )

    output_path = os.path.join(
        OUTPUT_DIR, f"available_feature_layers_{date}.tif"
    )

    for path in [hh_path, hv_path, ratio_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input: {path}")

    with rasterio.open(hh_path) as hh_src, \
         rasterio.open(hv_path) as hv_src, \
         rasterio.open(ratio_path) as ratio_src:

        if (hh_src.width, hh_src.height) != (
            hv_src.width, hv_src.height
        ):
            raise ValueError("HH/HV dimensions do not match")

        if (hh_src.width, hh_src.height) != (
            ratio_src.width, ratio_src.height
        ):
            raise ValueError("Ratio raster dimensions do not match")

        if hh_src.transform != hv_src.transform or \
           hh_src.transform != ratio_src.transform or \
           hh_src.crs != hv_src.crs or \
           hh_src.crs != ratio_src.crs:
            raise ValueError("Input rasters are not on the same grid")

        profile = hh_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=5,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(output_path, "w", **profile) as dst:

            descriptions = [
                "HH_Gamma0",
                "HV_Gamma0",
                "HH_dB",
                "HV_dB",
                "HV_HH_ratio_Gamma0_dB"
            ]

            for i, description in enumerate(descriptions, start=1):
                dst.set_band_description(i, description)

            for row in range(0, hh_src.height, BLOCK):
                for col in range(0, hh_src.width, BLOCK):

                    h = min(BLOCK, hh_src.height - row)
                    w = min(BLOCK, hh_src.width - col)
                    window = Window(col, row, w, h)

                    hh = hh_src.read(1, window=window).astype(np.float32)
                    hv = hv_src.read(1, window=window).astype(np.float32)
                    ratio_db = ratio_src.read(1, window=window).astype(np.float32)

                    # Respect source NoData and finite-value checks.
                    hh_valid = np.isfinite(hh)
                    hv_valid = np.isfinite(hv)
                    ratio_valid = np.isfinite(ratio_db)

                    if hh_src.nodata is not None:
                        hh_valid &= hh != hh_src.nodata
                    if hv_src.nodata is not None:
                        hv_valid &= hv != hv_src.nodata
                    if ratio_src.nodata is not None:
                        ratio_valid &= ratio_db != ratio_src.nodata

                    hh_out = np.where(hh_valid, hh, NODATA).astype(np.float32)
                    hv_out = np.where(hv_valid, hv, NODATA).astype(np.float32)

                    hh_db = convert_db(hh)
                    hv_db = convert_db(hv)

                    ratio_out = np.where(
                        ratio_valid, ratio_db, NODATA
                    ).astype(np.float32)

                    dst.write(hh_out, 1, window=window)
                    dst.write(hv_out, 2, window=window)
                    dst.write(hh_db, 3, window=window)
                    dst.write(hv_db, 4, window=window)
                    dst.write(ratio_out, 5, window=window)

                print(
                    f"{date}: rows "
                    f"{min(row + BLOCK, hh_src.height)}/{hh_src.height}"
                )

    print("Saved:", output_path)

print("\nSCRIPT 31 COMPLETED")
print("Output directory:", OUTPUT_DIR)
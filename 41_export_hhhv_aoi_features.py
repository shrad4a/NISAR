
import os
import glob
import h5py
import numpy as np
import rasterio

H5_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5"
AOI_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\aoi_outputs"
OUTPUT_DIR = (
    r"C:\Users\HP\Downloads\NISAR_REPO"
    r"\S_BAND\extracted_features\hhhv"
)

DATES = ["20260819", "20260924", "20261018"]

BASE = "/science/SSAR/GCOV/grids/frequencyA/"
HHHV_PATH = BASE + "HHHV"

# Verified against the existing AOI rasters
ROW_OFFSET = 7332
COL_OFFSET = 19359

os.makedirs(OUTPUT_DIR, exist_ok=True)

for date in DATES:
    print("\n" + "=" * 65)
    print("Processing:", date)

    h5_matches = [
        p for p in glob.glob(os.path.join(H5_DIR, "*.h5"))
        if date in os.path.basename(p)
    ]

    if len(h5_matches) != 1:
        raise RuntimeError(
            f"Expected one HDF5 file for {date}, "
            f"found {len(h5_matches)}"
        )

    h5_path = h5_matches[0]
    aoi_folder = os.path.join(AOI_DIR, date)
    output_folder = os.path.join(OUTPUT_DIR, date)

    os.makedirs(output_folder, exist_ok=True)

    # Read the existing AOI reference profile
    with rasterio.open(os.path.join(aoi_folder, "HHHH.tif")) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

    # Read quality masks
    with rasterio.open(os.path.join(aoi_folder, "mask.tif")) as src:
        mask = src.read(1)

    with rasterio.open(
        os.path.join(aoi_folder, "inputDataExceptionMask.tif")
    ) as src:
        exception_mask = src.read(1)

    # Extract only the verified AOI window
    with h5py.File(h5_path, "r") as f:
        ds = f[HHHV_PATH]

        hhhv = ds[
            ROW_OFFSET:ROW_OFFSET + height,
            COL_OFFSET:COL_OFFSET + width
        ]

    if hhhv.shape != (height, width):
        raise RuntimeError(
            f"HHHV shape mismatch for {date}: {hhhv.shape}"
        )

    # Same quality conditions as the existing HH/HV extraction
    valid = (
        (mask == 1) &
        np.isin(exception_mask, [0, 64]) &
        np.isfinite(hhhv.real) &
        np.isfinite(hhhv.imag)
    )

    print("AOI shape:", hhhv.shape)
    print("Valid pixels:", np.count_nonzero(valid))
    print("Rejected pixels:", valid.size - np.count_nonzero(valid))

    # Derive the four representations
    features = {
        "HHHV_real.tif": hhhv.real,
        "HHHV_imaginary.tif": hhhv.imag,
        "HHHV_magnitude.tif": np.abs(hhhv),
        "HHHV_phase_radians.tif": np.angle(hhhv)
    }

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        tiled=True
    )

    for filename, values in features.items():
        output = np.full(
            (height, width),
            -9999.0,
            dtype=np.float32
        )

        output[valid] = values[valid].astype(np.float32)

        output_path = os.path.join(output_folder, filename)

        with rasterio.open(
            output_path, "w", **output_profile
        ) as dst:
            dst.write(output, 1)
            dst.update_tags(
                source="NISAR S-band GCOV HHHV",
                acquisition_date=date,
                representation=filename.replace(".tif", ""),
                quality_mask="mask == 1",
                accepted_exception_flags="0,64",
                phase_units="radians" if "phase" in filename else "not_applicable"
            )

        print("Saved:", output_path)
        print(
            "  Min:",
            float(np.min(output[valid])),
            "Max:",
            float(np.max(output[valid]))
        )

    del hhhv, features, output

    print("Completed:", date)

print("\nAll HHHV AOI feature layers exported.")


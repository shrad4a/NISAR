
import os
import glob
import h5py
import numpy as np
import rasterio

H5_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5"
AOI_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\aoi_outputs"
FEATURE_DIR = r"C:\Users\HP\Downloads\NISAR_REPO\S_BAND\extracted_features\hhhv"

DATES = ["20260819", "20260924", "20261018"]

BASE = "/science/SSAR/GCOV/grids/frequencyA/"
ROW, COL = 7332, 19359

names = [
    "HHHV_real.tif",
    "HHHV_imaginary.tif",
    "HHHV_magnitude.tif",
    "HHHV_phase_radians.tif"
]

for date in DATES:
    print("\n" + "=" * 65)
    print("VALIDATING:", date)

    h5_matches = [
        p for p in glob.glob(os.path.join(H5_DIR, "*.h5"))
        if date in os.path.basename(p)
    ]

    if len(h5_matches) != 1:
        raise RuntimeError("Incorrect HDF5 match count")

    aoi = os.path.join(AOI_DIR, date)
    out = os.path.join(FEATURE_DIR, date)

    with rasterio.open(os.path.join(aoi, "HHHH.tif")) as ref:
        profile = ref.profile
        height, width = ref.height, ref.width

    with rasterio.open(os.path.join(aoi, "mask.tif")) as src:
        mask = src.read(1)

    with rasterio.open(
        os.path.join(aoi, "inputDataExceptionMask.tif")
    ) as src:
        exception = src.read(1)

    valid = (
        (mask == 1) &
        np.isin(exception, [0, 64])
    )

    with h5py.File(h5_matches[0], "r") as f:
        source = f[BASE + "HHHV"][
            ROW:ROW + height,
            COL:COL + width
        ]

    valid &= np.isfinite(source.real) & np.isfinite(source.imag)

    expected = {
        "HHHV_real.tif": source.real,
        "HHHV_imaginary.tif": source.imag,
        "HHHV_magnitude.tif": np.abs(source),
        "HHHV_phase_radians.tif": np.angle(source)
    }

    for name in names:
        path = os.path.join(out, name)

        with rasterio.open(path) as src:
            data = src.read(1)

            assert src.crs == profile["crs"], "CRS mismatch"
            assert src.transform == profile["transform"], "Transform mismatch"
            assert (src.height, src.width) == (height, width), "Shape mismatch"
            assert src.dtypes[0] == "float32", "Datatype mismatch"
            assert src.nodata == -9999.0, "Nodata mismatch"

        expected_data = expected[name].astype(np.float32)

        errors = np.abs(
            data[valid].astype(np.float64) -
            expected_data[valid].astype(np.float64)
        )

        nodata_count = np.count_nonzero(data == -9999.0)
        unexpected_nodata = np.count_nonzero(
            valid & (data == -9999.0)
        )

        print("\n", name)
        print("  Valid pixels:", np.count_nonzero(valid))
        print("  Nodata pixels:", nodata_count)
        print("  Unexpected nodata:", unexpected_nodata)
        print("  Maximum numerical error:", errors.max())

        assert unexpected_nodata == 0
        assert np.all(errors < 1e-5)

    print("\nPASS:", date)

print("\nALL HHHV FEATURE VALIDATIONS PASSED.")


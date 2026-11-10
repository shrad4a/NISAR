
import os
import glob
import h5py
import numpy as np
import rasterio

H5_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5"
AOI_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\aoi_outputs"

dates = ["20260819", "20260924", "20261018"]

h5_files = glob.glob(os.path.join(H5_DIR, "*.h5"))

for date in dates:
    print("\n" + "=" * 75)
    print("ACQUISITION:", date)

    h5_matches = [f for f in h5_files if date in os.path.basename(f)]
    if len(h5_matches) != 1:
        print("ERROR: Expected one HDF5 file, found:", len(h5_matches))
        continue

    h5_path = h5_matches[0]
    aoi_path = os.path.join(AOI_DIR, date)

    print("\nHDF5:", os.path.basename(h5_path))

    # Inspect AOI raster alignment
    print("\n--- AOI RASTER ALIGNMENT ---")

    raster_names = [
        "HHHH.tif",
        "HVHV.tif",
        "mask.tif",
        "inputDataExceptionMask.tif",
        "numberOfLooks.tif",
        "rtcGammaToSigmaFactor.tif"
    ]

    reference_transform = None
    reference_shape = None

    for name in raster_names:
        path = os.path.join(aoi_path, name)

        if not os.path.exists(path):
            print(name, "MISSING")
            continue

        with rasterio.open(path) as src:
            print("\n", name)
            print(" CRS:", src.crs)
            print(" Shape:", src.height, src.width)
            print(" Transform:", src.transform)
            print(" Bounds:", src.bounds)
            print(" Nodata:", src.nodata)

            if reference_transform is None:
                reference_transform = src.transform
                reference_shape = (src.height, src.width)
            else:
                print(" Same transform:",
                      src.transform == reference_transform)
                print(" Same shape:",
                      (src.height, src.width) == reference_shape)

    # Inspect HDF5 cross-covariance and RTC values in chunks
    print("\n--- HDF5 DATASET STATISTICS ---")

    with h5py.File(h5_path, "r") as f:
        base = "/science/SSAR/GCOV/grids/frequencyA/"

        for dataset_name in ["HHHH", "HVHV", "HHHV",
                             "rtcGammaToSigmaFactor"]:
            path = base + dataset_name

            if path not in f:
                print(dataset_name, "NOT FOUND")
                continue

            ds = f[path]
            print("\n", dataset_name)
            print(" Shape:", ds.shape)
            print(" Dtype:", ds.dtype)

            # Sample a small central block, not the full scene
            r0 = ds.shape[0] // 2
            c0 = ds.shape[1] // 2
            block = ds[
                max(0, r0 - 500):r0 + 500,
                max(0, c0 - 500):c0 + 500
            ]

            finite = np.isfinite(block)
            valid = block[finite]

            if valid.size == 0:
                print(" No finite sample values")
                continue

            print(" Finite sample count:", valid.size)

            if np.iscomplexobj(valid):
                print(" Real min/max:",
                      np.min(valid.real), np.max(valid.real))
                print(" Imag min/max:",
                      np.min(valid.imag), np.max(valid.imag))
                print(" Magnitude min/max:",
                      np.min(np.abs(valid)), np.max(np.abs(valid)))
            else:
                print(" Min:", np.min(valid))
                print(" Max:", np.max(valid))
                print(" Mean:", np.mean(valid))

    print("\nFinished:", date)

print("\nVerification complete.")


import os
import glob
import h5py
import numpy as np
import rasterio

H5_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\RAW_HDF5"
AOI_DIR = r"C:\Users\HP\Downloads\NISAR\S_BAND\aoi_outputs"

dates = ["20260819", "20260924", "20261018"]

datasets = {
    "HHHH": "HHHH.tif",
    "HVHV": "HVHV.tif",
    "rtcGammaToSigmaFactor": "rtcGammaToSigmaFactor.tif"
}

for date in dates:
    print("\n" + "=" * 65)
    print("DATE:", date)

    h5_matches = [
        p for p in glob.glob(os.path.join(H5_DIR, "*.h5"))
        if date in os.path.basename(p)
    ]

    if len(h5_matches) != 1:
        print("ERROR: HDF5 match count:", len(h5_matches))
        continue

    h5_path = h5_matches[0]
    folder = os.path.join(AOI_DIR, date)

    with rasterio.open(os.path.join(folder, "HHHH.tif")) as ref:
        transform = ref.transform
        height, width = ref.height, ref.width
        crs = ref.crs

        # AOI upper-left pixel centre
        x = transform.c + transform.a / 2
        y = transform.f + transform.e / 2

    with h5py.File(h5_path, "r") as f:
        base = "/science/SSAR/GCOV/grids/frequencyA/"

        x_coords = f[base + "xCoordinates"]
        y_coords = f[base + "yCoordinates"]

        x_first = float(x_coords[0])
        y_first = float(y_coords[0])

        dx = float(f[base + "xCoordinateSpacing"][()])
        dy = float(f[base + "yCoordinateSpacing"][()])

        col = round((x - x_first) / dx)
        row = round((y - y_first) / dy)

        print("CRS:", crs)
        print("AOI size:", height, width)
        print("HDF5 window row:", row, "column:", col)

        # Confirm calculated pixel centre
        print("Calculated HDF5 pixel centre:",
              x_first + col * dx,
              y_first + row * dy)

        print("\n--- VALUE COMPARISON ---")

        for dataset, tif_name in datasets.items():
            tif_path = os.path.join(folder, tif_name)
            h5_dataset = f[base + dataset]

            h5_block = h5_dataset[
                row:row + height,
                col:col + width
            ]

            with rasterio.open(tif_path) as src:
                tif_block = src.read(1)

            if h5_block.shape != tif_block.shape:
                print(dataset, "SHAPE MISMATCH",
                      h5_block.shape, tif_block.shape)
                continue

            valid = np.isfinite(h5_block) & np.isfinite(tif_block)

            if not np.any(valid):
                print(dataset, "NO COMMON FINITE PIXELS")
                continue

            difference = np.abs(
                h5_block[valid].astype(np.float64) -
                tif_block[valid].astype(np.float64)
            )

            print(dataset)
            print("  Compared pixels:", difference.size)
            print("  Maximum absolute difference:",
                  difference.max())
            print("  Mean absolute difference:",
                  difference.mean())
            print("  Exact matches:",
                  np.count_nonzero(difference == 0))

print("\nWindow verification complete.")


from pathlib import Path
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

BASE = Path(__file__).resolve().parent
RAW = BASE / "RAW_HDF5"
OUT = BASE / "full_scene_geotiffs"

GRID = "science/SSAR/GCOV/grids/frequencyA"
TILE = 512

LAYERS = [
    "HHHH",
    "HVHV",
    "HHHV",
    "mask",
    "inputDataExceptionMask"
]

# Identify acquisition date from the filename.
DATE_CODES = {
    "20260819": "20260819",
    "20260924": "20260924",
    "20261018": "20261018"
}

products = []

for path in RAW.rglob("*.h5"):
    date = next(
        (d for token, d in DATE_CODES.items() if token in path.name),
        None
    )
    if date:
        products.append((date, path))

products.sort()

if len(products) != 3:
    raise RuntimeError(
        f"Expected 3 acquisitions, found {len(products)}"
    )

OUT.mkdir(parents=True, exist_ok=True)

for date, h5_path in products:

    print("\n" + "=" * 70)
    print("PROCESSING:", date)
    print("SOURCE:", h5_path.name)

    date_out = OUT / date
    date_out.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as src:

        x = src[f"{GRID}/xCoordinates"][:]
        y = src[f"{GRID}/yCoordinates"][:]

        height = len(y)
        width = len(x)

        transform = from_origin(
            float(x[0]) - 5,
            float(y[0]) + 5,
            10,
            10
        )

        for layer in LAYERS:

            dataset_path = f"{GRID}/{layer}"

            if dataset_path not in src:
                raise KeyError(
                    f"Missing dataset: {dataset_path}"
                )

            dataset = src[dataset_path]
            dtype = np.dtype(dataset.dtype)

            output_path = date_out / f"{layer}.tif"

            print(
                f"\nLayer: {layer} | "
                f"Shape: {dataset.shape} | "
                f"Type: {dtype}"
            )

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": dtype,
                "crs": "EPSG:32645",
                "transform": transform,
                "tiled": True,
                "blockxsize": TILE,
                "blockysize": TILE,
                "compress": "DEFLATE",
                "BIGTIFF": "YES"
            }

            # Floating-point layers use a floating-point predictor.
            if dtype.kind == "f":
                profile["predictor"] = 3

            with rasterio.open(
                output_path, "w", **profile
            ) as dst:

                for row in range(0, height, TILE):
                    rows = min(TILE, height - row)

                    for col in range(0, width, TILE):
                        cols = min(TILE, width - col)

                        window = Window(
                            col, row, cols, rows
                        )

                        data = dataset[
                            row:row + rows,
                            col:col + cols
                        ]

                        dst.write(data, 1, window=window)

                    print(
                        f"\rRows completed: "
                        f"{min(row + TILE, height)}/{height}",
                        end=""
                    )

            print(f"\nSaved: {output_path}")

    print(f"\nCOMPLETED ACQUISITION: {date}")

print("\nALL THREE FULL-SCENE EXTRACTIONS COMPLETED.")

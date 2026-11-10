
from pathlib import Path
import h5py
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "RAW_HDF5"

GRID = "science/SSAR/GCOV/grids/frequencyA"

products = {}

for path in sorted(RAW.rglob("*.h5")):
    print("\n" + "=" * 65)
    print("PRODUCT:", path.name)

    with h5py.File(path, "r") as f:
        x = f[f"{GRID}/xCoordinates"][:]
        y = f[f"{GRID}/yCoordinates"][:]

        products[path.name] = (x, y)

        print("Raster dimensions:", len(y), "x", len(x))
        print("X extent:", x.min(), "to", x.max())
        print("Y extent:", y.min(), "to", y.max())
        print("X spacing:", np.median(np.abs(np.diff(x))))
        print("Y spacing:", np.median(np.abs(np.diff(y))))
        print("X ascending:", x[-1] > x[0])
        print("Y ascending:", y[-1] > y[0])

print("\n" + "=" * 65)
print("CROSS-DATE GRID COMPARISON")

names = list(products.keys())
reference_x, reference_y = products[names[0]]

for name in names[1:]:
    x, y = products[name]

    print("\nComparing:", name)
    print("Same X coordinates:",
          np.allclose(reference_x, x, rtol=0, atol=0.001))
    print("Same Y coordinates:",
          np.allclose(reference_y, y, rtol=0, atol=0.001))

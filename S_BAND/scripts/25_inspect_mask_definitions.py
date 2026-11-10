
from pathlib import Path
import h5py
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "RAW_HDF5"

GRID = "science/SSAR/GCOV/grids/frequencyA"

def show_attributes(obj, label):
    print(f"\n{label}")
    if not obj.attrs:
        print("  No attributes found")
    for key, value in obj.attrs.items():
        print(f"  {key}: {value}")

for path in sorted(RAW.glob("*.h5")):
    print("\n" + "=" * 75)
    print("PRODUCT:", path.name)

    with h5py.File(path, "r") as f:
        group = f[GRID]
        show_attributes(group, "FREQUENCYA GRID ATTRIBUTES")

        for layer in ["mask", "inputDataExceptionMask"]:
            ds = group[layer]
            show_attributes(ds, f"{layer} ATTRIBUTES")

            print(f"\n{layer} VALUE COUNTS")
            counts = {}

            tile = 1024
            height, width = ds.shape

            for row in range(0, height, tile):
                for col in range(0, width, tile):
                    block = ds[
                        row:min(row + tile, height),
                        col:min(col + tile, width)
                    ]

                    values, frequencies = np.unique(
                        block, return_counts=True
                    )

                    for value, frequency in zip(values, frequencies):
                        key = int(value)
                        counts[key] = counts.get(key, 0) + int(frequency)

            total = height * width
            for value, count in sorted(counts.items()):
                print(
                    f"  Value {value}: {count:,} "
                    f"({count / total * 100:.4f}%)"
                )


from pathlib import Path
import rasterio

BASE = Path(__file__).resolve().parent
ROOT = BASE / "full_scene_geotiffs"

dates = ["20260819", "20260924", "20261018"]
layers = [
    "HHHH", "HVHV", "HHHV",
    "mask", "inputDataExceptionMask"
]

for date in dates:
    print("\n", "=" * 65)
    print("DATE:", date)

    for layer in layers:
        path = ROOT / date / f"{layer}.tif"

        if not path.exists():
            print("MISSING:", path)
            continue

        with rasterio.open(path) as src:
            print(
                f"{layer:24s}",
                src.width, "x", src.height,
                "|", src.dtypes[0],
                "|", src.crs,
                "|", src.transform
            )

print("\nVERIFICATION FINISHED")

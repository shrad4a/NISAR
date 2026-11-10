
from pathlib import Path
import rasterio
from rasterio.windows import Window

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES\S_BAND"
)

CLASSIFICATION_DIR = BASE / "classification"
SHARED_DIR = BASE / "shared"

OUTPUT_DIR = BASE / "validation" / "aligned_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATES = ["20260819", "20260924", "20261018"]

# ============================================================
# ALIGNMENT
# ============================================================

print("\n" + "=" * 65)
print("NISAR CLASSIFICATION MASK ALIGNMENT")
print("=" * 65)

for date in DATES:

    print(f"\nProcessing: {date}")

    classification_path = (
        CLASSIFICATION_DIR / date / "water_mask.tif"
    )

    reference_path = (
        SHARED_DIR / f"reference_labels_{date}.tif"
    )

    output_folder = OUTPUT_DIR / date
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "aligned_water_mask.tif"

    if not classification_path.exists():
        raise FileNotFoundError(classification_path)

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    with rasterio.open(classification_path) as src, \
         rasterio.open(reference_path) as ref:

        # ----------------------------------------------------
        # Verify CRS and resolution
        # ----------------------------------------------------

        if src.crs != ref.crs:
            raise ValueError(
                f"CRS mismatch for {date}: "
                f"{src.crs} vs {ref.crs}"
            )

        if src.res != ref.res:
            raise ValueError(
                f"Resolution mismatch for {date}: "
                f"{src.res} vs {ref.res}"
            )

        # ----------------------------------------------------
        # Calculate reference grid origin in NISAR pixels
        # ----------------------------------------------------

        row_off, col_off = src.index(
            ref.bounds.left,
            ref.bounds.top
        )

        # Confirm the reference origin lies on a NISAR
        # pixel boundary.
        x_check, y_check = src.transform * (
            col_off, row_off
        )

        if (
            abs(x_check - ref.bounds.left) > 1e-6
            or abs(y_check - ref.bounds.top) > 1e-6
        ):
            raise ValueError(
                f"Reference origin is not pixel-aligned for {date}"
            )

        # ----------------------------------------------------
        # Define exact extraction window
        # ----------------------------------------------------

        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=ref.width,
            height=ref.height
        )

        if (
            col_off < 0
            or row_off < 0
            or col_off + ref.width > src.width
            or row_off + ref.height > src.height
        ):
            raise ValueError(
                f"Reference window falls outside NISAR raster: {date}"
            )

        # ----------------------------------------------------
        # Read classification window
        # ----------------------------------------------------

        aligned_data = src.read(
            1,
            window=window
        )

        if aligned_data.shape != (ref.height, ref.width):
            raise ValueError(
                f"Unexpected extracted dimensions for {date}"
            )

        # ----------------------------------------------------
        # Create output raster using reference grid
        # ----------------------------------------------------

        profile = src.profile.copy()

        # Remove tiled block settings that may not fit
        # the smaller output raster.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)

        profile.update(
            driver="GTiff",
            width=ref.width,
            height=ref.height,
            count=1,
            transform=ref.transform,
            crs=ref.crs,
            compress="deflate",
            tiled=False
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(aligned_data, 1)

        # ----------------------------------------------------
        # Verify output against reference
        # ----------------------------------------------------

        with rasterio.open(output_path) as aligned:

            checks = {
                "CRS": aligned.crs == ref.crs,
                "Dimensions": (
                    aligned.width == ref.width
                    and aligned.height == ref.height
                ),
                "Resolution": aligned.res == ref.res,
                "Transform": aligned.transform == ref.transform,
                "Bounds": aligned.bounds == ref.bounds
            }

            if not all(checks.values()):
                raise ValueError(
                    f"Output grid verification failed: {date}, {checks}"
                )

        print(f"  Extraction window: row={row_off}, col={col_off}")
        print(f"  Output dimensions: {ref.width} x {ref.height}")
        print(f"  CRS: {ref.crs}")
        print(f"  Exact grid match: PASSED")
        print(f"  Saved: {output_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ALL THREE CLASSIFICATION MASKS ALIGNED SUCCESSFULLY")
print("=" * 65)

print("\nOriginal classification masks were not modified.")
print("No interpolation or resampling was performed.")
print(f"\nAligned masks saved in:\n{OUTPUT_DIR}")

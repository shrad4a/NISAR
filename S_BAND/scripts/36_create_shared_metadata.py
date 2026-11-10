
import json
from pathlib import Path
import rasterio

# ============================================================
# PROJECT PATHS
# ============================================================

SHARED = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES\S_BAND\shared"
)

SHARED.mkdir(parents=True, exist_ok=True)

# ============================================================
# SENTINEL-2 REFERENCE SCENE DETAILS
# ============================================================

dates = {
    "20260819": "20260819T044701_20260819T045130_T45RVJ",
    "20260924": "20260928T044701_20260928T045730_T45RVJ",
    "20261018": "20261018T044811_20261018T050037_T45RVJ"
}

metadata_records = []
grid_records = []

# ============================================================
# READ TIFF METADATA
# ============================================================

for date, scene_id in dates.items():

    tif_path = SHARED / f"reference_labels_{date}.tif"

    if not tif_path.exists():
        raise FileNotFoundError(
            f"\nMissing reference TIFF:\n{tif_path}\n\n"
            f"Place the file in the shared folder and name it:\n"
            f"reference_labels_{date}.tif"
        )

    with rasterio.open(tif_path) as src:

        record = {
            "date": date,
            "source": "Sentinel-2 SR Harmonized",
            "scene_id": scene_id,
            "filename": tif_path.name,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": list(src.res),
            "bands": src.count,
            "data_type": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            },
            "transform": list(src.transform)
        }

        metadata_records.append(record)

        grid_records.append({
            "date": date,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": list(src.res),
            "transform": list(src.transform),
            "bounds": list(src.bounds)
        })

# ============================================================
# VERIFY COMMON GRID
# ============================================================

first = grid_records[0]

for grid in grid_records[1:]:

    for key in [
        "crs",
        "width",
        "height",
        "resolution",
        "transform",
        "bounds"
    ]:

        if grid[key] != first[key]:

            raise ValueError(
                f"Grid mismatch for {grid['date']}: {key}"
            )

# ============================================================
# CREATE reference_metadata.txt
# ============================================================

txt_path = SHARED / "reference_metadata.txt"

with open(txt_path, "w", encoding="utf-8") as f:

    f.write("SENTINEL-2 REFERENCE DATA METADATA\n")
    f.write("=" * 45 + "\n\n")

    f.write("Data source: COPERNICUS/S2_SR_HARMONIZED\n")
    f.write("Reference method: MNDWI threshold classification\n")
    f.write("MNDWI threshold: Greater than 0\n")
    f.write("Cloud/shadow masking: Sentinel-2 SCL\n\n")

    f.write("Pixel label convention:\n")
    f.write("0   = Non-water\n")
    f.write("1   = Water\n")
    f.write("255 = Invalid/cloud/shadow\n\n")

    f.write("Reference acquisitions:\n\n")

    for record in metadata_records:

        f.write(f"Date: {record['date']}\n")
        f.write(f"Scene ID: {record['scene_id']}\n")
        f.write(f"File: {record['filename']}\n")
        f.write(f"CRS: {record['crs']}\n")

        f.write(
            f"Dimensions: {record['width']} x "
            f"{record['height']}\n"
        )

        f.write(f"Resolution: {record['resolution']}\n")
        f.write(f"NoData metadata: {record['nodata']}\n\n")

    f.write(
        "Note: These are MNDWI-derived reference labels, "
        "not independently verified ground truth.\n"
    )

# ============================================================
# CREATE common_grid_specification.json
# ============================================================

grid_spec = {

    "description": (
        "Common Sentinel-2 reference raster grid "
        "used for spatial comparison with NISAR S-band."
    ),

    "reference_grid": {
        "crs": first["crs"],
        "width": first["width"],
        "height": first["height"],
        "resolution": first["resolution"],
        "transform": first["transform"],
        "bounds": first["bounds"]
    },

    "reference_dates": list(dates.keys()),

    "pixel_labels": {
        "0": "Non-water",
        "1": "Water",
        "255": "Invalid; excluded from validation"
    },

    "alignment": {

        "reference_grid_source": "Sentinel-2 GeoTIFF",

        "comparison": (
            "Spatial comparison must use matching geographic "
            "locations and exclude invalid reference pixels."
        ),

        "resampling": (
            "Categorical labels must use nearest-neighbour "
            "resampling if reprojection is required."
        )
    },

    "per_date_grid_verification": grid_records
}

json_path = SHARED / "common_grid_specification.json"

with open(json_path, "w", encoding="utf-8") as f:

    json.dump(grid_spec, f, indent=4)

# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 55)
print("SHARED METADATA CREATED SUCCESSFULLY")
print("=" * 55)

print(f"\n1. {txt_path}")
print(f"2. {json_path}")

print("\nVerified reference acquisitions:")

for record in metadata_records:
    print(
        f"  {record['date']} | "
        f"{record['width']} x {record['height']} | "
        f"{record['crs']}"
    )

print("\nAll three reference grids match.")
print("=" * 55)

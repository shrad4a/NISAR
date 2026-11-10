
from pathlib import Path
import rasterio

# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\HP\Downloads\NISAR\S_BAND\TEAM_DELIVERABLES\S_BAND"
)

CLASSIFICATION_DIR = BASE / "classification"
SHARED_DIR = BASE / "shared"

DATES = ["20260819", "20260924", "20261018"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_metadata(path):
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": tuple(src.res),
            "transform": tuple(src.transform),
            "bounds": src.bounds,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "count": src.count
        }


def compare_bounds(a, b):
    left = max(a.left, b.left)
    right = min(a.right, b.right)
    bottom = max(a.bottom, b.bottom)
    top = min(a.top, b.top)

    if left >= right or bottom >= top:
        return None

    return {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "width": right - left,
        "height": top - bottom
    }


# ============================================================
# AUDIT
# ============================================================

results = []

print("\n" + "=" * 65)
print("NISAR S-BAND vs SENTINEL-2 SPATIAL ALIGNMENT AUDIT")
print("=" * 65)

for date in DATES:

    classification_path = (
        CLASSIFICATION_DIR / date / "water_mask.tif"
    )

    reference_path = (
        SHARED_DIR / f"reference_labels_{date}.tif"
    )

    print(f"\nDATE: {date}")
    print("-" * 65)

    if not classification_path.exists():
        print(f"MISSING CLASSIFICATION: {classification_path}")
        continue

    if not reference_path.exists():
        print(f"MISSING REFERENCE: {reference_path}")
        continue

    c = get_metadata(classification_path)
    r = get_metadata(reference_path)

    same_crs = c["crs"] == r["crs"]
    same_dimensions = (
        c["width"] == r["width"]
        and c["height"] == r["height"]
    )
    same_resolution = c["resolution"] == r["resolution"]
    same_transform = c["transform"] == r["transform"]
    overlap = compare_bounds(
        rasterio.coords.BoundingBox(*c["bounds"]),
        rasterio.coords.BoundingBox(*r["bounds"])
    ) if same_crs else None

    exact_grid_match = (
        same_crs
        and same_dimensions
        and same_resolution
        and same_transform
    )

    print("\nCLASSIFICATION MASK:")
    print(f"  CRS: {c['crs']}")
    print(f"  Dimensions: {c['width']} x {c['height']}")
    print(f"  Resolution: {c['resolution']}")
    print(f"  Bounds: {c['bounds']}")
    print(f"  Transform: {c['transform']}")

    print("\nREFERENCE MASK:")
    print(f"  CRS: {r['crs']}")
    print(f"  Dimensions: {r['width']} x {r['height']}")
    print(f"  Resolution: {r['resolution']}")
    print(f"  Bounds: {r['bounds']}")
    print(f"  Transform: {r['transform']}")

    print("\nCOMPARISON:")
    print(f"  Same CRS: {same_crs}")
    print(f"  Same dimensions: {same_dimensions}")
    print(f"  Same resolution: {same_resolution}")
    print(f"  Same transform: {same_transform}")
    print(f"  Exact grid match: {exact_grid_match}")

    if overlap:
        print("\nGEOGRAPHIC OVERLAP:")
        print(f"  Width: {overlap['width']:.2f} map units")
        print(f"  Height: {overlap['height']:.2f} map units")
    elif same_crs:
        print("\nGEOGRAPHIC OVERLAP: NONE")
    else:
        print("\nGEOGRAPHIC OVERLAP: Cannot compare directly; CRS differs.")

    if exact_grid_match:
        status = "EXACT GRID MATCH"
    elif overlap:
        status = "OVERLAP EXISTS - ALIGNMENT REQUIRED"
    else:
        status = "NO DIRECT SPATIAL MATCH"

    print(f"\nSTATUS: {status}")

    results.append({
        "date": date,
        "same_crs": same_crs,
        "same_dimensions": same_dimensions,
        "same_resolution": same_resolution,
        "same_transform": same_transform,
        "exact_grid_match": exact_grid_match,
        "overlap_exists": overlap is not None
    })

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL ALIGNMENT SUMMARY")
print("=" * 65)

for item in results:
    print(
        f"{item['date']}: "
        f"{'EXACT MATCH' if item['exact_grid_match'] else 'CHECK REQUIRED'}"
    )

print("\nAudit completed. No raster files were modified.")

# Save audit report
report_path = BASE / "validation" / "spatial_alignment_audit.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("NISAR S-BAND SPATIAL ALIGNMENT AUDIT\n")
    f.write("=" * 50 + "\n\n")

    for item in results:
        f.write(f"Date: {item['date']}\n")
        for key, value in item.items():
            if key != "date":
                f.write(f"{key}: {value}\n")
        f.write("\n")

print(f"\nReport saved to:\n{report_path}")

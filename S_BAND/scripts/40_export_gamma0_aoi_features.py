
from pathlib import Path

import numpy as np
import rasterio

SOURCE = Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\aoi_outputs")
OUTPUT = Path(r"C:\Users\HP\Downloads\NISAR\S_BAND\gamma0_features")

DATES = ["20260819", "20260924", "20261018"]
NODATA = -9999.0

for date in DATES:
    src_dir = SOURCE / date
    out_dir = OUTPUT / date
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_dir / "mask.tif") as ds:
        quality_mask = ds.read(1)

    with rasterio.open(src_dir / "inputDataExceptionMask.tif") as ds:
        exception_mask = ds.read(1)

    # Retain fully focused pixels and the documented sample-shift flag.
    valid_quality = quality_mask == 1
    valid_exception = np.isin(exception_mask, [0, 64])
    valid = valid_quality & valid_exception

    for polarization in ["HHHH", "HVHV"]:
        input_path = src_dir / f"{polarization}.tif"

        with rasterio.open(input_path) as src:
            gamma0 = src.read(1).astype(np.float32)
            profile = src.profile.copy()

        if gamma0.shape != valid.shape:
            raise ValueError(f"Grid mismatch for {date} {polarization}")

        valid_pixels = valid & np.isfinite(gamma0) & (gamma0 > 0)

        if not np.any(valid_pixels):
            raise ValueError(f"No valid positive pixels: {date} {polarization}")

        linear = np.full(gamma0.shape, NODATA, dtype=np.float32)
        db = np.full(gamma0.shape, NODATA, dtype=np.float32)

        linear[valid_pixels] = gamma0[valid_pixels]
        db[valid_pixels] = 10.0 * np.log10(gamma0[valid_pixels])

        output_profile = profile.copy()
        output_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

        for representation, array in [
            ("linear", linear),
            ("db", db)
        ]:
            output_path = (
                out_dir / f"{polarization}_gamma0_{representation}.tif"
            )

            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(array, 1)
                dst.set_band_description(
                    1, f"{polarization} gamma0 {representation}"
                )
                dst.update_tags(
                    acquisition_date=date,
                    polarization=polarization,
                    quantity="gamma0",
                    representation=representation,
                    db_formula="10*log10(linear_gamma0)" if representation == "db" else "none",
                    source=input_path.name,
                    valid_mask="mask == 1",
                    retained_exception_flags="0,64",
                    nodata_value=str(NODATA)
                )

            print(f"CREATED: {output_path}")

    print(f"COMPLETED: {date}")

print("\nALL THREE ACQUISITIONS PROCESSED.")

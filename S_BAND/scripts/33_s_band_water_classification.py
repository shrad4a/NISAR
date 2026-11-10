
import os
import glob
import h5py
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from scipy.special import expit

# ============================================================
# SCRIPT 33: S-BAND WATER CLASSIFICATION
# ============================================================

BASE = r"C:\Users\HP\Downloads\NISAR\S_BAND"

FEATURE_DIR = os.path.join(
    BASE, "TEAM_DELIVERABLES", "S_BAND",
    "extracted_features"
)

RAW_DIR = os.path.join(BASE, "RAW_HDF5")

OUTPUT_DIR = os.path.join(
    BASE, "TEAM_DELIVERABLES", "S_BAND",
    "classification"
)

DATES = ["20260819", "20260924", "20261018"]

BLOCK = 1024
RADIUS = 25
WINDOW_SIZE = 2 * RADIUS + 1
HALO = RADIUS

NODATA = -9999.0

# Provisional rule-based parameters.
# These must be checked against independent reference labels.
SCORE_THRESHOLD = 0.60
TEXTURE_MAX_DB = 4.5
SPATIAL_SUPPORT_MIN = 0.45

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND ORIGINAL HDF5
# ============================================================

def find_hdf5(date):

    patterns = [
        os.path.join(RAW_DIR, f"*{date}*.h5"),
        os.path.join(RAW_DIR, f"*{date}*.hdf5")
    ]

    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No HDF5 found for {date} in {RAW_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple HDF5 files found for {date}. "
            f"Resolve duplicates before running: {matches}"
        )

    return matches[0]


# ============================================================
# MASK-AWARE LOCAL STATISTICS
# ============================================================

def local_statistics(data, valid):

    valid_float = valid.astype(np.float32)

    values = np.where(valid, data, 0).astype(np.float32)
    squares = values * values

    count = uniform_filter(
        valid_float,
        size=WINDOW_SIZE,
        mode="nearest"
    )

    safe_count = np.maximum(count, 1e-6)

    mean = uniform_filter(
        values,
        size=WINDOW_SIZE,
        mode="nearest"
    ) / safe_count

    mean_square = uniform_filter(
        squares,
        size=WINDOW_SIZE,
        mode="nearest"
    ) / safe_count

    variance = np.maximum(
        mean_square - mean * mean,
        0
    )

    std = np.sqrt(variance)

    return mean, std, count


# ============================================================
# PROCESS ACQUISITIONS
# ============================================================

for date in DATES:

    print("\n" + "=" * 65)
    print("CLASSIFYING:", date)
    print("=" * 65)

    feature_path = os.path.join(
        FEATURE_DIR,
        f"available_feature_layers_{date}.tif"
    )

    hdf5_path = find_hdf5(date)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)

    date_output = os.path.join(OUTPUT_DIR, date)
    os.makedirs(date_output, exist_ok=True)

    mask_path = (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/mask"
    )

    exception_path = (
        "/science/SSAR/GCOV/grids/"
        "frequencyA/inputDataExceptionMask"
    )

    with rasterio.open(feature_path) as src, \
         h5py.File(hdf5_path, "r") as h5:

        if src.count != 5:
            raise ValueError(
                f"{date}: expected 5 feature bands, "
                f"found {src.count}"
            )

        if mask_path not in h5 or exception_path not in h5:
            raise KeyError(
                "Original HDF5 quality masks are missing"
            )

        ds_mask = h5[mask_path]
        ds_exception = h5[exception_path]

        if ds_mask.shape != (src.height, src.width):
            raise ValueError("NISAR mask dimensions do not match")

        if ds_exception.shape != (src.height, src.width):
            raise ValueError("Exception mask dimensions do not match")

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER"
        )

        mask_output = os.path.join(
            date_output, "water_mask.tif"
        )

        probability_output = os.path.join(
            date_output, "water_probability.tif"
        )

        # Probability filename retained for the requested
        # deliverable. Values are uncalibrated confidence scores.
        probability_profile = profile.copy()
        probability_profile.update(
            dtype="float32",
            nodata=NODATA,
            predictor=3
        )

        with rasterio.open(
            mask_output, "w", **profile
        ) as mask_dst, rasterio.open(
            probability_output, "w", **probability_profile
        ) as score_dst:

            mask_dst.set_band_description(
                1, "Rule-based water classification: 1=water, 0=non-water"
            )

            score_dst.set_band_description(
                1, "Uncalibrated water-confidence score, 0 to 1"
            )

            height = src.height
            width = src.width

            for row in range(0, height, BLOCK):

                for col in range(0, width, BLOCK):

                    h = min(BLOCK, height - row)
                    w = min(BLOCK, width - col)

                    # Read a halo around the processing block.
                    r0 = max(0, row - HALO)
                    c0 = max(0, col - HALO)

                    r1 = min(height, row + h + HALO)
                    c1 = min(width, col + w + HALO)

                    read_window = Window(
                        c0, r0, c1 - c0, r1 - r0
                    )

                    # Band 3: HH dB
                    # Band 4: HV dB
                    hh_db = src.read(
                        3, window=read_window
                    ).astype(np.float32)

                    hv_db = src.read(
                        4, window=read_window
                    ).astype(np.float32)

                    slices = (
                        slice(r0, r1),
                        slice(c0, c1)
                    )

                    mask = np.asarray(ds_mask[slices])
                    exception = np.asarray(
                        ds_exception[slices]
                    )

                    # Retain clean and SSAR-shift flagged
                    # samples, but exclude other flag values.
                    valid = (
                        (mask == 1) &
                        np.isin(exception, [0, 64]) &
                        np.isfinite(hh_db) &
                        np.isfinite(hv_db) &
                        (hh_db != NODATA) &
                        (hv_db != NODATA)
                    )

                    # Local adaptive HH and HV statistics.
                    hh_mean, hh_std, local_count = (
                        local_statistics(hh_db, valid)
                    )

                    hv_mean, hv_std, _ = (
                        local_statistics(hv_db, valid)
                    )

                    # Darker-than-neighbourhood evidence.
                    z_hh = (
                        (hh_mean - hh_db) /
                        np.maximum(hh_std, 0.5)
                    )

                    z_hv = (
                        (hv_mean - hv_db) /
                        np.maximum(hv_std, 0.5)
                    )

                    # Combined polarimetric confidence score.
                    score = (
                        0.65 * expit(z_hh) +
                        0.35 * expit(z_hv)
                    )

                    # Texture constraint based on local HH
                    # standard deviation.
                    texture_ok = hh_std <= TEXTURE_MAX_DB

                    raw_candidate = (
                        valid &
                        (score >= SCORE_THRESHOLD) &
                        texture_ok
                    )

                    # Spatial support: fraction of neighbouring
                    # valid pixels that are also candidates.
                    candidate_fraction = uniform_filter(
                        raw_candidate.astype(np.float32),
                        size=3,
                        mode="nearest"
                    )

                    valid_fraction = uniform_filter(
                        valid.astype(np.float32),
                        size=3,
                        mode="nearest"
                    )

                    support = (
                        candidate_fraction /
                        np.maximum(valid_fraction, 1e-6)
                    )

                    final_water = (
                        raw_candidate &
                        (support >= SPATIAL_SUPPORT_MIN)
                    )

                    # Crop away the halo.
                    crop_r0 = row - r0
                    crop_c0 = col - c0

                    crop = (
                        slice(crop_r0, crop_r0 + h),
                        slice(crop_c0, crop_c0 + w)
                    )

                    final_crop = final_water[crop]
                    valid_crop = valid[crop]
                    score_crop = score[crop]

                    mask_block = np.full(
                        (h, w), 255, dtype=np.uint8
                    )

                    mask_block[valid_crop] = 0
                    mask_block[final_crop] = 1

                    score_block = np.full(
                        (h, w), NODATA, dtype=np.float32
                    )

                    score_block[valid_crop] = (
                        score_crop[valid_crop]
                    )

                    write_window = Window(col, row, w, h)

                    mask_dst.write(
                        mask_block, 1, window=write_window
                    )

                    score_dst.write(
                        score_block, 1, window=write_window
                    )

                print(
                    f"{date}: processed "
                    f"{min(row + BLOCK, height)}/{height} rows"
                )

    # ========================================================
    # BACKSCATTER MAP
    # ========================================================

    backscatter_path = os.path.join(
        date_output, "backscatter_map.png"
    )

    with rasterio.open(feature_path) as src:

        scale = max(
            src.width / 1600,
            src.height / 1600,
            1
        )

        out_h = max(1, int(src.height / scale))
        out_w = max(1, int(src.width / scale))

        overview = src.read(
            3,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.average
        ).astype(np.float32)

        if src.nodata is not None:
            overview[overview == src.nodata] = np.nan

    plt.figure(figsize=(10, 8))

    image = plt.imshow(
        overview,
        cmap="gray",
        vmin=-30,
        vmax=0
    )

    plt.title(f"S-band HH Backscatter: {date}")
    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.colorbar(
        image,
        label="HH backscatter (dB)"
    )

    plt.tight_layout()
    plt.savefig(
        backscatter_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nCompleted:", date)
    print("Water mask:", mask_output)
    print("Confidence:", probability_output)
    print("Backscatter:", backscatter_path)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("SCRIPT 33 COMPLETED")
print("=" * 65)
print("Output:", OUTPUT_DIR)
print("All three acquisitions processed.")

"""
11_stage_B_water_candidates.py

Stage B of the Quality-Gated Polarimetric and Temporal Water Mapping algorithm.

Purpose:
  Generate LOW-BACKSCATTER WATER CANDIDATES for all three S-band acquisitions.
  These are candidates only, NOT final water classifications.

Input folders:
  aoi_outputs/20260819/
  aoi_outputs/20260924/
  aoi_outputs/20261018/

Required files in each folder:
  HHHH.npy
  mask.npy
  inputDataExceptionMask.npy
  and (for acquisition quality treatment) the Stage A quality flag file:
  acquisition_quality_output/<DATE>_quality_flags.npy

Method:
  1. Convert positive HHHH Gamma0 values to dB: 10*log10(HHHH).
  2. Use one COMMON threshold for all dates, derived from pooled August+October
     HH dB values using a reproducible histogram-based Otsu split.
  3. Apply that same threshold to all three dates.
  4. Save raw candidates and quality-aware candidate evidence separately.
  5. September candidates are retained but tagged FLAGGED_ACQUISITION.
  6. Do not use this candidate mask as a final water map.

Important:
  - Otsu is only an initial data-driven candidate-separation threshold. It does
    NOT prove that the lower-backscatter population is water.
  - The threshold is estimated from August and October because September has a
    known HV/HH anomaly. September is still classified using the exact same
    threshold, not a separately tuned threshold.
  - This script uses HH only for candidate generation. Polarimetric validation
    is Stage C.
  - It does not apply rtcGammaToSigmaFactor. The input HHHH is the product's
    stored Gamma0 term; do not convert it again without verifying the product
    convention and calibration workflow.
  - It records exception-mask codes but does not automatically reject code 64.

Run:
  python 11_stage_B_water_candidates.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------- CONFIG -------------------------------

BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
STAGE_A_DIR = BASE_DIR / "acquisition_quality_output"
OUTPUT_DIR = BASE_DIR / "stage_B_candidate_output"

DATES = ["20260819", "20260924", "20261018"]
REFERENCE_DATES = ["20260819", "20261018"]  # exclude flagged Sep from threshold fitting

# Histogram-based Otsu configuration. The candidate threshold is learned once
# from the pooled reference-date HH dB distribution.
HIST_BINS = 512
HIST_RANGE_DB = (-40.0, 15.0)

# Avoid extreme values in threshold estimation; preserve all finite valid pixels
# for applying the threshold and report how many lie outside the histogram range.
SAMPLE_MAX_PER_DATE = 1_000_000
RANDOM_SEED = 42

# Stage A pixel flag values: 0 invalid, 1 ACCEPT acquisition, 2 FLAG/REJECT acquisition.
STAGE_A_FLAG_VALUES = {"invalid": 0, "accepted": 1, "flagged": 2}


# ------------------------------- HELPERS ------------------------------

def load_npy(folder: Path, name: str):
    path = folder / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return np.load(path, mmap_mode="r")


def valid_hh_mask(hh, source_mask):
    # Product mask: 255 outside acquisition, 0 invalid/not fully focused,
    # nonzero values other than 255 are valid subswath identifiers.
    return (
        (source_mask != 0)
        & (source_mask != 255)
        & np.isfinite(hh)
        & (hh > 0)
    )


def to_db(hh):
    out = np.full(hh.shape, np.nan, dtype=np.float32)
    good = np.isfinite(hh) & (hh > 0)
    out[good] = (10.0 * np.log10(hh[good])).astype(np.float32)
    return out


def otsu_threshold_from_histogram(values, bins=512, value_range=(-40, 15)):
    """Calculate an Otsu split from finite 1-D values using a fixed histogram."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("Not enough valid HH dB samples to estimate threshold.")

    hist, edges = np.histogram(values, bins=bins, range=value_range)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        raise ValueError("No samples fall inside the configured histogram range.")

    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg

    sum_bg = np.cumsum(hist * centers)
    sum_total = sum_bg[-1]
    mean_bg = np.divide(
        sum_bg, weight_bg, out=np.zeros_like(sum_bg), where=weight_bg > 0
    )
    mean_fg = np.divide(
        sum_total - sum_bg,
        weight_fg,
        out=np.zeros_like(sum_bg),
        where=weight_fg > 0,
    )

    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[(weight_bg == 0) | (weight_fg == 0)] = -1
    best_idx = int(np.argmax(between))
    threshold = float(centers[best_idx])
    return threshold, hist, edges


def deterministic_sample(values, max_count, rng):
    values = np.asarray(values)
    if values.size <= max_count:
        return values
    indices = rng.choice(values.size, size=max_count, replace=False)
    return values[indices]


def load_stage_a_flag(date):
    path = STAGE_A_DIR / f"{date}_quality_flags.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage A flag file: {path}\n"
            "Run 10_acquisition_quality_gate.py first."
        )
    return np.load(path, mmap_mode="r")


def percentile_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {}
    return {
        "p02_dB": float(np.percentile(values, 2)),
        "median_dB": float(np.percentile(values, 50)),
        "p98_dB": float(np.percentile(values, 98)),
        "mean_dB": float(np.mean(values)),
    }


# ------------------------- THRESHOLD ESTIMATION -----------------------

def estimate_common_threshold():
    rng = np.random.default_rng(RANDOM_SEED)
    samples = []

    for date in REFERENCE_DATES:
        folder = AOI_DIR / date
        hh = np.asarray(load_npy(folder, "HHHH"), dtype=np.float32)
        source_mask = np.asarray(load_npy(folder, "mask"))
        valid = valid_hh_mask(hh, source_mask)
        hh_db = to_db(hh)
        vals = hh_db[valid & np.isfinite(hh_db)]
        vals = deterministic_sample(vals, SAMPLE_MAX_PER_DATE, rng)
        samples.append(vals)
        print(
            f"Threshold training {date}: {vals.size:,} sampled valid HH pixels"
        )

    pooled = np.concatenate(samples)
    threshold, hist, edges = otsu_threshold_from_histogram(
        pooled, HIST_BINS, HIST_RANGE_DB
    )
    in_range_fraction = float(
        np.mean((pooled >= HIST_RANGE_DB[0]) & (pooled <= HIST_RANGE_DB[1]))
    )
    return threshold, pooled, hist, edges, in_range_fraction


# ------------------------------ MAIN ----------------------------------

def main():
    if not AOI_DIR.exists():
        raise FileNotFoundError(f"AOI folder not found: {AOI_DIR}")
    if not STAGE_A_DIR.exists():
        raise FileNotFoundError(f"Stage A output folder not found: {STAGE_A_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_db, training_sample, hist, edges, in_range_fraction = (
        estimate_common_threshold()
    )

    print("\n========== COMMON CANDIDATE THRESHOLD ==========")
    print(f"Otsu HH threshold: {threshold_db:.3f} dB")
    print(f"Reference dates: {', '.join(REFERENCE_DATES)}")
    print(f"Training sample inside histogram range: {in_range_fraction:.2%}")

    # Save threshold-estimation histogram.
    plt.figure(figsize=(9, 5))
    plt.hist(training_sample, bins=HIST_BINS, range=HIST_RANGE_DB)
    plt.axvline(
        threshold_db, linestyle="--", linewidth=2,
        label=f"Common Otsu threshold = {threshold_db:.2f} dB"
    )
    plt.xlabel("HH Gamma0 (dB)")
    plt.ylabel("Pixel count")
    plt.title("Stage B: pooled reference-date HH threshold estimation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "common_HH_threshold_histogram.png", dpi=200)
    plt.close()

    rows = []
    run_report = {
        "stage": "B - Low-backscatter water candidate generation",
        "candidate_definition": "HH Gamma0 dB <= common pooled-reference Otsu threshold",
        "threshold_db": threshold_db,
        "threshold_method": "Histogram-based Otsu on pooled August and October HH dB samples",
        "reference_dates": REFERENCE_DATES,
        "all_dates_apply_same_threshold": True,
        "september_treatment": (
            "Candidate mask generated with same threshold; acquisition remains "
            "FLAGGED and must not be trusted for quantitative temporal comparison."
        ),
        "interpretation_warning": (
            "Low backscatter is candidate evidence only. This output is not a "
            "validated water classification."
        ),
        "acquisitions": {},
    }

    for date in DATES:
        print(f"\nProcessing {date} ...")
        folder = AOI_DIR / date

        hh = np.asarray(load_npy(folder, "HHHH"), dtype=np.float32)
        source_mask = np.asarray(load_npy(folder, "mask"))
        exception = np.asarray(load_npy(folder, "inputDataExceptionMask"))
        stage_a_flag = np.asarray(load_stage_a_flag(date))

        if not (hh.shape == source_mask.shape == exception.shape == stage_a_flag.shape):
            raise ValueError(
                f"{date}: HHHH, mask, exception mask and Stage A flag shapes differ."
            )

        valid = valid_hh_mask(hh, source_mask) & (stage_a_flag != 0)
        hh_db = to_db(hh)

        # Low HH backscatter creates candidates only.
        candidate = valid & np.isfinite(hh_db) & (hh_db <= threshold_db)

        # Preserve quality treatment in a separate evidence mask:
        # 0 invalid, 1 accepted-acquisition candidate, 2 flagged-acquisition candidate.
        candidate_evidence = np.zeros(hh.shape, dtype=np.uint8)
        accepted_candidate = candidate & (stage_a_flag == 1)
        flagged_candidate = candidate & (stage_a_flag == 2)
        candidate_evidence[accepted_candidate] = 1
        candidate_evidence[flagged_candidate] = 2

        # Candidate map: 0=not candidate/invalid, 1=candidate.
        candidate_mask = candidate.astype(np.uint8)

        date_out = OUTPUT_DIR / date
        date_out.mkdir(parents=True, exist_ok=True)
        np.save(date_out / "HH_dB.npy", hh_db)
        np.save(date_out / "water_candidate_mask.npy", candidate_mask)
        np.save(date_out / "candidate_evidence_quality.npy", candidate_evidence)

        # Plot HH dB and candidate overlay as separate maps.
        plt.figure(figsize=(9, 7))
        im = plt.imshow(hh_db, cmap="gray", vmin=-30, vmax=5)
        plt.colorbar(im, label="HH Gamma0 (dB)")
        plt.title(f"{date}: HH backscatter (dB)")
        plt.xlabel("AOI column")
        plt.ylabel("AOI row")
        plt.tight_layout()
        plt.savefig(date_out / "HH_backscatter_dB.png", dpi=200)
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.imshow(candidate_mask, cmap="gray", vmin=0, vmax=1)
        plt.title(
            f"{date}: low-backscatter candidates\n"
            f"Common threshold = {threshold_db:.2f} dB"
        )
        plt.xlabel("AOI column")
        plt.ylabel("AOI row")
        plt.tight_layout()
        plt.savefig(date_out / "candidate_map.png", dpi=200)
        plt.close()

        # Statistics separated by acquisition quality.
        ratio_candidate = float(candidate.sum() / valid.sum()) if valid.any() else float("nan")
        exc64_candidate = (
            float(np.mean(exception[candidate] == 64)) if candidate.any() else float("nan")
        )

        stats = {
            "date": date,
            "stage_A_acquisition_status": (
                "ACCEPT" if np.any(stage_a_flag == 1) and not np.any(stage_a_flag == 2)
                else "FLAGGED_ACQUISITION" if np.any(stage_a_flag == 2)
                else "CHECK_STAGE_A_FLAGS"
            ),
            "common_HH_threshold_dB": threshold_db,
            "valid_pixel_count": int(valid.sum()),
            "candidate_pixel_count": int(candidate.sum()),
            "candidate_fraction_of_valid": ratio_candidate,
            "accepted_candidate_count": int(accepted_candidate.sum()),
            "flagged_candidate_count": int(flagged_candidate.sum()),
            "candidate_HH_statistics": percentile_summary(hh_db[candidate]),
            "all_valid_HH_statistics": percentile_summary(hh_db[valid]),
            "candidate_fraction_with_exception_code_64": exc64_candidate,
            "interpretation": (
                "Low-backscatter candidate only; not confirmed water."
            ),
        }
        rows.append({
            k: json.dumps(v) if isinstance(v, dict) else v
            for k, v in stats.items()
        })
        run_report["acquisitions"][date] = stats

        print(f"  Valid pixels: {valid.sum():,}")
        print(f"  Candidate pixels: {candidate.sum():,}")
        print(f"  Candidate fraction: {ratio_candidate:.2%}")
        print(f"  Stage A flagged candidate pixels: {flagged_candidate.sum():,}")

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "candidate_statistics.csv", index=False
    )
    with open(OUTPUT_DIR / "stage_B_report.json", "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, allow_nan=True)

    print("\n========== STAGE B COMPLETE ==========")
    print(f"Common threshold: {threshold_db:.3f} dB")
    print(f"Outputs: {OUTPUT_DIR}")
    print("Reminder: candidates are NOT final water classifications.")


if __name__ == "__main__":
    main()

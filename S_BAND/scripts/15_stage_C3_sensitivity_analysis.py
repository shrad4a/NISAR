"""
15_stage_C3_sensitivity_analysis.py

Stage C.3: Threshold sensitivity and classification robustness.

Processes ALL THREE dates:
  20260819 (reference / ACCEPT)
  20260924 (diagnostic / FLAG)
  20261018 (reference / ACCEPT)

Uses the exact Stage C.1 feature arrays and Stage B candidate masks.
It does NOT silently reclassify September as a normal polarimetric date:
September HH + spatial evidence is retained, while HV and HV/HH remain
excluded from its score because its Stage A quality flag is FLAG.

Sensitivity design:
  - pooled-reference low-backscatter quantile q = 0.40, 0.50, 0.60
  - local spatial-support fraction = 0.40, 0.50, 0.60
  - all three feature quantiles (HH, HV, HV/HH) use the same q in each run
  - accepted dates: water-supported score >= 3; non-water-supported <= 1
  - September: water-supported only when HH + spatial evidence both agree;
    no non-water class is assigned from its incomplete feature set.

Outputs:
  stage_C3_sensitivity_output/
    sensitivity_runs.csv
    date_sensitivity_summary.csv
    pixel_class_stability.csv
    reference_thresholds_by_quantile.json
    robust_water_support_frequency.npy
    robust_water_support_frequency.png
    september_candidate_frequency.npy
    september_candidate_frequency.png
    aug_oct_persistent_water_frequency.npy
    aug_oct_persistent_water_frequency.png
    stage_C3_report.json

Run from the project folder:
  python 15_stage_C3_sensitivity_analysis.py

This is a robustness test of exploratory evidence classes, not accuracy
validation. Independent reference labels are still required.
"""

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
AOI_DIR = BASE_DIR / "aoi_outputs"
STAGE_A_DIR = BASE_DIR / "acquisition_quality_output"
STAGE_B_DIR = BASE_DIR / "stage_B_candidate_output"
C1_DIR = BASE_DIR / "stage_C1_validation_output"
OUTPUT_DIR = BASE_DIR / "stage_C3_sensitivity_output"

DATES = ["20260819", "20260924", "20261018"]
REFERENCE_DATES = ["20260819", "20261018"]
QUANTILES = [0.40, 0.50, 0.60]
LOCAL_FRACTIONS = [0.40, 0.50, 0.60]

# C.1 definitions retained exactly.
WATER_SUPPORT_MIN_SCORE = 3
NON_WATER_MAX_SCORE = 1


def load_array(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return np.load(path, mmap_mode="r")


def get_status(flags):
    flags = np.asarray(flags)
    if np.any(flags == 2):
        return "FLAG"
    if np.any(flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


def finite_quantile(arrays, q):
    # Each AOI is small enough to concatenate; use float32 to limit memory.
    parts = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        v = a[np.isfinite(a)]
        if v.size:
            parts.append(v)
    if not parts:
        return float("nan")
    pooled = np.concatenate(parts)
    return float(np.quantile(pooled, q))


def percent(n, d):
    return 100.0 * n / d if d else float("nan")


def save_map(array, path, title, cmap="viridis", vmin=0, vmax=1):
    plt.figure(figsize=(9, 7))
    im = plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("AOI column")
    plt.ylabel("AOI row")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    for folder in (AOI_DIR, STAGE_A_DIR, STAGE_B_DIR, C1_DIR):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the saved C.1 features and Stage B candidate masks.
    data = {}
    for date in DATES:
        c1 = C1_DIR / date
        b = STAGE_B_DIR / date
        flags = load_array(STAGE_A_DIR / f"{date}_quality_flags.npy")
        status = get_status(flags)

        hh_db = np.asarray(load_array(c1 / "HH_dB.npy"), dtype=np.float32)
        hv_db = np.asarray(load_array(c1 / "HV_dB.npy"), dtype=np.float32)
        ratio_db = np.asarray(load_array(c1 / "HV_HH_ratio_dB.npy"), dtype=np.float32)
        local_frac = np.asarray(load_array(c1 / "local_candidate_fraction.npy"),
                                dtype=np.float32)
        c1_class = np.asarray(load_array(c1 / "preliminary_class.npy"))
        candidate = np.asarray(load_array(b / "water_candidate_mask.npy")).astype(bool)

        shapes = {x.shape for x in (hh_db, hv_db, ratio_db, local_frac,
                                    c1_class, candidate, flags)}
        if len(shapes) != 1:
            raise ValueError(f"{date}: arrays have inconsistent shapes: {shapes}")

        # Stage C.1 class 0 marks invalid pixels. Retain the original C.1
        # validity decision rather than inventing a new quality mask.
        valid = c1_class != 0
        candidate &= valid

        data[date] = {
            "hh": hh_db, "hv": hv_db, "ratio": ratio_db,
            "local": local_frac, "candidate": candidate,
            "valid": valid, "status": status
        }

    # Derive pooled-reference thresholds independently at each quantile.
    thresholds = {}
    for q in QUANTILES:
        thresholds[str(q)] = {
            "HH_low_dB": finite_quantile(
                [data[d]["hh"][data[d]["valid"]] for d in REFERENCE_DATES], q),
            "HV_low_dB": finite_quantile(
                [data[d]["hv"][data[d]["valid"] & np.isfinite(data[d]["hv"])]
                 for d in REFERENCE_DATES], q),
            "HV_HH_ratio_low_dB": finite_quantile(
                [data[d]["ratio"][data[d]["valid"] & np.isfinite(data[d]["ratio"])]
                 for d in REFERENCE_DATES], q)
        }

    with open(OUTPUT_DIR / "reference_thresholds_by_quantile.json",
              "w", encoding="utf-8") as f:
        json.dump({
            "reference_dates": REFERENCE_DATES,
            "quantiles": thresholds,
            "note": "Thresholds are pooled reference-date quantiles, not physical water boundaries."
        }, f, indent=2)

    # Count the number of tested parameter combinations: 3 quantiles x 3
    # spatial support levels = 9. Every combination is applied to all dates.
    run_rows = []
    date_rows = []

    water_freq = {d: None for d in DATES}
    candidate_freq = {d: None for d in DATES}
    persistent_freq = None

    for q, spatial_min in itertools.product(QUANTILES, LOCAL_FRACTIONS):
        t = thresholds[str(q)]
        water_masks = {}
        candidate_masks = {}
        classes_by_date = {}

        for date in DATES:
            x = data[date]
            valid = x["valid"]
            candidate = x["candidate"]

            hh_low = valid & np.isfinite(x["hh"]) & (x["hh"] <= t["HH_low_dB"])
            spatial = valid & np.isfinite(x["local"]) & (x["local"] >= spatial_min)

            hv_low = valid & np.isfinite(x["hv"]) & (x["hv"] <= t["HV_low_dB"])
            ratio_low = valid & np.isfinite(x["ratio"]) & (
                x["ratio"] <= t["HV_HH_ratio_low_dB"]
            )

            score = np.zeros(valid.shape, dtype=np.uint8)
            score[candidate & hh_low] += 1
            score[candidate & spatial] += 1

            if x["status"] == "ACCEPT":
                score[candidate & hv_low] += 1
                score[candidate & ratio_low] += 1
                water = candidate & (score >= WATER_SUPPORT_MIN_SCORE)
                nonwater = candidate & (score <= NON_WATER_MAX_SCORE)
            else:
                # September: preserve HH + spatial candidates; do not use
                # anomalous HV/ratio evidence to score or reject pixels.
                water = candidate & (score == 2)
                nonwater = np.zeros(valid.shape, dtype=bool)

            classes = np.zeros(valid.shape, dtype=np.uint8)
            classes[valid] = 3  # uncertain by default
            classes[nonwater] = 1
            classes[water] = 2

            water_masks[date] = water
            candidate_masks[date] = candidate & hh_low & spatial
            classes_by_date[date] = classes

            n_valid = int(valid.sum())
            n_candidate = int(candidate.sum())
            n_water = int(water.sum())
            n_nonwater = int(nonwater.sum())
            n_uncertain = int(np.sum(classes == 3))

            date_rows.append({
                "quantile": q,
                "local_support_min": spatial_min,
                "date": date,
                "stage_A_status": x["status"],
                "hv_ratio_used": x["status"] == "ACCEPT",
                "valid_pixels": n_valid,
                "stage_B_candidate_pixels": n_candidate,
                "hh_and_spatial_supported_pixels": int(
                    np.sum(candidate & hh_low & spatial)
                ),
                "water_supported_pixels": n_water,
                "water_supported_pct_of_valid": percent(n_water, n_valid),
                "nonwater_supported_pixels": n_nonwater,
                "nonwater_supported_pct_of_valid": percent(n_nonwater, n_valid),
                "uncertain_pixels": n_uncertain,
                "uncertain_pct_of_valid": percent(n_uncertain, n_valid)
            })

            if water_freq[date] is None:
                water_freq[date] = water.astype(np.uint8)
                candidate_freq[date] = candidate_masks[date].astype(np.uint8)
            else:
                water_freq[date] += water.astype(np.uint8)
                candidate_freq[date] += candidate_masks[date].astype(np.uint8)

        persistent = water_masks["20260819"] & water_masks["20261018"]
        if persistent_freq is None:
            persistent_freq = persistent.astype(np.uint8)
        else:
            persistent_freq += persistent.astype(np.uint8)

        run_rows.append({
            "quantile": q,
            "local_support_min": spatial_min,
            "aug_water_pixels": int(water_masks["20260819"].sum()),
            "sep_HH_spatial_water_pixels": int(water_masks["20260924"].sum()),
            "oct_water_pixels": int(water_masks["20261018"].sum()),
            "aug_oct_persistent_water_pixels": int(persistent.sum()),
            "aug_oct_persistent_water_pct": percent(
                int(persistent.sum()), int(data["20260819"]["valid"].sum())
            )
        })

    runs_df = pd.DataFrame(run_rows)
    dates_df = pd.DataFrame(date_rows)
    runs_df.to_csv(OUTPUT_DIR / "sensitivity_runs.csv", index=False)
    dates_df.to_csv(OUTPUT_DIR / "date_sensitivity_summary.csv", index=False)

    n_runs = len(QUANTILES) * len(LOCAL_FRACTIONS)
    stability_rows = []
    for date in DATES:
        freq = water_freq[date].astype(np.float32) / n_runs
        stable_80 = freq >= 0.80
        stable_100 = freq >= 1.0
        valid = data[date]["valid"]
        stability_rows.append({
            "date": date,
            "stage_A_status": data[date]["status"],
            "runs_tested": n_runs,
            "pixels_water_supported_in_at_least_80pct_runs": int(stable_80.sum()),
            "pct_valid_water_stable_80pct": percent(int(stable_80.sum()), int(valid.sum())),
            "pixels_water_supported_in_all_runs": int(stable_100.sum()),
            "pct_valid_water_stable_100pct": percent(int(stable_100.sum()), int(valid.sum())),
            "interpretation": (
                "September HH/spatial-only stability; not polarimetrically validated."
                if date == "20260924" else
                "Sensitivity stability of exploratory polarimetric evidence classes."
            )
        })

        np.save(OUTPUT_DIR / f"{date}_water_support_frequency.npy", freq)
        save_map(
            freq,
            OUTPUT_DIR / f"{date}_water_support_frequency.png",
            f"{date}: water-support frequency across {n_runs} sensitivity runs",
            vmin=0, vmax=1
        )

        if date == "20260924":
            cfreq = candidate_freq[date].astype(np.float32) / n_runs
            np.save(OUTPUT_DIR / "september_candidate_frequency.npy", cfreq)
            save_map(
                cfreq, OUTPUT_DIR / "september_candidate_frequency.png",
                "September: HH + spatial candidate frequency (HV excluded)",
                vmin=0, vmax=1
            )

    persistent_frequency = persistent_freq.astype(np.float32) / n_runs
    np.save(OUTPUT_DIR / "aug_oct_persistent_water_frequency.npy",
            persistent_frequency)
    save_map(
        persistent_frequency,
        OUTPUT_DIR / "aug_oct_persistent_water_frequency.png",
        "August-October persistent water-support frequency across sensitivity runs",
        vmin=0, vmax=1
    )

    pd.DataFrame(stability_rows).to_csv(
        OUTPUT_DIR / "pixel_class_stability.csv", index=False
    )

    # Quantify how stable the temporal consensus itself is.
    persistent_80 = persistent_frequency >= 0.80
    persistent_100 = persistent_frequency >= 1.0
    ref_valid = data["20260819"]["valid"] & data["20261018"]["valid"]
    report = {
        "stage": "C.3 - Threshold sensitivity and robustness analysis",
        "dates_included": DATES,
        "reference_dates": REFERENCE_DATES,
        "september_policy": (
            "September remains included in all date-level outputs. Its HH and "
            "spatial evidence is tested; its anomalous HV and HV/HH evidence "
            "is retained in the source arrays but excluded from scoring."
        ),
        "parameter_grid": {
            "pooled_reference_quantiles": QUANTILES,
            "local_candidate_fraction_thresholds": LOCAL_FRACTIONS,
            "number_of_runs": n_runs
        },
        "persistent_reference_water": {
            "definition": "Water-supported in both August 19 and October 18 within each parameter run.",
            "pixels_supported_in_at_least_80pct_runs": int(persistent_80.sum()),
            "pct_of_joint_valid_reference_pixels": percent(
                int(persistent_80.sum()), int(ref_valid.sum())
            ),
            "pixels_supported_in_all_runs": int(persistent_100.sum()),
            "pct_of_joint_valid_reference_pixels_all_runs": percent(
                int(persistent_100.sum()), int(ref_valid.sum())
            )
        },
        "limitations": [
            "This evaluates parameter sensitivity, not classification accuracy.",
            "C.1 evidence classes remain exploratory, not validated water labels.",
            "August and October are only two reference acquisitions.",
            "September water-support frequency is based on HH and spatial evidence only.",
            "Independent reference data are required for accuracy assessment."
        ],
        "outputs": [
            "sensitivity_runs.csv",
            "date_sensitivity_summary.csv",
            "pixel_class_stability.csv",
            "reference_thresholds_by_quantile.json",
            "20260819_water_support_frequency.npy/png",
            "20260924_water_support_frequency.npy/png",
            "20261018_water_support_frequency.npy/png",
            "september_candidate_frequency.npy/png",
            "aug_oct_persistent_water_frequency.npy/png"
        ]
    }
    with open(OUTPUT_DIR / "stage_C3_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTAGE C.3 COMPLETE")
    print(f"Dates processed: {', '.join(DATES)}")
    print(f"Sensitivity runs: {n_runs}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("\nPersistent Aug-Oct water support:")
    print(f"  Stable in >=80% runs: {int(persistent_80.sum()):,} pixels")
    print(f"  Stable in all runs:   {int(persistent_100.sum()):,} pixels")
    print("\nSeptember remains included as HH/spatial diagnostic evidence.")
    print("These are robustness results, NOT validated water maps.")


if __name__ == "__main__":
    main()

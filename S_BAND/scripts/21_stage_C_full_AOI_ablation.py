"""
21_stage_C_full_AOI_ablation.py

QGPT-WM S-band: corrected Stage C feature ablation.

Main correction:
- Evaluate A/B/C across the FULL Stage-A-valid AOI, not only the
  Stage-B HH-derived candidate mask.
- Keep Stage-B candidate masks as a separate diagnostic overlay/statistic.
- Include all three dates, including flagged September.
- Use fixed thresholds already produced by Scripts 13 and 16.
- Reuse Script 19's saved continuous anomaly-aware fusion score for D.
  The score must cover the valid AOI; the script reports its coverage and
  stops if it does not, rather than silently treating missing pixels as
  non-water.

Experiments:
 A: HH low-backscatter evidence
 B: HH + local spatial candidate support
 C: HH + HH texture + local spatial candidate support
 D: Script 19 anomaly-aware continuous fusion score (equal, spatial and
    texture weighting configurations)

Classification:
 score >= 2/3       -> water-supported
 score <= 1/3       -> non-water-supported
 otherwise          -> uncertain

For A, binary evidence means only water/non-water can occur.
For B, a score of 0.5 is uncertain.
For C, scores 0, 1/3, 2/3 and 1 are possible.

Important scientific note:
- The spatial support layer is calculated from Stage-B candidate masks.
  It is contextual evidence, not an independent sensor measurement.
- These are feature-ablation diagnostics, not accuracy estimates.
  Independent reference labels are still required for accuracy assessment.

Run from:
 C:\\Users\\HP\\Downloads\\NISAR\\S_BAND

    python 21_stage_C_full_AOI_ablation.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

BASE = Path(__file__).resolve().parent
AOI = BASE / "aoi_outputs"
QUALITY = BASE / "acquisition_quality_output"
CANDIDATES = BASE / "stage_B_candidate_output"
C1 = BASE / "stage_C1_validation_output"
TEXTURE = BASE / "stage_C_texture_output"
S19 = BASE / "stage_C_anomaly_aware_output"
OUT = BASE / "stage_C_full_AOI_ablation_output"

DATES = ["20260819", "20260924", "20261018"]
CONFIGS = {
    "A_HH_only": ["HH"],
    "B_HH_spatial": ["HH", "SPATIAL"],
    "C_HH_texture_spatial": ["HH", "HH_TEXTURE", "SPATIAL"],
}
WATER_THRESHOLD = 2.0 / 3.0
NONWATER_THRESHOLD = 1.0 / 3.0
SPATIAL_THRESHOLD = 0.50
MIN_D_COVERAGE = 0.95

CLASS_COLORS = ["black", "saddlebrown", "deepskyblue", "gold"]
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CLASS_CMAP.N)


def load_npy(folder, stem):
    path = folder / f"{stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Required array not found: {path}")
    return np.asarray(np.load(path, mmap_mode="r"))


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def quality_status(flags):
    vals = np.unique(flags)
    if 2 in vals:
        return "FLAG"
    if 1 in vals:
        return "ACCEPT"
    return "UNKNOWN"


def classify_score(score, valid):
    """
    Class values:
      0 invalid / unavailable
      1 non-water-supported
      2 water-supported
      3 uncertain
    """
    classes = np.zeros(score.shape, dtype=np.uint8)
    finite = valid & np.isfinite(score)
    classes[finite] = 3
    classes[finite & (score <= NONWATER_THRESHOLD)] = 1
    classes[finite & (score >= WATER_THRESHOLD)] = 2
    return classes, finite


def save_class_map(classes, title, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(classes, cmap=CLASS_CMAP, norm=CLASS_NORM)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["Invalid", "Non-water", "Water", "Uncertain"])
    ax.set_title(title)
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_score_map(score, valid, title, path):
    display = np.where(valid & np.isfinite(score), score, np.nan)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(display, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Water-support score")
    ax.set_title(title)
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def find_fusion_score(date, config):
    folder = S19 / date / config
    possible = [
        "anomaly_aware_fusion_score.npy",
        "fusion_score.npy",
        "anomaly_aware_score.npy",
        "water_support_score.npy",
    ]
    for name in possible:
        p = folder / name
        if p.is_file():
            return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32), p
    found = sorted(folder.glob("*score*.npy")) if folder.exists() else []
    if len(found) == 1:
        return np.asarray(np.load(found[0], mmap_mode="r"), dtype=np.float32), found[0]
    raise FileNotFoundError(
        f"No unambiguous continuous fusion score found for {date}/{config}. "
        f"Looked in {folder}. Expected one of: {possible}. "
        "Do not substitute Script 19's candidate-only class map; a full-AOI "
        "ablation requires the continuous score over valid pixels."
    )


def count_row(date, status, experiment, classes, score, valid, candidate):
    evaluated = valid & np.isfinite(score)
    denom = int(valid.sum())
    eval_count = int(evaluated.sum())
    cand_count = int(candidate.sum())
    water = evaluated & (classes == 2)
    nonwater = evaluated & (classes == 1)
    uncertain = evaluated & (classes == 3)

    candidate_eval = candidate & evaluated
    candidate_denom = int(candidate_eval.sum())

    return {
        "date": date,
        "stage_A_status": status,
        "experiment": experiment,
        "valid_AOI_pixels": denom,
        "evaluated_valid_pixels": eval_count,
        "score_coverage_pct_of_valid_AOI": (
            100.0 * eval_count / denom if denom else np.nan
        ),
        "water_pixels": int(water.sum()),
        "nonwater_pixels": int(nonwater.sum()),
        "uncertain_pixels": int(uncertain.sum()),
        "water_pct_of_valid_AOI": (
            100.0 * water.sum() / denom if denom else np.nan
        ),
        "nonwater_pct_of_valid_AOI": (
            100.0 * nonwater.sum() / denom if denom else np.nan
        ),
        "uncertain_pct_of_valid_AOI": (
            100.0 * uncertain.sum() / denom if denom else np.nan
        ),
        "stage_B_candidate_pixels": cand_count,
        "stage_B_candidates_evaluated": candidate_denom,
        "candidate_water_supported_pixels": int(
            np.sum(candidate_eval & (classes == 2))
        ),
        "candidate_water_support_pct": (
            100.0 * np.sum(candidate_eval & (classes == 2)) / candidate_denom
            if candidate_denom else np.nan
        ),
    }


def run_date(date, c1_report, texture_report):
    aoi_dir = AOI / date
    candidate_dir = CANDIDATES / date
    c1_dir = C1 / date
    texture_dir = TEXTURE / date

    flags = load_npy(QUALITY, f"{date}_quality_flags")
    hh = load_npy(aoi_dir, "HHHH").astype(np.float32)
    mask = load_npy(aoi_dir, "mask")
    candidate = load_npy(candidate_dir, "water_candidate_mask").astype(bool)
    hh_db = load_npy(c1_dir, "HH_dB").astype(np.float32)
    hh_texture = load_npy(texture_dir, "HH_texture_CV_5x5").astype(np.float32)
    local_candidate_fraction = load_npy(
        texture_dir, "local_candidate_fraction"
    ).astype(np.float32)

    shape = hh.shape
    arrays = {
        "flags": flags,
        "mask": mask,
        "candidate": candidate,
        "hh_db": hh_db,
        "hh_texture": hh_texture,
        "local_candidate_fraction": local_candidate_fraction,
    }
    for name, arr in arrays.items():
        if arr.shape != shape:
            raise ValueError(
                f"{date}: {name} shape {arr.shape} does not match HH {shape}"
            )

    # Match Stage-A valid-data logic while retaining all quality-flagged
    # acquisitions. FLAG is a diagnostic status, not an exclusion switch.
    valid = (
        (mask != 0) & (mask != 255)
        & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
    )
    candidate &= valid
    status = quality_status(flags)

    hh_threshold = c1_report["common_thresholds"]["HH_low_threshold_dB"]
    texture_threshold = texture_report["thresholds"][
        "HH_texture_low_CV_threshold"
    ]

    hh_feature = np.full(shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(hh_db)
    hh_feature[ok] = (hh_db[ok] <= hh_threshold).astype(np.float32)

    spatial_feature = np.full(shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(local_candidate_fraction)
    spatial_feature[ok] = (
        local_candidate_fraction[ok] >= SPATIAL_THRESHOLD
    ).astype(np.float32)

    texture_feature = np.full(shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(hh_texture)
    texture_feature[ok] = (
        hh_texture[ok] <= texture_threshold
    ).astype(np.float32)

    features = {
        "HH": hh_feature,
        "SPATIAL": spatial_feature,
        "HH_TEXTURE": texture_feature,
    }

    out_date = OUT / date
    out_date.mkdir(parents=True, exist_ok=True)
    rows = []
    class_arrays = {}
    score_arrays = {}

    # A-C: full valid AOI, not restricted to Stage-B candidates.
    for experiment, feature_names in CONFIGS.items():
        stack = np.stack([features[name] for name in feature_names], axis=0)
        available = np.isfinite(stack)
        positive = (stack >= 0.5) & available
        n_available = available.sum(axis=0)
        n_positive = positive.sum(axis=0)

        score = np.full(shape, np.nan, dtype=np.float32)
        complete = valid & (n_available == len(feature_names))
        score[complete] = (
            n_positive[complete] / float(len(feature_names))
        ).astype(np.float32)

        classes, evaluated = classify_score(score, valid)
        class_arrays[experiment] = classes
        score_arrays[experiment] = score

        exp_dir = out_date / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.save(exp_dir / "support_score_full_AOI.npy", score)
        np.save(exp_dir / "classification_full_AOI.npy", classes)
        np.save(exp_dir / "feature_availability_count.npy", n_available)

        save_class_map(
            classes, f"{date} | {experiment} | Stage A {status}",
            exp_dir / "classification_full_AOI.png"
        )
        save_score_map(
            score, valid, f"{date} | {experiment} | support score",
            exp_dir / "support_score_full_AOI.png"
        )

        rows.append(
            count_row(date, status, experiment, classes, score, valid, candidate)
        )

    # D: import continuous Script-19 fusion scores and reclassify over full AOI.
    for config in ("equal", "spatial_emphasis", "texture_emphasis"):
        fusion_score, source_path = find_fusion_score(date, config)
        if fusion_score.shape != shape:
            raise ValueError(
                f"{date}/{config}: fusion score shape {fusion_score.shape} "
                f"does not match AOI {shape}"
            )
        classes, evaluated = classify_score(fusion_score, valid)
        coverage = evaluated.sum() / max(1, valid.sum())
        if coverage < MIN_D_COVERAGE:
            raise RuntimeError(
                f"{date}/{config}: Script 19 score covers only "
                f"{coverage:.1%} of valid AOI. Full-AOI D comparison is not "
                f"possible from this array ({source_path}). Do not use the "
                "candidate-only class map as a substitute."
            )

        experiment = f"D_full_{config}"
        class_arrays[experiment] = classes
        score_arrays[experiment] = fusion_score

        exp_dir = out_date / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.save(exp_dir / "support_score_full_AOI.npy", fusion_score)
        np.save(exp_dir / "classification_full_AOI.npy", classes)
        save_class_map(
            classes, f"{date} | {experiment} | Stage A {status}",
            exp_dir / "classification_full_AOI.png"
        )
        save_score_map(
            fusion_score, valid, f"{date} | {experiment} | support score",
            exp_dir / "support_score_full_AOI.png"
        )
        row = count_row(
            date, status, experiment, classes, fusion_score, valid, candidate
        )
        row["script19_score_source"] = str(source_path)
        rows.append(row)

    # Compare class overlap only on the full valid AOI.
    overlap_rows = []
    keys = list(class_arrays)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            left_water = valid & (class_arrays[left] == 2)
            right_water = valid & (class_arrays[right] == 2)
            union = int(np.sum(left_water | right_water))
            intersection = int(np.sum(left_water & right_water))
            overlap_rows.append({
                "date": date,
                "experiment_1": left,
                "experiment_2": right,
                "water_intersection_pixels": intersection,
                "water_union_pixels": union,
                "water_jaccard_full_valid_AOI": (
                    intersection / union if union else np.nan
                ),
            })
    pd.DataFrame(overlap_rows).to_csv(
        out_date / "pairwise_water_overlap_full_AOI.csv", index=False
    )

    # Six-panel map: A/B/C and three D weighting configurations.
    panel_keys = [
        "A_HH_only", "B_HH_spatial", "C_HH_texture_spatial",
        "D_full_equal", "D_full_spatial_emphasis", "D_full_texture_emphasis",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, key in zip(axes.flat, panel_keys):
        ax.imshow(class_arrays[key], cmap=CLASS_CMAP, norm=CLASS_NORM)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{date}: FULL valid-AOI feature ablation | Stage A {status}"
    )
    fig.tight_layout()
    fig.savefig(out_date / "full_AOI_ablation_comparison.png", dpi=180)
    plt.close(fig)

    # Candidate boundary diagnostic: show where Stage B candidates lie.
    fig, ax = plt.subplots(figsize=(9, 7))
    overlay = np.zeros(shape, dtype=np.uint8)
    overlay[valid] = 1
    overlay[candidate] = 2
    cmap = ListedColormap(["black", "lightgray", "deepskyblue"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax.imshow(overlay, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["Invalid", "Valid AOI", "Stage B candidate"])
    ax.set_title(f"{date}: valid AOI and Stage B candidate footprint")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_date / "valid_AOI_candidate_footprint.png", dpi=170)
    plt.close(fig)

    return rows


def main():
    for folder in (AOI, QUALITY, CANDIDATES, C1, TEXTURE, S19):
        if not folder.is_dir():
            raise FileNotFoundError(f"Required folder missing: {folder}")

    c1_report = load_json(C1 / "stage_C1_report.json")
    texture_report = load_json(
        TEXTURE / "texture_reference_thresholds.json"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for date in DATES:
        print(f"\nProcessing {date}")
        rows = run_date(date, c1_report, texture_report)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  {row['experiment']}: "
                f"water={row['water_pixels']:,} "
                f"({row['water_pct_of_valid_AOI']:.2f}% valid AOI), "
                f"non-water={row['nonwater_pixels']:,}, "
                f"uncertain={row['uncertain_pixels']:,}, "
                f"coverage={row['score_coverage_pct_of_valid_AOI']:.2f}%"
            )

    stats = pd.DataFrame(all_rows)
    stats.to_csv(OUT / "stage_C_full_AOI_ablation_statistics.csv", index=False)

    report = {
        "stage": "QGPT-WM S-band Stage C full-valid-AOI ablation",
        "dates": DATES,
        "evaluation_domain": (
            "All Stage-A-valid AOI pixels. Stage-B candidate masks are "
            "reported separately and do not restrict A-C evaluation."
        ),
        "experiments": {
            "A_HH_only": ["HH low-backscatter evidence"],
            "B_HH_spatial": [
                "HH low-backscatter evidence",
                "local support derived from Stage-B candidate masks"
            ],
            "C_HH_texture_spatial": [
                "HH low-backscatter evidence",
                "HH local texture evidence",
                "local support derived from Stage-B candidate masks"
            ],
            "D_full": [
                "Script 19 continuous anomaly-aware fusion score",
                "equal, spatial-emphasis and texture-emphasis configurations"
            ],
        },
        "thresholds": {
            "water_support": WATER_THRESHOLD,
            "nonwater_support": NONWATER_THRESHOLD,
            "uncertain": "between non-water and water thresholds",
            "spatial_support": SPATIAL_THRESHOLD,
            "HH": "existing common Stage C1 threshold",
            "HH_texture": "existing Stage C texture reference threshold",
        },
        "september_handling": (
            "September remains included. Its Stage-A FLAG is retained as a "
            "diagnostic status. Script 19 anomaly-aware continuous fusion "
            "scores are used for D; HV-derived measurements are not discarded."
        ),
        "limitations": [
            "A-C are feature-ablation diagnostics, not calibrated water probabilities.",
            "Spatial support is derived from Stage-B candidate masks and is "
            "contextual evidence, not an independent sensor measurement.",
            "D depends on Script 19's exploratory anomaly-reliability heuristic.",
            "No independent reference labels are used; no accuracy estimate is produced.",
            "D requires a continuous Script 19 fusion score covering at least "
            f"{MIN_D_COVERAGE:.0%} of valid AOI pixels."
        ],
    }
    with (OUT / "stage_C_full_AOI_ablation_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\nFULL-AOI STAGE C ABLATION COMPLETE")
    print(f"Output folder: {OUT}")
    print("Main table: stage_C_full_AOI_ablation_statistics.csv")
    print("Review each date's full_AOI_ablation_comparison.png")


if __name__ == "__main__":
    main()

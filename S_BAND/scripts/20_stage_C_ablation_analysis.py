"""
20_stage_C_ablation_analysis.py

QGPT-WM S-band - Stage C feature ablation analysis.

Compares four fixed-threshold evidence configurations on the SAME Stage B
candidate mask for each date:
 A: HH backscatter only
 B: HH + spatial support
 C: HH + HH texture + spatial support
 D: Full anomaly-aware evidence fusion (Script 19 family outputs)

September is included in every experiment. Its HV-derived evidence is
not deleted; experiment D uses Script 19's reliability-weighted family
fusion outputs. Experiments A-C deliberately isolate non-HV evidence
contributions so the source of September's high water-support rate can
be diagnosed.

Important:
- Uses existing thresholds, not new threshold fitting.
- A-C use transparent feature fractions and the same water/non-water
  cutoffs across dates. Their outputs are diagnostic ablations, not
  replacements for Script 19.
- D imports the already-computed Script 19 classes for each weighting
  configuration.
- No independent accuracy claim is made.

Run from the S_BAND folder:
    python 20_stage_C_ablation_analysis.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

BASE = Path(__file__).resolve().parent
AOI = BASE / "aoi_outputs"
A = BASE / "acquisition_quality_output"
B = BASE / "stage_B_candidate_output"
C1 = BASE / "stage_C1_validation_output"
TEX = BASE / "stage_C_texture_output"
S19 = BASE / "stage_C_anomaly_aware_output"
OUT = BASE / "stage_C_ablation_output"

DATES = ["20260819", "20260924", "20261018"]

# Fixed screening cutoffs, held constant across dates and experiments.
WATER_FRACTION = 2.0 / 3.0
NONWATER_FRACTION = 0.25
SPATIAL_THRESHOLD = 0.50

CONFIGS = {
    "A_HH_only": ["HH"],
    "B_HH_spatial": ["HH", "SPATIAL"],
    "C_HH_texture_spatial": ["HH", "HH_TEXTURE", "SPATIAL"],
}


def load(folder, name):
    p = folder / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing input: {p}")
    return np.asarray(np.load(p, mmap_mode="r"))


def status_of(flags):
    if np.any(flags == 2):
        return "FLAG"
    if np.any(flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


def make_classes(candidate, valid, features):
    """Per-pixel positive fraction for the explicitly supplied features."""
    n = np.zeros(candidate.shape, dtype=np.uint8)
    k = np.zeros(candidate.shape, dtype=np.uint8)

    for feature in features:
        available = np.isfinite(feature)
        positive = available & (feature >= 0.5)
        n[available] += 1
        k[positive] += 1

    fraction = np.full(candidate.shape, np.nan, dtype=np.float32)
    has = n > 0
    fraction[has] = k[has] / n[has]

    classes = np.zeros(candidate.shape, dtype=np.uint8)
    classes[valid] = 3
    eligible = candidate & (n == len(features))
    water = eligible & (fraction >= WATER_FRACTION)
    nonwater = eligible & (fraction <= NONWATER_FRACTION)
    classes[water] = 2
    classes[nonwater] = 1
    classes[candidate & ~eligible] = 3
    return classes, fraction, n, k, eligible


def save_map(arr, title, path, categorical=False, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    if categorical:
        cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        im = ax.imshow(arr, cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cb.ax.set_yticklabels(["Invalid", "Non-water", "Water", "Uncertain"])
    else:
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_date(date):
    aoi = AOI / date
    c1 = C1 / date
    tex = TEX / date
    bdir = B / date
    flags = load(A, f"{date}_quality_flags")
    hh = load(aoi, "HHHH").astype(np.float32)
    mask = load(aoi, "mask")
    candidate = load(bdir, "water_candidate_mask").astype(bool)
    hh_db = load(c1, "HH_dB").astype(np.float32)
    hh_tex = load(tex, "HH_texture_CV_5x5").astype(np.float32)
    local = load(tex, "local_candidate_fraction").astype(np.float32)

    valid = (
        (mask != 0) & (mask != 255) & (flags != 0)
        & np.isfinite(hh) & (hh > 0)
    )
    candidate &= valid
    status = status_of(flags)

    with open( C1 / "stage_C1_report.json", "r", encoding="utf-8") as f:
        c1_report = json.load(f)
    with open(TEX / "texture_reference_thresholds.json",
              "r", encoding="utf-8") as f:
        tex_report = json.load(f)

    hh_thr = c1_report["common_thresholds"]["HH_low_threshold_dB"]
    tex_thr = tex_report["thresholds"]["HH_texture_low_CV_threshold"]

    hh_feature = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(hh_db)
    hh_feature[ok] = (hh_db[ok] <= hh_thr).astype(np.float32)

    spatial_feature = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(local)
    spatial_feature[ok] = (local[ok] >= SPATIAL_THRESHOLD).astype(np.float32)

    texture_feature = np.full(hh.shape, np.nan, dtype=np.float32)
    ok = valid & np.isfinite(hh_tex)
    texture_feature[ok] = (hh_tex[ok] <= tex_thr).astype(np.float32)

    features = {
        "HH": hh_feature,
        "SPATIAL": spatial_feature,
        "HH_TEXTURE": texture_feature,
    }

    out_date = OUT / date
    out_date.mkdir(parents=True, exist_ok=True)
    rows = []
    classes_by_config = {}

    for config, names in CONFIGS.items():
        selected = [features[n] for n in names]
        classes, fraction, available, positive, eligible = make_classes(
            candidate, valid, selected
        )
        classes_by_config[config] = classes

        cdir = out_date / config
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(cdir / "class.npy", classes)
        np.save(cdir / "support_fraction.npy", fraction)
        np.save(cdir / "available_feature_count.npy", available)
        np.save(cdir / "positive_feature_count.npy", positive)
        save_map(
            classes, f"{date} | {config} | Stage A {status}",
            cdir / "classification_map.png", categorical=True
        )
        save_map(
            fraction, f"{date} | {config} | positive evidence fraction",
            cdir / "support_fraction.png", vmin=0, vmax=1
        )

        rows.append({
            "date": date,
            "stage_A_status": status,
            "experiment": config,
            "candidate_pixels": int(candidate.sum()),
            "eligible_candidate_pixels": int(eligible.sum()),
            "water_supported_pixels": int((classes == 2).sum()),
            "nonwater_supported_pixels": int((classes == 1).sum()),
            "uncertain_candidate_pixels": int(
                (candidate & (classes == 3)).sum()
            ),
            "water_pct_of_candidates": (
                100.0 * (classes[candidate] == 2).sum() / candidate.sum()
                if candidate.sum() else np.nan
            ),
            "nonwater_pct_of_candidates": (
                100.0 * (classes[candidate] == 1).sum() / candidate.sum()
                if candidate.sum() else np.nan
            ),
        })

    # Import Script 19 full-fusion classes for direct comparison.
    for config in ("equal", "spatial_emphasis", "texture_emphasis"):
        p = S19 / date / config / "anomaly_aware_class.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Run Script 19 first; missing full-fusion output: {p}"
            )
        classes = np.asarray(np.load(p, mmap_mode="r"))
        if classes.shape != candidate.shape:
            raise ValueError(f"{date}/{config}: shape mismatch with candidate.")
        classes_by_config[f"D_full_{config}"] = classes

        rows.append({
            "date": date,
            "stage_A_status": status,
            "experiment": f"D_full_{config}",
            "candidate_pixels": int(candidate.sum()),
            "eligible_candidate_pixels": int(
                np.sum(candidate & (classes != 3) & (classes != 0))
            ),
            "water_supported_pixels": int((classes == 2).sum()),
            "nonwater_supported_pixels": int((classes == 1).sum()),
            "uncertain_candidate_pixels": int(
                (candidate & (classes == 3)).sum()
            ),
            "water_pct_of_candidates": (
                100.0 * (classes[candidate] == 2).sum() / candidate.sum()
                if candidate.sum() else np.nan
            ),
            "nonwater_pct_of_candidates": (
                100.0 * (classes[candidate] == 1).sum() / candidate.sum()
                if candidate.sum() else np.nan
            ),
        })

    # Pairwise overlap among experiments, restricted to common candidate mask.
    overlap_rows = []
    keys = list(classes_by_config)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            l = classes_by_config[left]
            r = classes_by_config[right]
            lm = candidate & (l == 2)
            rm = candidate & (r == 2)
            union = np.sum(lm | rm)
            intersection = np.sum(lm & rm)
            overlap_rows.append({
                "date": date,
                "experiment_1": left,
                "experiment_2": right,
                "water_intersection_pixels": int(intersection),
                "water_union_pixels": int(union),
                "water_jaccard": (
                    float(intersection / union) if union else np.nan
                ),
            })

    pd.DataFrame(overlap_rows).to_csv(
        out_date / "pairwise_water_overlap.csv", index=False
    )

    # Compact visual comparison panel.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    show_keys = [
        "A_HH_only", "B_HH_spatial", "C_HH_texture_spatial",
        "D_full_equal", "D_full_spatial_emphasis", "D_full_texture_emphasis"
    ]
    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    for ax, key in zip(axes.flat, show_keys):
        ax.imshow(classes_by_config[key], cmap=cmap, norm=norm)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{date}: Stage C feature ablation | {status}")
    fig.tight_layout()
    fig.savefig(out_date / "ablation_comparison_panel.png", dpi=180)
    plt.close(fig)

    return rows


def main():
    for folder in (AOI, A, B, C1, TEX, S19):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder missing: {folder}")

    all_rows = []
    for date in DATES:
        print(f"\nProcessing {date}")
        rows = run_date(date)
        all_rows.extend(rows)
        for row in rows:
            print(
                f"  {row['experiment']}: "
                f"water={row['water_supported_pixels']:,} "
                f"({row['water_pct_of_candidates']:.2f}% candidates), "
                f"nonwater={row['nonwater_supported_pixels']:,}, "
                f"uncertain candidates={row['uncertain_candidate_pixels']:,}"
            )

    pd.DataFrame(all_rows).to_csv(
        OUT / "stage_C_ablation_statistics.csv", index=False
    )

    report = {
        "stage": "QGPT-WM S-band Stage C feature ablation",
        "dates": DATES,
        "candidate_mask": "Same Stage B water_candidate_mask for all experiments within each date.",
        "experiments": {
            "A_HH_only": ["HH low backscatter"],
            "B_HH_spatial": ["HH low backscatter", "local candidate support"],
            "C_HH_texture_spatial": [
                "HH low backscatter", "HH low texture", "local candidate support"
            ],
            "D_full": [
                "Script 19 anomaly-aware family fusion, with equal/spatial/texture configurations"
            ],
        },
        "fixed_thresholds": {
            "HH": "Existing Stage C1 common HH threshold",
            "HH_texture": "Existing Stage C texture reference threshold",
            "spatial_support": SPATIAL_THRESHOLD,
            "water_support_fraction": WATER_FRACTION,
            "nonwater_support_fraction": NONWATER_FRACTION,
        },
        "september_policy": (
            "September is included in all experiments. A-C isolate non-HV "
            "evidence. D imports Script 19 output in which HV-derived evidence "
            "is reliability-weighted, not excluded."
        ),
        "limitations": [
            "A-C are diagnostic ablations, not calibrated final classifiers.",
            "D uses Script 19's exploratory anomaly reliability heuristic.",
            "No independent reference labels are used; these results measure "
            "feature contribution and agreement, not accuracy."
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "stage_C_ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTAGE C ABLATION COMPLETE")
    print(f"Output folder: {OUT}")
    print("Review stage_C_ablation_statistics.csv and each date's comparison panel.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()

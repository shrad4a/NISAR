"""
Stage C.2 - Temporal consistency and change analysis
NISAR S-band AOI: 20260819, 20260924, 20261018

Purpose:
- Use August and October (Stage A ACCEPT) as the two-date reference pair.
- Keep September in the analysis, but never let its flagged HV/HV-HH evidence
  contribute to the reference consensus.
- Treat September's C.1 WATER_SUPPORTED pixels as provisional HH + spatial
  candidates, not equivalent to polarimetrically supported water.
- Produce temporal agreement, change, and September diagnostic maps.

Expected inputs:
  <project folder>/
      aoi_outputs/
          20260819/...
          20260924/...
          20261018/...
      stage_C1_validation_output/
          20260819/preliminary_class.npy
          20260819/evidence_score.npy
          20260924/preliminary_class.npy
          20260924/evidence_score.npy
          20261018/preliminary_class.npy
          20261018/evidence_score.npy

The script searches recursively for the C.1 arrays, so the C.1 output folder
may have a different name. If duplicate matches exist, it stops and asks you
to remove/rename the duplicate rather than silently choosing one.

Class codes from C.1:
  0 INVALID
  1 NON_WATER_SUPPORTED
  2 WATER_SUPPORTED
  3 UNCERTAIN

IMPORTANT:
These are exploratory evidence classes, not ground-truth water labels.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

DATES = ["20260819", "20260924", "20261018"]
REFERENCE_DATES = ["20260819", "20261018"]
SEPTEMBER = "20260924"

CLASS_INVALID = 0
CLASS_NONWATER = 1
CLASS_WATER = 2
CLASS_UNCERTAIN = 3

# C.1's Stage A result, confirmed by the C.1 report.
STAGE_A_STATUS = {
    "20260819": "ACCEPT",
    "20260924": "FLAG",
    "20261018": "ACCEPT",
}

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage_C2_temporal_output"
OUT.mkdir(exist_ok=True)


def locate_unique(date, filename):
    matches = list(ROOT.rglob(filename))
    matches = [p for p in matches if date in p.parts]
    # Ignore this script's own output folder.
    matches = [p for p in matches if OUT not in p.parents]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} for {date}.\n"
            f"Expected it inside a folder whose path includes {date}."
        )
    if len(matches) > 1:
        paths = "\n".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple {filename} files found for {date}. "
            f"Please keep one unambiguous C.1 output:\n{paths}"
        )
    return matches[0]


def load_c1(date):
    class_path = locate_unique(date, "preliminary_class.npy")
    score_path = locate_unique(date, "evidence_score.npy")
    classes = np.load(class_path).astype(np.uint8)
    scores = np.load(score_path).astype(np.float32)

    if classes.shape != scores.shape:
        raise ValueError(f"Shape mismatch for {date}: classes {classes.shape}, scores {scores.shape}")
    if not np.isin(classes, [0, 1, 2, 3]).all():
        raise ValueError(f"Unexpected class code in {class_path}")
    return classes, scores, class_path, score_path


def pct(mask, valid):
    n = int(np.count_nonzero(mask & valid))
    d = int(np.count_nonzero(valid))
    return 100.0 * n / d if d else float("nan")


def save_class_map(arr, path, title, labels):
    cmap = ListedColormap(["black", "#8c4517", "#00b8ed", "#ffd400"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.82)
    cbar.ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_temporal_map(arr, path, title, categories, colors):
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(categories) + 0.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(categories)), shrink=0.82)
    cbar.ax.set_yticklabels(categories)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    data = {}
    input_files = {}
    for date in DATES:
        classes, scores, cp, sp = load_c1(date)
        data[date] = {"class": classes, "score": scores}
        input_files[date] = {"class": str(cp), "score": str(sp)}

    shapes = {v["class"].shape for v in data.values()}
    if len(shapes) != 1:
        raise ValueError(f"All three class maps must have identical dimensions. Found: {shapes}")

    aug = data["20260819"]["class"]
    sep = data["20260924"]["class"]
    octo = data["20261018"]["class"]

    # A pixel is valid in a date unless C.1 marked it INVALID.
    valid_aug = aug != CLASS_INVALID
    valid_sep = sep != CLASS_INVALID
    valid_oct = octo != CLASS_INVALID
    valid_ref = valid_aug & valid_oct

    # Reference consensus uses only the two Stage-A ACCEPT dates.
    aug_water = aug == CLASS_WATER
    oct_water = octo == CLASS_WATER
    aug_nonwater = aug == CLASS_NONWATER
    oct_nonwater = octo == CLASS_NONWATER

    # Codes:
    # 0 invalid/no common coverage
    # 1 persistent water-supported (both accepted dates)
    # 2 water-supported in August only
    # 3 water-supported in October only
    # 4 persistent non-water-supported (both accepted dates)
    # 5 mixed/uncertain evidence
    temporal = np.zeros(aug.shape, dtype=np.uint8)
    temporal[valid_ref] = 5
    temporal[valid_ref & aug_nonwater & oct_nonwater] = 4
    temporal[valid_ref & aug_water & ~oct_water] = 2
    temporal[valid_ref & ~aug_water & oct_water] = 3
    temporal[valid_ref & aug_water & oct_water] = 1

    # September is deliberately not part of the reference consensus.
    # Its C.1 water-supported class is only HH + spatial candidate support
    # because Stage A flagged the HV and HV/HH channels.
    sep_hh_spatial_candidate = (sep == CLASS_WATER) & valid_sep
    sep_nonwater_evidence = (sep == CLASS_NONWATER) & valid_sep
    sep_uncertain = (sep == CLASS_UNCERTAIN) & valid_sep

    # Compare September's provisional candidate with the accepted-date reference.
    reference_persistent_water = temporal == 1
    reference_any_water = np.isin(temporal, [1, 2, 3])
    sep_compare = np.zeros(sep.shape, dtype=np.uint8)
    # 0 invalid
    # 1 Sep HH/spatial candidate agrees with persistent reference water
    # 2 Sep HH/spatial candidate outside persistent reference water
    # 3 accepted-date persistent water not flagged as Sep candidate
    # 4 all other / uncertain
    sep_compare[valid_ref & valid_sep] = 4
    sep_compare[valid_ref & valid_sep & sep_hh_spatial_candidate & reference_persistent_water] = 1
    sep_compare[valid_ref & valid_sep & sep_hh_spatial_candidate & ~reference_persistent_water] = 2
    sep_compare[valid_ref & valid_sep & ~sep_hh_spatial_candidate & reference_persistent_water] = 3
    sep_compare[~(valid_ref & valid_sep)] = 0

    # Per-date area/evidence summary. Percentages are within each date's valid pixels.
    rows = []
    for date in DATES:
        c = data[date]["class"]
        valid = c != CLASS_INVALID
        row = {
            "date": date,
            "stage_A_status": STAGE_A_STATUS[date],
            "valid_pixels": int(valid.sum()),
            "nonwater_supported_pixels": int(((c == CLASS_NONWATER) & valid).sum()),
            "water_supported_pixels_C1": int(((c == CLASS_WATER) & valid).sum()),
            "uncertain_pixels": int(((c == CLASS_UNCERTAIN) & valid).sum()),
            "water_supported_pct_of_valid": pct(c == CLASS_WATER, valid),
            "uncertain_pct_of_valid": pct(c == CLASS_UNCERTAIN, valid),
            "interpretation": (
                "HH + spatial provisional candidates only; polarimetric evidence excluded"
                if date == SEPTEMBER else
                "C.1 exploratory evidence class; polarimetric features passed Stage A"
            )
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(OUT / "stage_C2_date_summary.csv", index=False)

    # Temporal reference summary.
    valid_count = int(valid_ref.sum())
    temporal_rows = []
    temporal_names = {
        1: "PERSISTENT_WATER_SUPPORTED_AUG_OCT",
        2: "AUG_ONLY_WATER_SUPPORTED",
        3: "OCT_ONLY_WATER_SUPPORTED",
        4: "PERSISTENT_NONWATER_SUPPORTED_AUG_OCT",
        5: "MIXED_OR_UNCERTAIN",
    }
    for code, name in temporal_names.items():
        count = int((temporal == code).sum())
        temporal_rows.append({
            "class_code": code,
            "class_name": name,
            "pixel_count": count,
            "percent_of_common_valid_pixels": 100.0 * count / valid_count if valid_count else np.nan,
        })
    pd.DataFrame(temporal_rows).to_csv(OUT / "stage_C2_temporal_consensus_summary.csv", index=False)

    sep_rows = []
    for code, name in {
        1: "SEPTEMBER_CANDIDATE_AGREES_WITH_PERSISTENT_REFERENCE_WATER",
        2: "SEPTEMBER_CANDIDATE_OUTSIDE_PERSISTENT_REFERENCE_WATER",
        3: "REFERENCE_PERSISTENT_WATER_NOT_SEPTEMBER_CANDIDATE",
        4: "OTHER_OR_UNCERTAIN",
    }.items():
        count = int((sep_compare == code).sum())
        denom = int((sep_compare > 0).sum())
        sep_rows.append({
            "class_code": code,
            "class_name": name,
            "pixel_count": count,
            "percent_of_comparable_pixels": 100.0 * count / denom if denom else np.nan,
        })
    pd.DataFrame(sep_rows).to_csv(OUT / "stage_C2_september_comparison.csv", index=False)

    # Save machine-readable rasters.
    np.save(OUT / "aug_oct_temporal_consensus.npy", temporal)
    np.save(OUT / "september_vs_reference_diagnostic.npy", sep_compare)
    np.save(OUT / "september_hh_spatial_candidate_mask.npy", sep_hh_spatial_candidate.astype(np.uint8))

    save_temporal_map(
        temporal,
        OUT / "aug_oct_temporal_consensus.png",
        "Stage C.2: August-October reference temporal consensus",
        ["Invalid", "Persistent water support", "August only", "October only",
         "Persistent non-water support", "Mixed / uncertain"],
        ["black", "#00a6d6", "#f28e2b", "#7b61a8", "#8c4517", "#ffd400"]
    )
    save_temporal_map(
        sep_compare,
        OUT / "september_vs_reference_diagnostic.png",
        "Stage C.2: September HH/spatial candidates vs accepted-date reference",
        ["Invalid", "Candidate agrees with persistent water",
         "Candidate outside persistent water", "Reference water, no Sep candidate",
         "Other / uncertain"],
        ["black", "#00a6d6", "#f28e2b", "#7b61a8", "#ffd400"]
    )

    for date in DATES:
        c = data[date]["class"]
        # September water class is explicitly renamed in its map.
        title = f"{date}: C.1 evidence class (exploratory)"
        labels = ["Invalid", "Non-water support", "Water support", "Uncertain"]
        if date == SEPTEMBER:
            title = f"{date}: HH + spatial candidates only (HV evidence flagged)"
            labels = ["Invalid", "Non-water support", "Provisional HH/spatial candidate", "Uncertain"]
        save_class_map(c, OUT / f"{date}_C1_interpreted.png", title, labels)

    report = {
        "stage": "C.2 - Temporal consistency and change analysis",
        "dates": DATES,
        "reference_dates": REFERENCE_DATES,
        "september_status": "FLAG",
        "september_role": (
            "Included as a diagnostic date. Its HH and spatial candidate evidence is compared "
            "against the August-October reference. Its anomalous HV and HV/HH evidence is not "
            "used to construct the reference consensus."
        ),
        "reference_consensus_definition": (
            "Persistent water support requires C.1 WATER_SUPPORTED in both August 19 and "
            "October 18. One-date-only support is marked separately; it is not automatically "
            "called a real hydrological change."
        ),
        "limitations": [
            "C.1 classes are exploratory evidence classes, not validated water labels.",
            "August and October are only two acquisitions; their agreement is not ground truth.",
            "A one-date difference may reflect water change, acquisition conditions, noise, "
            "or classification uncertainty.",
            "September's provisional HH/spatial candidates are not directly equivalent to "
            "polarimetrically supported water in the accepted dates.",
            "Independent reference data and threshold sensitivity analysis are still required."
        ],
        "input_files": input_files,
        "outputs": [
            "stage_C2_date_summary.csv",
            "stage_C2_temporal_consensus_summary.csv",
            "stage_C2_september_comparison.csv",
            "aug_oct_temporal_consensus.npy",
            "september_vs_reference_diagnostic.npy",
            "september_hh_spatial_candidate_mask.npy",
            "aug_oct_temporal_consensus.png",
            "september_vs_reference_diagnostic.png",
        ]
    }
    with open(OUT / "stage_C2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTAGE C.2 COMPLETE")
    print(f"Output folder: {OUT}")
    print("\nDate summary:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nTemporal consensus:")
    print(pd.DataFrame(temporal_rows).to_string(index=False))
    print("\nSeptember comparison:")
    print(pd.DataFrame(sep_rows).to_string(index=False))
    print("\nUpload these for review:")
    print("- stage_C2_date_summary.csv")
    print("- stage_C2_temporal_consensus_summary.csv")
    print("- stage_C2_september_comparison.csv")
    print("- aug_oct_temporal_consensus.png")
    print("- september_vs_reference_diagnostic.png")


if __name__ == "__main__":
    main()

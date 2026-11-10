"""QGPT-WM V2 Stage C: add HH/HV 5x5 local-CV texture features.

Keeps the supplied Stage C.1 outputs untouched. Uses pooled Aug+Oct
reference thresholds. September remains included; HV-derived evidence is
reported but excluded from its score because Stage A flagged the acquisition.
Outputs are exploratory evidence classes, not validated water labels.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import uniform_filter

BASE = Path(__file__).resolve().parent
AOI = BASE / "aoi_outputs"
A = BASE / "acquisition_quality_output"
B = BASE / "stage_B_candidate_output"
C1 = BASE / "stage_C1_validation_output"
OUT = BASE / "stage_C_texture_output"
DATES = ["20260819", "20260924", "20261018"]
REF_DATES = ["20260819", "20261018"]
WINDOW = 5
MIN_VALID_FRACTION = 0.80
HH_Q = HV_Q = 0.50
SPATIAL_WINDOW = 3
MIN_SPATIAL_SUPPORT = 0.50
EPS = 1e-12


def load(folder, name):
    p = folder / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing required input: {p}")
    return np.asarray(np.load(p, mmap_mode="r"))


def to_db(a):
    a = np.asarray(a, dtype=np.float32)
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & (a > 0)
    out[ok] = 10.0 * np.log10(a[ok])
    return out


def local_cv(data, valid, size=WINDOW):
    """NaN-aware local coefficient of variation on linear backscatter."""
    x = np.asarray(data, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(x) & (x > 0)
    w = ok.astype(np.float32)
    z = np.where(ok, x, 0).astype(np.float32)
    n = size * size
    count = uniform_filter(w, size=size, mode="constant") * n
    s1 = uniform_filter(z, size=size, mode="constant") * n
    s2 = uniform_filter(z * z, size=size, mode="constant") * n
    mean = np.divide(s1, count, out=np.zeros_like(s1), where=count > 0)
    mean2 = np.divide(s2, count, out=np.zeros_like(s2), where=count > 0)
    var = np.maximum(mean2 - mean * mean, 0)
    result = np.full(x.shape, np.nan, dtype=np.float32)
    good = (count >= MIN_VALID_FRACTION * n) & (mean > 0)
    result[good] = np.sqrt(var[good]) / (mean[good] + EPS)
    return result


def local_fraction(binary, valid, size=SPATIAL_WINDOW):
    n = size * size
    w = np.asarray(valid, dtype=np.float32)
    z = (np.asarray(binary, dtype=bool) & np.asarray(valid, dtype=bool)).astype(np.float32)
    count = uniform_filter(w, size=size, mode="constant") * n
    total = uniform_filter(z, size=size, mode="constant") * n
    out = np.full(w.shape, np.nan, dtype=np.float32)
    good = count > 0
    out[good] = total[good] / count[good]
    return out


def qtile(a, q):
    v = np.asarray(a)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q * 100)) if v.size else float("nan")


def status(flags):
    if np.any(flags == 2):
        return "FLAG"
    if np.any(flags == 1):
        return "ACCEPT"
    return "UNKNOWN"


def stats(a):
    v = np.asarray(a)
    v = v[np.isfinite(v)]
    if not v.size:
        return {}
    return {"p02": float(np.percentile(v, 2)), "median": float(np.median(v)),
            "p98": float(np.percentile(v, 98)), "mean": float(v.mean())}


def derive_texture_thresholds():
    hh_values, hv_values = [], []
    for date in REF_DATES:
        f = AOI / date
        hh = load(f, "HHHH").astype(np.float32)
        hv = load(f, "HVHV").astype(np.float32)
        mask = load(f, "mask")
        flags = load(A, f"{date}_quality_flags")
        hh_ok = (mask != 0) & (mask != 255) & (flags != 0) & np.isfinite(hh) & (hh > 0)
        hv_ok = hh_ok & np.isfinite(hv) & (hv > 0)
        htc = local_cv(hh, hh_ok)
        vtc = local_cv(hv, hv_ok)
        hh_values.append(htc[np.isfinite(htc)])
        hv_values.append(vtc[np.isfinite(vtc)])
    return {
        "HH_texture_low_CV_threshold": qtile(np.concatenate(hh_values), HH_Q),
        "HV_texture_low_CV_threshold": qtile(np.concatenate(hv_values), HV_Q),
    }


def process(date, tex_thr, polar_thr):
    f, bf, oldf = AOI / date, B / date, C1 / date
    hh = load(f, "HHHH").astype(np.float32)
    hv = load(f, "HVHV").astype(np.float32)
    mask = load(f, "mask")
    flags = load(A, f"{date}_quality_flags")
    candidates = load(bf, "water_candidate_mask").astype(bool)
    old_class = load(oldf, "preliminary_class")
    old_score = load(oldf, "evidence_score")  # validate matching C1 output exists
    if len({x.shape for x in (hh, hv, mask, flags, candidates, old_class, old_score)}) != 1:
        raise ValueError(f"{date}: input array shapes do not match")

    acq_status = status(flags)
    valid = (mask != 0) & (mask != 255) & (flags != 0) & np.isfinite(hh) & (hh > 0)
    hv_valid = valid & np.isfinite(hv) & (hv > 0)
    hh_db, hv_db = to_db(hh), to_db(hv)
    ratio_db = np.full(hh.shape, np.nan, dtype=np.float32)
    ratio_db[hv_valid] = 10 * np.log10(np.maximum(hv[hv_valid], EPS) / np.maximum(hh[hv_valid], EPS))

    hh_tex = local_cv(hh, valid)
    hv_tex = local_cv(hv, hv_valid)
    candidates &= valid
    support = local_fraction(candidates, valid)

    hh_low = valid & (hh_db <= polar_thr["HH_low_threshold_dB"])
    hv_low = hv_valid & (hv_db <= polar_thr["HV_low_threshold_dB"])
    ratio_low = hv_valid & (ratio_db <= polar_thr["HV_HH_ratio_low_threshold_dB"])
    spatial = valid & (support >= MIN_SPATIAL_SUPPORT)
    hh_tex_low = np.isfinite(hh_tex) & (hh_tex <= tex_thr["HH_texture_low_CV_threshold"])
    hv_tex_low = np.isfinite(hv_tex) & (hv_tex <= tex_thr["HV_texture_low_CV_threshold"])

    # Six eligible dimensions on ACCEPT dates; three on September/FLAG dates.
    eligible = candidates & np.isfinite(hh_tex) & np.isfinite(support)
    score = np.zeros(hh.shape, dtype=np.uint8)
    score[eligible & hh_low] += 1
    score[eligible & spatial] += 1
    score[eligible & hh_tex_low] += 1
    max_features = 3
    if acq_status == "ACCEPT":
        score[eligible & hv_low] += 1
        score[eligible & ratio_low] += 1
        score[eligible & hv_tex_low] += 1
        max_features = 6

    fraction = np.full(hh.shape, np.nan, dtype=np.float32)
    fraction[eligible] = score[eligible] / max_features
    classes = np.zeros(hh.shape, dtype=np.uint8)
    classes[valid] = 3
    # Use normalized thresholds so dates with fewer reliable features are not
    # mechanically penalized. This remains exploratory, not calibrated confidence.
    classes[eligible & (fraction >= 2/3)] = 2
    classes[eligible & (fraction <= 1/6)] = 1

    out = OUT / date
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "HH_texture_CV_5x5": hh_tex,
        "HV_texture_CV_5x5": hv_tex,
        "local_candidate_fraction": support,
        "texture_enhanced_evidence_score": score,
        "texture_enhanced_score_fraction": fraction,
        "texture_enhanced_class": classes,
        "HH_texture_low_flag": hh_tex_low.astype(np.uint8),
        "HV_texture_low_flag": hv_tex_low.astype(np.uint8),
    }
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (hh_tex, "HH local CV (5x5)", 0, 2),
        (hv_tex, "HV local CV (5x5)", 0, 2),
        (support, "Local candidate fraction", 0, 1),
        (score, "Texture-enhanced score", 0, max_features),
        (fraction, "Normalized evidence fraction", 0, 1),
        (classes, "Texture-enhanced classes", 0, 3),
    ]
    for ax, (arr, title, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("AOI column")
        ax.set_ylabel("AOI row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{date} | QGPT-WM V2 Stage C | Stage A: {acq_status} | HV score enabled: {acq_status == 'ACCEPT'}")
    fig.tight_layout()
    fig.savefig(out / "texture_feature_evidence.png", dpi=180)
    plt.close(fig)

    cmap = ListedColormap(["black", "saddlebrown", "deepskyblue", "gold"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(classes, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["Invalid", "Non-water supported", "Water supported", "Uncertain"])
    ax.set_title(f"{date}: texture-enhanced exploratory classes ({acq_status})")
    ax.set_xlabel("AOI column")
    ax.set_ylabel("AOI row")
    fig.tight_layout()
    fig.savefig(out / "texture_enhanced_classification_map.png", dpi=180)
    plt.close(fig)

    comparable = valid & (old_class > 0)
    changed = comparable & (classes != old_class)
    return {
        "date": date,
        "stage_A_status": acq_status,
        "HV_derived_features_used_in_score": acq_status == "ACCEPT",
        "texture_window": WINDOW,
        "minimum_valid_fraction_in_window": MIN_VALID_FRACTION,
        "HH_texture_threshold_CV": tex_thr["HH_texture_low_CV_threshold"],
        "HV_texture_threshold_CV": tex_thr["HV_texture_low_CV_threshold"],
        "valid_pixels": int(valid.sum()),
        "candidate_pixels": int(candidates.sum()),
        "HH_texture_valid_pixels": int(np.isfinite(hh_tex).sum()),
        "HV_texture_valid_pixels": int(np.isfinite(hv_tex).sum()),
        "HH_texture_CV_statistics": stats(hh_tex[valid]),
        "HV_texture_CV_statistics": stats(hv_tex[hv_valid]),
        "water_supported_pixels": int((classes == 2).sum()),
        "nonwater_supported_pixels": int((classes == 1).sum()),
        "uncertain_pixels": int((classes == 3).sum()),
        "old_C1_water_supported_pixels": int((old_class == 2).sum()),
        "class_changed_vs_C1_pixels": int(changed.sum()),
        "comparison_pixels": int(comparable.sum()),
        "note": "Exploratory classes only. September HV-derived evidence is diagnostic only.",
    }


def main():
    for folder in (AOI, A, B, C1):
        if not folder.exists():
            raise FileNotFoundError(f"Required folder not found: {folder}")
    with open(C1 / "stage_C1_report.json", "r", encoding="utf-8") as f:
        polar_thr = json.load(f)["common_thresholds"]

    OUT.mkdir(parents=True, exist_ok=True)
    tex_thr = derive_texture_thresholds()
    with open(OUT / "texture_reference_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({
            "reference_dates": REF_DATES,
            "window": WINDOW,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "quantiles": {"HH_texture": HH_Q, "HV_texture": HV_Q},
            "thresholds": tex_thr,
            "definition": "5x5 local coefficient of variation on linear Gamma0 backscatter",
            "caution": "Low texture is supporting evidence, not a physical water boundary.",
        }, f, indent=2)

    print("COMMON TEXTURE THRESHOLDS")
    for k, v in tex_thr.items():
        print(f"  {k}: {v:.6f}")
    results = []
    for date in DATES:
        print(f"\nProcessing {date} ...")
        r = process(date, tex_thr, polar_thr)
        results.append(r)
        print(f"  Stage A: {r['stage_A_status']}")
        print(f"  HV-derived evidence used: {r['HV_derived_features_used_in_score']}")
        print(f"  Water-supported: {r['water_supported_pixels']:,}")
        print(f"  Non-water-supported: {r['nonwater_supported_pixels']:,}")
        print(f"  Uncertain: {r['uncertain_pixels']:,}")
        print(f"  Changed vs C1: {r['class_changed_vs_C1_pixels']:,}")
    pd.DataFrame(results).to_csv(OUT / "stage_C_texture_statistics.csv", index=False)
    report = {
        "stage": "QGPT-WM Version 2 - Stage C texture-enhanced features",
        "dates_processed": DATES,
        "reference_dates": REF_DATES,
        "texture_thresholds": tex_thr,
        "september_policy": "September retained; HH texture is score-eligible; HV texture and other HV-derived features are diagnostic only.",
        "class_codes": {"0": "INVALID", "1": "NON_WATER_SUPPORTED", "2": "WATER_SUPPORTED", "3": "UNCERTAIN"},
        "warning": "Exploratory evidence classes, not trained or independently validated water labels. Texture is affected by speckle, mixed pixels, vegetation and land-cover heterogeneity.",
        "acquisitions": {r["date"]: r for r in results},
    }
    with open(OUT / "stage_C_texture_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(f"\nSTAGE C TEXTURE ENHANCEMENT COMPLETE. Outputs: {OUT}")


if __name__ == "__main__":
    main()

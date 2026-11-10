"""
diagnostics.py — measurement of spatial independence for a partition.

Extracted into its own module because this is the component that supplies the
EVIDENCE for every independence claim in the study. A strategy's name is never
accepted as evidence (research_protocol.md §9).

Representation note — the reason this module is more than a few lines:

  Partitions are stored in one of two forms.

  COMPACT (S0, S1, S2, S4)
      One row per sample. `fold` is the fold the sample belongs to; `split` is
      'val' for every row, because a sample is validation in exactly one fold and
      training in the others. The training set for fold k is therefore derived
      as `fold != k`, NOT read from a 'train' label. Reading a literal 'train'
      label here would yield an empty training set and silently null every
      diagnostic — which is the defect this module exists to avoid.

  EXPANDED (S3, buffered)
      One row per (sample, eval_fold). Buffering excludes a sample from the
      training set of a *specific* fold only, so a single `split` column cannot
      express it. `eval_fold` identifies the fold whose partition the row
      describes, and `split` is then literal: 'val', 'train' or 'excluded_buffer'.

  Global exclusions (the matched-sample-size control) appear in both forms as
  `split == 'excluded_matched_sample_size'` and are always honoured.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EXCLUDED_SPLITS = {"excluded_buffer", "excluded_matched_sample_size", "excluded"}


def fold_train_sizes(a: pd.DataFrame) -> dict[int, int]:
    """Per-fold TRAINING-set size, correct for both partition representations.

    This exists because `(assign["split"] == "train").sum()` is WRONG for the
    compact representation: `random_folds` marks every sample `split="val"` and
    gives each its own `fold`, so the literal train count is 0. Code that
    trusted that count would compute a training size of zero and then silently
    skip the matched-sample-size control -- a protocol-required safeguard
    disabled by a representation mismatch rather than by a visible error.

    Callers must use this instead of counting the `split` column directly.
    """
    return {k: len(tr) for k, _val, tr in _fold_partitions(a)}


def fold_val_sizes(a: pd.DataFrame) -> dict[int, int]:
    """Per-fold VALIDATION-set size, correct for both representations."""
    return {k: len(val) for k, val, _tr in _fold_partitions(a)}


def representative_train_size(a: pd.DataFrame) -> int:
    """Median per-fold training size, for reporting and for matching controls.

    The median rather than the mean: with leave-one-region-out the fold sizes
    can differ substantially, and a mean would describe no actual fold.
    """
    sizes = list(fold_train_sizes(a).values())
    if not sizes:
        return 0
    return int(np.median(sizes))


def fold_frames(a: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) for a single fold.

    The training loop must obtain its membership from here rather than filtering
    the `split` column itself. Filtering literally is wrong for the compact
    representation (every sample is `split="val"`), and the resulting mistake is
    silent: the loader would either train on the validation fold or on nothing.
    """
    for k, val, tr in _fold_partitions(a):
        if int(k) == int(fold):
            return tr, val
    raise KeyError(
        f"fold {fold} not present in this partition; available: "
        f"{sorted(int(k) for k, _, _ in _fold_partitions(a))}")


def _fold_partitions(a: pd.DataFrame) -> list[tuple[Any, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_id, val_frame, train_frame) for every fold in the partition."""
    out: list[tuple[Any, pd.DataFrame, pd.DataFrame]] = []

    if "eval_fold" in a.columns:
        for k in sorted(a["eval_fold"].unique()):
            sub = a[a["eval_fold"] == k]
            val = sub[sub["split"] == "val"]
            tr = sub[sub["split"] == "train"]
            out.append((k, val, tr))
        return out

    if "fold" not in a.columns:
        raise ValueError("partition has neither 'eval_fold' nor 'fold'")

    for k in sorted(a["fold"].unique()):
        val = a[a["fold"] == k]
        tr = a[(a["fold"] != k) & (~a["split"].isin(EXCLUDED_SPLITS))]
        out.append((k, val, tr))
    return out


def _pairs_prop(vals: pd.Series, train_vals: set) -> float | None:
    if len(vals) == 0:
        return None
    return float(np.mean([v in train_vals for v in vals]))


def leakage_diagnostics(
    assign: pd.DataFrame,
    meta: pd.DataFrame,
    coord_cols: tuple[str, str] = ("lon", "lat"),
    phash: dict[str, str] | None = None,
    sha: dict[str, str] | None = None,
    scene_col: str | None = None,
    region_col: str = "region",
    adjacency_km: float = 0.05,
    phash_threshold: int = 6,
) -> dict[str, Any]:
    """Quantify spatial independence of one partition, aggregated over folds.

    Returns the row for the spatial independence diagnostics table. Nothing here
    is inferred from the strategy's name.
    """
    from splits.strategies import nearest_distance_km

    a = assign.merge(meta, on="sample_id", how="left", suffixes=("", "_m"))
    if "split" not in a.columns:
        a["split"] = "val"

    try:
        folds = _fold_partitions(a)
    except ValueError as exc:
        return {"strategy": None, "error": str(exc)}

    xy_ok = (coord_cols[0] in a.columns and coord_cols[1] in a.columns
             and a[coord_cols[0]].notna().any() and a[coord_cols[1]].notna().any())

    mins, medians, means, p05s, adj = [], [], [], [], []
    dup, near, scene, region = [], [], [], []
    n_train, n_val = [], []

    import imagehash  # imported once; the loop below is O(n_val * n_train)

    for _k, val, train in folds:
        n_train.append(len(train))
        n_val.append(len(val))

        if xy_ok and len(val) and len(train):
            vxy = val[list(coord_cols)].to_numpy(float)
            txy = train[list(coord_cols)].to_numpy(float)
            # Filter each side separately — the two masks differ in length.
            vxy = vxy[np.isfinite(vxy).all(axis=1)]
            txy = txy[np.isfinite(txy).all(axis=1)]
            if len(vxy) and len(txy):
                d = nearest_distance_km(vxy, txy)
                mins.append(float(np.nanmin(d)))
                medians.append(float(np.nanmedian(d)))
                means.append(float(np.nanmean(d)))
                p05s.append(float(np.nanpercentile(d, 5)))
                adj.append(float(np.nanmean(d < adjacency_km)))

        if sha is not None and len(val):
            tr = {sha[s] for s in train["sample_id"] if s in sha}
            dup.append(_pairs_prop(val["sample_id"].map(sha), tr))

        if phash is not None and len(val) and len(train):
            trh = []
            for s in train["sample_id"]:
                h = phash.get(s)
                if h:
                    try:
                        trh.append(imagehash.hex_to_hash(h))
                    except Exception:                      # noqa: BLE001
                        pass
            if trh:
                hits = 0
                n = 0
                for s in val["sample_id"]:
                    h = phash.get(s)
                    if not h:
                        continue
                    n += 1
                    try:
                        vh = imagehash.hex_to_hash(h)
                    except Exception:                      # noqa: BLE001
                        continue
                    if any((vh - t) <= phash_threshold for t in trh):
                        hits += 1
                if n:
                    near.append(hits / n)

        if scene_col and scene_col in a.columns and len(val):
            scene.append(_pairs_prop(val[scene_col].astype(str),
                                     set(train[scene_col].astype(str))))

        if region_col in a.columns and len(val):
            region.append(_pairs_prop(val[region_col].astype(str),
                                      set(train[region_col].astype(str))))

    def agg(vals: list[float], how: str) -> float | None:
        v = [x for x in vals if x is not None and np.isfinite(x)]
        if not v:
            return None
        return float({"min": np.min, "median": np.median, "mean": np.mean}[how](v))

    res: dict[str, Any] = {
        "n_folds": len(folds),
        "n_train_median": int(np.median(n_train)) if n_train else None,
        "n_train_min": int(np.min(n_train)) if n_train else None,
        "n_train_max": int(np.max(n_train)) if n_train else None,
        "n_val_median": int(np.median(n_val)) if n_val else None,
        "n_excluded_buffer": int((a.get("split", pd.Series(dtype=str))
                                  == "excluded_buffer").sum()),
        "n_excluded_matched": int((a.get("split", pd.Series(dtype=str))
                                   == "excluded_matched_sample_size").sum()),
    }

    if mins:
        res.update({
            "min_train_val_dist_km": agg(mins, "min"),
            "median_train_val_dist_km": agg(medians, "median"),
            "mean_train_val_dist_km": agg(means, "mean"),
            "p05_train_val_dist_km": agg(p05s, "mean"),
            "adjacent_patch_prop": agg(adj, "mean"),
        })
    else:
        res.update({
            "min_train_val_dist_km": None,
            "median_train_val_dist_km": None,
            "mean_train_val_dist_km": None,
            "p05_train_val_dist_km": None,
            "adjacent_patch_prop": None,
            "distance_note": ("coordinates unavailable, or no fold produced a "
                              "non-empty training set"),
        })

    res["duplicate_prop"] = agg(dup, "mean")
    res["near_duplicate_prop"] = agg(near, "mean")
    res["same_scene_contamination_prop"] = agg(scene, "mean")
    res["same_region_contamination_prop"] = agg(region, "mean")
    return res

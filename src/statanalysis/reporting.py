"""
reporting.py — turn the experiment registry into the study's result tables.

Kept separate from `analysis.py` so that the statistical primitives stay
independently testable and this module only has to worry about assembly:
grouping, pairing, and the bookkeeping the protocol demands (resampling unit,
family definition, and the explicit primary/secondary distinction).

The one rule that shapes everything here: **the primary contrast is
pre-specified and tested without multiplicity adjustment; every other contrast
is secondary and exploratory and goes through Benjamini-Hochberg.** That
distinction is enforced by which function a contrast is routed to, not by a
convention the analyst is asked to remember.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from statanalysis.analysis import (
    benjamini_hochberg, paired_bootstrap_diff, wilcoxon_signed_rank,
)

__all__ = [
    "estimator_summary",
    "paired_contrasts",
    "primary_contrast_s0_vs_ext",
    "strategy_level_summary",
]

ESTIMATOR_COLS = ["ob_miou", "aee_miou", "gg_miou", "ri_miou"]


def _finite(x: Iterable[float]) -> np.ndarray:
    return np.asarray([v for v in x if v is not None and np.isfinite(v)],
                      dtype=float)


def _filter_eq(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Filter `frame` to rows where `column` equals `value`.

    Written as a helper rather than inline because the obvious
    `frame.get(column, pd.Series(dtype=str)) == value` silently produces an
    empty Series when the column is absent, and using that as a boolean mask
    raises on the index mismatch instead of returning no rows.
    """
    if column not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame[column].astype(str) == value]


# --------------------------------------------------------------------------- #
def estimator_summary(
    df: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = ("val_miou", "ext_miou", "ob_miou", "aee_miou",
                                  "gg_miou", "ri_miou"),
    group_cols: Sequence[str] = ("strategy", "model_id"),
    resampling_unit: str = "seed",
    seed: int = 0,
) -> pd.DataFrame:
    """Mean, sd and bootstrap interval of every metric, per group.

    The interval is over the group's own replicates (seeds by default), so
    `resampling_unit` must name what those replicates are. It is written into
    the output as a column rather than being left implicit, because protocol
    13.1 requires the unit to be stated wherever an interval is reported.
    """
    rows: list[dict] = []
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(group_cols) + ["metric", "n", "mean",
                                                        "sd", "ci_low", "ci_high",
                                                        "resampling_unit"])

    keys = [c for c in group_cols if c in df.columns]
    if not keys:
        raise ValueError(f"none of {group_cols} present in the frame")

    for key, sub in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(keys, [str(k) for k in key]))
        for col in metric_cols:
            if col not in sub.columns:
                continue
            v = _finite(sub[col])
            if len(v) == 0:
                continue
            ci = paired_bootstrap_diff(v, np.zeros(len(v)),
                                       n_resamples=4000, seed=seed,
                                       resampling_unit=resampling_unit)
            rows.append({
                **base, "metric": col, "n": int(len(v)),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "median": float(np.median(v)),
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                "resampling_unit": resampling_unit,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def strategy_level_summary(
    df: pd.DataFrame,
    *,
    metric: str = "miou",
    strategy_col: str = "strategy",
    model_col: str = "model_id",
    resampling_unit: str = "region",
    per_region: pd.DataFrame | None = None,
    eval_set: str = "ext",
) -> pd.DataFrame:
    """One row per strategy, aggregating over models and seeds.

    If `per_region` is supplied the interval is computed over regions, which is
    the correct unit for a geographic design: regions, not patches or seeds, are
    the independent replicates when the question is how a strategy behaves
    across geography.
    """
    rows: list[dict] = []
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[strategy_col, "metric", "n", "mean", "sd",
                                     "ci_low", "ci_high", "resampling_unit",
                                     "region_mean", "region_ci_low",
                                     "region_ci_high", "n_regions",
                                     "region_resampling_unit"])

    for strat, sub in df.groupby(strategy_col):
        col = f"{eval_set}_{metric}" if f"{eval_set}_{metric}" in sub.columns else metric
        v = _finite(sub[col])
        if len(v) == 0:
            continue
        rec: dict[str, Any] = {
            "strategy": str(strat), "metric": col, "n": int(len(v)),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci_low": None, "ci_high": None, "resampling_unit": resampling_unit,
            "region_mean": None, "region_ci_low": None, "region_ci_high": None,
            "n_regions": None, "region_resampling_unit": None,
        }

        # ---- the interval that belongs with THIS point estimate ----------- #
        # `mean` averages over runs (models x seeds). The interval must
        # therefore resample runs. It previously resampled REGIONS instead while
        # keeping the run-level mean, so the two described different estimands:
        # for S0 the row read mean = 0.5242 with CI [0.4428, 0.5129] -- an
        # interval that does not contain its own point estimate, which is the
        # signature of exactly this mistake (D-048).
        ci_run = paired_bootstrap_diff(v, np.zeros(len(v)), n_resamples=4000,
                                       resampling_unit="run")
        rec.update(ci_low=ci_run["ci_low"], ci_high=ci_run["ci_high"],
                   resampling_unit="model x seed")

        # ---- the region-level estimand, reported separately --------------- #
        # Protocol section 13.1 asks for the region as the resampling unit for
        # geographic designs. That is a statement about a REGION-level estimand,
        # so it is reported under its own name with its own mean, rather than
        # bolted onto a run-level mean.
        if per_region is not None and len(per_region):
            pr = _filter_eq(per_region, strategy_col, str(strat))
            pr = _filter_eq(pr, "eval_set", eval_set)
            if len(pr) and pr["region"].nunique() >= 3:
                # Average over models/seeds within each region first, so each
                # region contributes exactly one value and the resampling unit
                # really is the region.
                by_region = (pr.groupby("region")[metric].mean()
                             .dropna().to_numpy(dtype=float))
                if len(by_region) >= 3:
                    ci_reg = paired_bootstrap_diff(
                        by_region, np.zeros(len(by_region)), n_resamples=4000,
                        resampling_unit="region")
                    rec.update(
                        region_mean=float(np.mean(by_region)),
                        region_ci_low=ci_reg["ci_low"],
                        region_ci_high=ci_reg["ci_high"],
                        n_regions=int(len(by_region)),
                        region_resampling_unit="region")
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def strategy_id(label: Any) -> str:
    """The protocol's strategy ID for a partition label.

    Partitions are named by design -- `S0_random_patch`, `S3_buffered_1r` -- while
    research_protocol.md and the pre-registered contrast name strategies by ID:
    S0, S1, S2, S3, S4, EXT. The primary contrast compared the two directly, so
    it matched nothing and returned

        {"note": "no runs for strategy 'S0'"}      (D-047)

    while the statistics gate still reported PASS. Mapping the label to its ID is
    the one place that knowledge belongs; callers keep passing protocol IDs.
    """
    return str(label).split("_", 1)[0].split("-", 1)[0].strip().upper()


def select_strategy(df: pd.DataFrame, strategy: str,
                    strategy_col: str = "strategy") -> pd.DataFrame:
    """Rows belonging to a protocol strategy ID, whatever the partition is named."""
    want = str(strategy).strip().upper()
    return df[df[strategy_col].map(strategy_id) == want]


# --------------------------------------------------------------------------- #
def primary_contrast_s0_vs_ext(
    df: pd.DataFrame,
    *,
    strategy: str = "S0",
    val_col: str = "val_miou",
    ext_col: str = "ext_miou",
    per_region: pd.DataFrame | None = None,
    strategy_col: str = "strategy",
) -> dict[str, Any]:
    """The ONE pre-specified contrast: S0 versus EXT on mIoU (protocol 13.4).

    Tested without multiplicity adjustment, because it was named in advance and
    is not one of a family of comparisons searched for significance.

    The contrast is a comparison of two ESTIMATORS of the same model's
    generalisation, paired within each (model, seed): the validation-fold
    estimate and the external geographic estimate. Both are computed on disjoint
    data by construction, so the pairing is over runs, not over samples.

    A region-level version is attempted and will normally be **unavailable by
    design**: the external regions are excluded from every partition, so no
    region carries both a validation and an external estimate. When that is the
    case the reason is returned in `region_level_unavailable_reason` rather than
    being left as a silent None, because "not identifiable" and "not computed
    yet" are different claims and only one of them is true here.
    """
    out: dict[str, Any] = {
        "name": "PRIMARY: S0 vs EXT on mIoU",
        "pre_specified": True,
        "multiplicity_adjusted": False,
        "rationale": ("Named in advance in protocol section 13.4; tested without "
                      "adjustment. Every other contrast in this study is "
                      "secondary and exploratory."),
        "strategy": strategy, "region_level": None, "run_level": None,
        "region_level_unavailable_reason": None, "note": None,
    }
    if df is None or len(df) == 0:
        out["note"] = "no experiments on record"
        return out

    sub = select_strategy(df, strategy, strategy_col)
    if len(sub) == 0:
        out["note"] = (f"no runs for strategy {strategy!r} "
                       f"(running: "
                       f"{sorted(set(df[strategy_col].astype(str)))[:8]})")
        return out

    v = _finite(sub[val_col])
    e = _finite(sub[ext_col])
    if len(v) and len(e):
        out["run_level"] = paired_bootstrap_diff(
            v, e, n_resamples=10_000, resampling_unit="model x seed",
            seed=0)
        out["run_level"]["mean_ob"] = float(np.mean(v[:min(len(v), len(e))]
                                                   - e[:min(len(v), len(e))]))
        w = wilcoxon_signed_rank(v, e)
        out["run_level"]["wilcoxon"] = w

    if per_region is not None and len(per_region):
        pr = _filter_eq(per_region, strategy_col, strategy)
        if len(pr) == 0:
            pr = select_strategy(per_region, strategy, strategy_col)
        if len(pr):
            val_regions = set(_filter_eq(pr, "eval_set", "val")["region"].astype(str))
            ext_regions = set(_filter_eq(pr, "eval_set", "ext")["region"].astype(str))
            shared = val_regions & ext_regions

            # A region-level paired contrast requires a region to carry BOTH a
            # validation and an external estimate. That is structurally
            # impossible here, and the impossibility is a property of the study
            # design rather than a gap in the data: the external regions are
            # excluded from every partition, so an external region never has a
            # validation estimate, and a development region never has an
            # external one. Reporting this as "not computed" would understate
            # it -- it is not available in principle.
            if not shared:
                out["region_level_unavailable_reason"] = (
                    f"the validation and external region sets are disjoint "
                    f"({len(val_regions)} observed validation regions vs "
                    f"{len(ext_regions)} external regions, 0 in common), which is "
                    f"guaranteed by the freeze: external regions are excluded "
                    f"from every partition, so no region carries both a "
                    f"validation and an external estimate. A region-level paired "
                    f"contrast is therefore not identifiable by design, and the "
                    f"run-level interval is conditional on the fixed geography; "
                    f"shared evaluation data and geographic sampling uncertainty "
                    f"are not captured by model x seed resampling.")
            elif len(shared) < 3:
                out.setdefault("notes", []).append(
                    f"only {len(shared)} region(s) carry both a validation and an "
                    f"external estimate under {strategy}; too few for a "
                    f"region-level paired contrast")
            else:
                # One value per (region, model, seed) then averaged per region,
                # so each region contributes a single paired difference.
                wide = pr.pivot_table(
                    index=["region", "model_id", "seed"], columns="eval_set",
                    values="miou", aggfunc="mean")
                if {"val", "ext"} <= set(wide.columns):
                    wide = wide.dropna(subset=["val", "ext"])
                    by_region = (wide["val"] - wide["ext"]).groupby("region").mean()
                    by_region = by_region[np.isfinite(by_region)]
                    if len(by_region) >= 3:
                        out["region_level"] = paired_bootstrap_diff(
                            by_region.to_numpy(), np.zeros(len(by_region)),
                            n_resamples=10_000, resampling_unit="region", seed=0)
                        out["region_level"]["n_regions"] = int(len(by_region))
                        out["region_level"]["mean_ob"] = float(by_region.mean())
                        out["region_level"]["median_ob"] = float(by_region.median())
                        out["region_level"]["n_regions_positive"] = int(
                            (by_region > 0).sum())
                        out["region_level"]["n_regions_negative"] = int(
                            (by_region < 0).sum())

    if out["region_level"] is None and out["run_level"] is None:
        out["note"] = "insufficient paired observations for either version"
    return out


# --------------------------------------------------------------------------- #
def paired_contrasts(
    df: pd.DataFrame,
    *,
    a_col: str = "ob_miou",
    pairs: Sequence[tuple[str, str]] | None = None,
    strategy_col: str = "strategy",
    alpha: float = 0.05,
    family: str = "all pairwise strategy contrasts on OB (secondary, exploratory)",
) -> pd.DataFrame:
    """Every secondary contrast, with BH control over the stated family.

    The family is fixed here and recorded in the output, because protocol 13.4
    requires the family to be defined in advance rather than assembled after
    seeing which comparisons were interesting.

    The pairing is on (model, seed), so each contrast is a within-model,
    within-seed comparison between two strategies. `a_col` is the signed
    optimism bias, so a positive mean difference means the first strategy is
    more optimistic than the second.
    """
    cols = ["contrast", "strategy_a", "strategy_b", "n_pairs", "mean_a", "mean_b",
            "mean_diff", "ci_low", "ci_high", "p_value", "p_adjusted", "reject",
            "cohens_dz", "resampling_unit", "family"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    strategies = sorted(df[strategy_col].astype(str).unique().tolist())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(strategies)
                 for b in strategies[i + 1:]]

    pair_keys = [c for c in ("model_id", "seed") if c in df.columns]
    rows: list[dict] = []
    for sa, sb in pairs:
        da = df[df[strategy_col].astype(str) == sa]
        db = df[df[strategy_col].astype(str) == sb]
        if len(da) == 0 or len(db) == 0:
            continue
        if pair_keys:
            ma = da.groupby(pair_keys)[a_col].mean()
            mb = db.groupby(pair_keys)[a_col].mean()
            common = ma.index.intersection(mb.index)
            if len(common) < 2:
                continue
            va, vb = ma.loc[common].to_numpy(), mb.loc[common].to_numpy()
        else:
            va, vb = _finite(da[a_col]), _finite(db[a_col])
            if len(va) < 2 or len(vb) < 2:
                continue
        r = paired_bootstrap_diff(va, vb, n_resamples=10_000,
                                  resampling_unit="model x seed", seed=0)
        rows.append({
            "contrast": f"{sa} - {sb}", "strategy_a": sa, "strategy_b": sb,
            "n_pairs": r["n_pairs"], "mean_a": r["mean_a"], "mean_b": r["mean_b"],
            "mean_diff": r["mean_diff"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "p_value": r["p_value"],
            "cohens_dz": r["cohens_dz"], "resampling_unit": "model x seed",
            "family": family,
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    bh = benjamini_hochberg(out["p_value"].tolist(), alpha=alpha)
    out["p_adjusted"] = bh["p_adjusted"]
    out["reject"] = bh["reject"]
    out["family"] = family
    return out[cols]

"""
analysis.py — the statistical layer (protocol section 13).

Three things are implemented here, in the order the protocol specifies them.

1. UNCERTAINTY (13.1). Bootstrap intervals with an explicitly declared
   resampling unit. The unit is not a detail: for a geographic design the
   independent unit is the region, and resampling patches would produce an
   interval that is far too narrow, because patches inside a region are
   spatially dependent. Every function that returns an interval also returns the
   unit it used, so the value cannot be reported without it.

2. MODELLING (13.2). The pre-registered mixed-effects model
   `Metric ~ Strategy + Model + Strategy x Model + (1 | Region)`, with explicit
   assumption diagnostics. The protocol requires that a switch away from the
   parametric model be reported together with its reason, so the diagnostics are
   returned as data rather than printed and forgotten.

3. RANKING STABILITY (13.3) and MULTIPLICITY (13.4). Spearman rho / Kendall tau
   against the external-test ranking with individually examined reversals, and
   Benjamini-Hochberg FDR control over a family defined in advance.

Directional neutrality is enforced by construction: `ranking_concordance`
reports concordance and reversal without any branch that treats a reversal as
the expected outcome. If the rankings agree, that is what is reported.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "fit_mixed_effects",
    "paired_bootstrap_diff",
    "ranking_concordance",
    "wilcoxon_signed_rank",
]


def _clean(x: Iterable[float]) -> np.ndarray:
    return np.asarray([v for v in x if v is not None and np.isfinite(v)],
                      dtype=float)


# --------------------------------------------------------------------------- #
# 13.1 Uncertainty
# --------------------------------------------------------------------------- #
def paired_bootstrap_diff(
    a: Iterable[float],
    b: Iterable[float],
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    resampling_unit: str = "unspecified",
) -> dict[str, Any]:
    """Paired bootstrap interval for `mean(a) - mean(b)`.

    Pairing matters: the two arms of every contrast in this study are measured
    on the *same* seed and the *same* external test set, so the seed-to-seed
    variation is shared and cancels. Treating the arms as independent samples
    would inflate the interval.

    The p-value is a two-sided sign-flip permutation test on the paired
    differences. It is exact under exchangeability and assumes nothing about the
    distribution of the differences, which is what the protocol asks for when
    the parametric assumptions of the mixed model do not hold.

    `resampling_unit` is echoed back into the result and must be stated wherever
    the interval is reported.
    """
    av, bv = _clean(a), _clean(b)
    n = min(len(av), len(bv))
    out: dict[str, Any] = {
        "n_pairs": int(n), "resampling_unit": resampling_unit,
        "n_resamples": int(n_resamples), "confidence_level": float(confidence_level),
        "mean_a": None, "mean_b": None, "mean_diff": None,
        "ci_low": None, "ci_high": None, "p_value": None,
        "cohens_dz": None, "cliffs_delta": None,
    }
    if n < 2:
        if n == 1:
            out.update(mean_a=float(av[0]), mean_b=float(bv[0]),
                       mean_diff=float(av[0] - bv[0]), ci_low=float(av[0] - bv[0]),
                       ci_high=float(av[0] - bv[0]), p_value=None)
        return out

    av, bv = av[:n], bv[:n]
    d = av - bv
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    alpha = 1.0 - confidence_level
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided sign-flip permutation test.
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    null = (signs * d).mean(axis=1)
    # +1 in both counts keeps the p-value strictly positive, so a result can
    # never be reported as p = 0 from a finite number of resamples.
    p = float((np.abs(null) >= abs(obs)).sum() + 1) / (n_resamples + 1)

    sd = d.std(ddof=1)
    gt = float(sum((x > y) for x in av for y in bv))
    lt = float(sum((x < y) for x in av for y in bv))

    out.update(
        mean_a=float(av.mean()), mean_b=float(bv.mean()), mean_diff=obs,
        ci_low=float(lo), ci_high=float(hi), p_value=p,
        cohens_dz=float(obs / sd) if sd > 0 else None,
        cliffs_delta=float((gt - lt) / (n * n)),
    )
    return out


def wilcoxon_signed_rank(a: Iterable[float], b: Iterable[float]
                         ) -> dict[str, Any]:
    """Non-parametric fallback for a paired contrast (protocol 13.2)."""
    from scipy.stats import wilcoxon

    av, bv = _clean(a), _clean(b)
    n = min(len(av), len(bv))
    if n < 6:
        return {"statistic": None, "p_value": None, "n_pairs": int(n),
                "note": "too few pairs for a signed-rank test; not computed"}
    d = av[:n] - bv[:n]
    if np.allclose(d, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(n),
                "note": "all paired differences are zero"}
    stat, p = wilcoxon(d, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": int(n),
            "note": ""}


# --------------------------------------------------------------------------- #
# 13.4 Multiplicity
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05
                       ) -> dict[str, Any]:
    """Benjamini-Hochberg FDR control over a pre-defined family.

    Returns the adjusted p-values in the original order together with the
    reject/retain decision at `alpha`. NaN entries are passed through and
    excluded from the ranking rather than being treated as 1.0, so an undefined
    test cannot silently absorb a rejection.
    """
    p = np.asarray([np.nan if v is None else float(v) for v in p_values],
                   dtype=float)
    m = int(np.isfinite(p).sum())
    adj = np.full(p.shape, np.nan, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return {"p_values": p.tolist(), "p_adjusted": adj.tolist(),
                "reject": reject.tolist(), "n_tests": 0, "alpha": float(alpha)}

    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order]

    # Step-up: adjusted_(i) = min over j >= i of (m / j) * p_(j)
    scaled = ranked * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    adj[order] = scaled
    reject[order] = scaled <= alpha

    return {
        "p_values": p.tolist(), "p_adjusted": adj.tolist(),
        "reject": reject.tolist(), "n_tests": m, "alpha": float(alpha),
    }


# --------------------------------------------------------------------------- #
# 13.2 Modelling
# --------------------------------------------------------------------------- #
def fit_mixed_effects(
    df,
    *,
    response: str = "miou",
    fixed_effects: str = "C(strategy) * C(model_id)",
    group: str = "region",
    reml: bool = False,
) -> dict[str, Any]:
    """Fit `response ~ fixed_effects + (1 | group)` and diagnose it.

    Returns a dictionary, never raises on a model failure: a singular fit is a
    finding to be reported, and the protocol requires the switch to a
    non-parametric alternative to be recorded with its reason. The caller
    decides what to do; this function only reports.

    Diagnostics returned:
      - convergence flag and the exception text if it failed
      - residual normality (Shapiro-Wilk) and its p-value
      - heteroscedasticity (Breusch-Pagan) and its p-value
      - random-effect variance and the intraclass correlation
      - the fixed-effect table as plain records
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    out: dict[str, Any] = {
        "formula": f"{response} ~ {fixed_effects} + (1 | {group})",
        "response": response, "group": group, "reml": bool(reml),
        "n_obs": 0, "n_groups": 0, "converged": False,
        "error": None, "fixed_effects": [], "random_effect_var": None,
        "residual_var": None, "icc": None,
        "shapiro_w": None, "shapiro_p": None,
        "breusch_pagan_lm": None, "breusch_pagan_p": None,
        "singular_covariance": False,
        "assumptions_met": None, "fallback_recommended": None,
        "fallback_reason": None,
    }

    # Fail with a diagnosis rather than a patsy NameError. A term that is
    # simply absent from the frame is a schema mismatch between the artefact
    # and the pre-registered formula, and that is worth naming explicitly --
    # patsy reports it as an undefined variable deep inside formula parsing.
    referenced = {
        t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fixed_effects)
        if t not in {"C", "np", "log", "exp", "I", "Q", "center", "standardize"}
    }
    referenced.add(group)
    referenced.add(response)
    missing = sorted(t for t in referenced if t not in df.columns)
    if missing:
        out["error"] = (
            f"the frame is missing term(s) {missing} required by the formula "
            f"'{response} ~ {fixed_effects} + (1 | {group})'. Available columns: "
            f"{sorted(df.columns)}. This is a schema mismatch between the "
            f"artefact and the pre-registered model, not a fitting failure.")
        out["fallback_recommended"] = True
        out["fallback_reason"] = out["error"]
        return out

    need = [response, group]
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    # A group with a single observation contributes nothing to a random-effect
    # variance and can make the covariance singular.
    counts = d.groupby(group)[group].transform("size")
    dropped_singletons = int((counts < 2).sum())
    d = d[counts >= 2].copy()
    out["n_obs"] = int(len(d))
    out["n_groups"] = int(d[group].nunique()) if len(d) else 0
    out["dropped_singleton_rows"] = dropped_singletons

    if out["n_obs"] < 10 or out["n_groups"] < 3:
        out["fallback_recommended"] = True
        out["fallback_reason"] = (
            f"insufficient structure for a random intercept: n_obs={out['n_obs']}, "
            f"n_groups={out['n_groups']}")
        return out

    try:
        md = smf.mixedlm(f"{response} ~ {fixed_effects}", d, groups=d[group])
        fit = md.fit(reml=reml, method="lbfgs")
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["fallback_recommended"] = True
        out["fallback_reason"] = f"mixed model failed to fit: {out['error']}"
        return out

    out["converged"] = bool(getattr(fit, "converged", False))
    out["error"] = None if out["converged"] else "optimiser reported non-convergence"

    try:
        params = fit.params
        bse = fit.bse
        pvals = fit.pvalues
        out["fixed_effects"] = [
            {"term": str(k), "coef": float(params[k]), "se": float(bse[k]),
             "p_value": float(pvals[k])}
            for k in params.index
        ]
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = (out["error"] or "") + f" | parameter extraction failed: {exc}"

    try:
        re_var = float(np.asarray(fit.cov_re).ravel()[0])
        resid_var = float(fit.scale)
        out["random_effect_var"] = re_var
        out["residual_var"] = resid_var
        if re_var + resid_var > 0:
            out["icc"] = re_var / (re_var + resid_var)
    except Exception:                                          # noqa: BLE001
        pass

    # ---- assumption diagnostics ------------------------------------------- #
    # Every access below is guarded. statsmodels raises
    # `ValueError: Cannot predict random effects from singular covariance
    # structure` from `fit.resid` / `fit.fittedvalues` when the random-effect
    # variance sits on the boundary. That is a substantive finding -- it means
    # the data do not support a region random effect -- and it must be reported
    # as such, not allowed to abort the analysis.
    out["singular_covariance"] = False
    resid = np.empty(0, dtype=float)
    try:
        resid = np.asarray(fit.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
    except Exception as exc:                                   # noqa: BLE001
        if "singular" in str(exc).lower():
            out["singular_covariance"] = True
        out["residual_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    if len(resid) >= 3:
        try:
            from scipy.stats import shapiro
            if len(resid) <= 5000:
                w, pw = shapiro(resid)
            else:
                rng = np.random.default_rng(0)
                w, pw = shapiro(rng.choice(resid, 5000, replace=False))
            out["shapiro_w"], out["shapiro_p"] = float(w), float(pw)
        except Exception:                                      # noqa: BLE001
            pass

    try:
        import statsmodels.stats.diagnostic as diag
        fitted = np.asarray(fit.fittedvalues, dtype=float)
        if len(fitted) == len(resid) and len(resid) >= 8:
            lm, lm_p, _, _ = diag.het_breuschpagan(resid, np.column_stack(
                [np.ones(len(fitted)), fitted]))
            out["breusch_pagan_lm"], out["breusch_pagan_p"] = float(lm), float(lm_p)
    except Exception as exc:                                   # noqa: BLE001
        if "singular" in str(exc).lower():
            out["singular_covariance"] = True
        out.setdefault("heteroscedasticity_diagnostics_error",
                       f"{type(exc).__name__}: {exc}")

    reasons: list[str] = []
    if not out["converged"]:
        reasons.append("the optimiser did not converge")
    if out["singular_covariance"]:
        reasons.append(
            "the random-effect covariance is singular, i.e. the estimated "
            "region variance is on the boundary of the parameter space, so "
            "these data do not support a random intercept for region")
    if out["shapiro_p"] is not None and out["shapiro_p"] < 0.01:
        reasons.append(f"residuals are not normal (Shapiro-Wilk p={out['shapiro_p']:.4g})")
    if out["breusch_pagan_p"] is not None and out["breusch_pagan_p"] < 0.01:
        reasons.append(
            f"residual variance is not constant (Breusch-Pagan p={out['breusch_pagan_p']:.4g})")
    if out["icc"] is not None and out["icc"] < 1e-6:
        reasons.append(
            "the region random-effect variance is effectively zero, so the "
            "clustered structure is not supported by these data")

    out["assumptions_met"] = len(reasons) == 0
    out["fallback_recommended"] = len(reasons) > 0
    out["fallback_reason"] = "; ".join(reasons) if reasons else None
    if out["fallback_recommended"]:
        out["fallback_note"] = (
            "Protocol 13.2: where assumptions are not met a non-parametric or "
            "bootstrap alternative is used instead, and the switch and its "
            "reason are reported. See paired_bootstrap_diff / "
            "wilcoxon_signed_rank for that alternative.")
    return out


# --------------------------------------------------------------------------- #
# 13.3 Ranking stability (RQ5)
# --------------------------------------------------------------------------- #
def ranking_concordance(
    df,
    *,
    strategy_col: str = "strategy",
    model_col: str = "model_id",
    metric_col: str = "miou",
    reference_strategy: str = "EXT",
    group_cols: Sequence[str] = ("seed", "fold"),
) -> dict[str, Any]:
    """Rank models by `metric_col` under each strategy and compare with EXT.

    Concordance is measured against the external geographic test ranking, which
    is the reference because it is the only ranking not produced by a
    validation strategy under study.

    Reversals are listed individually, as the protocol requires, rather than
    being reduced to a count: which pair of models swapped, and by how much, is
    the interpretable quantity.

    Directional neutrality: there is no branch here that treats a reversal as
    the expected outcome. A concordant result is returned as a concordant
    result.
    """
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr

    out: dict[str, Any] = {
        "reference_strategy": reference_strategy, "metric": metric_col,
        "n_strategies": 0, "per_strategy": [], "reference_ranking": None,
        "note": None,
    }
    if df is None or len(df) == 0:
        out["note"] = "empty input"
        return out

    d = df.dropna(subset=[metric_col]).copy()
    if d.empty:
        out["note"] = f"no finite values in {metric_col}"
        return out

    gcols = [c for c in group_cols if c in d.columns]
    if not gcols:
        d["_all"] = 0
        gcols = ["_all"]

    def _mean_rank(frame):
        m = frame.groupby(model_col)[metric_col].mean().sort_values(ascending=False)
        return {str(k): i + 1 for i, k in enumerate(m.index)}, m.to_dict()

    per_group: dict[tuple, dict] = {}
    for key, sub in d.groupby(gcols):
        if not isinstance(key, tuple):
            key = (key,)
        per_group[key] = _mean_rank(sub)

    ref_map: dict[tuple, dict] = {}
    if strategy_col in d.columns:
        ref_rows = d[d[strategy_col].astype(str) == reference_strategy]
        for key, sub in ref_rows.groupby(gcols):
            if not isinstance(key, tuple):
                key = (key,)
            ref_map[key] = _mean_rank(sub)

    if not ref_map:
        out["note"] = (f"no rows for the reference strategy {reference_strategy!r}; "
                       f"ranking concordance not computed")
        return out

    ref_mean = (d[d[strategy_col].astype(str) == reference_strategy]
                .groupby(model_col)[metric_col].mean()
                .sort_values(ascending=False))
    out["reference_ranking"] = [str(x) for x in ref_mean.index]

    strategies = [s for s in d[strategy_col].astype(str).unique()
                  if s != reference_strategy]
    out["n_strategies"] = len(strategies)

    for strat in sorted(strategies):
        sub = d[d[strategy_col].astype(str) == strat]
        mean_by_model = sub.groupby(model_col)[metric_col].mean()
        common = [m for m in ref_mean.index if m in mean_by_model.index]
        rec: dict[str, Any] = {
            "strategy": strat, "n_models": len(common),
            "ranking": [str(m) for m in
                        mean_by_model.sort_values(ascending=False).index],
            "spearman_rho": None, "spearman_p": None,
            "kendall_tau": None, "kendall_p": None,
            "n_reversals": 0, "reversals": [],
            "rank_difference_max": None,
        }
        if len(common) >= 3:
            a = np.array([ref_mean[m] for m in common], dtype=float)
            b = np.array([mean_by_model[m] for m in common], dtype=float)
            rho, pr = spearmanr(a, b)
            tau, pt = kendalltau(a, b)
            rec.update(spearman_rho=float(rho), spearman_p=float(pr),
                       kendall_tau=float(tau), kendall_p=float(pt))
        elif len(common) >= 2:
            a = np.array([ref_mean[m] for m in common], dtype=float)
            b = np.array([mean_by_model[m] for m in common], dtype=float)
            rho, pr = spearmanr(a, b)
            rec.update(spearman_rho=float(rho), spearman_p=float(pr))

        # Rank reversals, examined individually.
        ref_rank = {str(m): i + 1 for i, m in enumerate(ref_mean.index)}
        st_rank = {str(m): i + 1 for i, m in
                   enumerate(mean_by_model.sort_values(ascending=False).index)}
        revs: list[dict] = []
        for i, m1 in enumerate(common):
            for m2 in common[i + 1:]:
                r1 = ref_rank[str(m1)] - ref_rank[str(m2)]
                r2 = st_rank[str(m1)] - st_rank[str(m2)]
                if r1 * r2 < 0:
                    revs.append({
                        "model_a": str(m1), "model_b": str(m2),
                        "rank_a_ext": ref_rank[str(m1)],
                        "rank_b_ext": ref_rank[str(m2)],
                        "rank_a_strategy": st_rank[str(m1)],
                        "rank_b_strategy": st_rank[str(m2)],
                        "metric_a_ext": float(ref_mean[m1]),
                        "metric_b_ext": float(ref_mean[m2]),
                        "metric_a_strategy": float(mean_by_model[m1]),
                        "metric_b_strategy": float(mean_by_model[m2]),
                    })
        rec["reversals"] = revs
        rec["n_reversals"] = len(revs)
        diffs = [abs(ref_rank[str(m)] - st_rank[str(m)]) for m in common]
        rec["rank_difference_max"] = int(max(diffs)) if diffs else None
        out["per_strategy"].append(rec)

    return out

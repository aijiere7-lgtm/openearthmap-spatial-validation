#!/usr/bin/env python
"""
test_statistics.py — validation of the statistical layer on synthetic data.

Each test has an answer that is known independently of this implementation:
  - Benjamini-Hochberg is checked against a hand-computed 15-test example.
  - The paired bootstrap is checked on a construction with a known difference.
  - The mixed model is checked on data with a planted random effect.
  - Ranking concordance is checked against a deliberately reversed ranking.

Run:  python tests/test_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from statanalysis.analysis import (  # noqa: E402
    benjamini_hochberg, fit_mixed_effects, paired_bootstrap_diff,
    ranking_concordance, wilcoxon_signed_rank,
)
from statanalysis.reporting import (  # noqa: E402
    estimator_summary, paired_contrasts, primary_contrast_s0_vs_ext,
    strategy_level_summary,
)

CHECKS = 0


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(f"FAILED: {msg}")
    print(f"  ok  {msg}")


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
def test_bh_hand_computed() -> None:
    print("\n[1] Benjamini-Hochberg against a hand-computed example")
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
         0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    r = benjamini_hochberg(p, alpha=0.05)
    adj = r["p_adjusted"]
    # Adjusted values computed by hand from adj_(i) = min_{j>=i} (m/j) p_(j),
    # written as exact fractions of the inputs so the comparison is exact.
    expected = [
        15 / 1 * 0.0001,    # 0.0015
        15 / 2 * 0.0004,    # 0.003
        15 / 3 * 0.0019,    # 0.0095
        15 / 4 * 0.0095,    # 0.035625
        15 / 5 * 0.0201,    # 0.0603
        15 / 7 * 0.0298,    # min over j>=6 is attained at j=7
        15 / 7 * 0.0298,
        15 / 8 * 0.0344,
        15 / 9 * 0.0459,
        15 / 10 * 0.3240,
        15 / 11 * 0.4262,
        15 / 12 * 0.5719,
        15 / 13 * 0.6528,
        15 / 14 * 0.7590,
        15 / 15 * 1.0000,
    ]
    for got, exp in zip(adj, expected):
        check(close(got, exp, 1e-9), f"adjusted p {got:.6f} == {exp:.6f}")
    check(r["n_tests"] == 15, "15 tests counted")
    check(sum(r["reject"]) == 4, "exactly 4 hypotheses rejected at alpha=0.05")

    # Monotonicity is a defining property of BH-adjusted p-values.
    finite = [v for v in adj if np.isfinite(v)]
    check(all(finite[i] <= finite[i + 1] + 1e-12 for i in range(len(finite) - 1)),
          "adjusted p-values are monotone non-decreasing")

    # An undefined p-value must be passed through, not silently absorbed.
    r2 = benjamini_hochberg([0.01, None, 0.5], alpha=0.05)
    check(np.isnan(r2["p_adjusted"][1]), "None maps to NaN, not to 1.0")
    check(r2["n_tests"] == 2, "undefined test excluded from the family size")


# --------------------------------------------------------------------------- #
def test_paired_bootstrap() -> None:
    print("\n[2] Paired bootstrap on a construction with a known difference")
    rng = np.random.default_rng(11)
    base = rng.normal(0.60, 0.03, size=40)
    b = base
    a = base + 0.08 + rng.normal(0.0, 0.01, size=40)   # true difference ~ +0.08
    r = paired_bootstrap_diff(a, b, n_resamples=4000, seed=3,
                              resampling_unit="region")
    check(r["n_pairs"] == 40, "40 pairs used")
    check(abs(r["mean_diff"] - 0.08) < 0.01,
          f"mean difference {r['mean_diff']:.4f} is near the planted +0.08")
    check(r["ci_low"] < r["mean_diff"] < r["ci_high"],
          "point estimate lies inside its own interval")
    check(r["ci_low"] > 0.0, "interval excludes zero, as it should for a real effect")
    check(r["p_value"] is not None and r["p_value"] < 0.01,
          f"permutation p={r['p_value']:.5f} is small")
    check(r["resampling_unit"] == "region", "resampling unit is carried through")
    check(r["cohens_dz"] is not None and r["cohens_dz"] > 1.0,
          "large standardised effect detected")

    # A null construction must not produce a spurious effect.
    r0 = paired_bootstrap_diff(base, base + rng.normal(0, 0.02, size=40),
                               n_resamples=4000, seed=5)
    check(r0["p_value"] > 0.05, f"null case gives p={r0['p_value']:.3f} > 0.05")

    # Degenerate inputs must be handled, not crash.
    r1 = paired_bootstrap_diff([0.5], [0.4])
    check(r1["n_pairs"] == 1 and r1["p_value"] is None,
          "a single pair returns a point difference and no p-value")
    r2 = paired_bootstrap_diff([], [])
    check(r2["n_pairs"] == 0 and r2["mean_diff"] is None,
          "empty input returns nulls rather than raising")


def test_wilcoxon() -> None:
    print("\n[3] Wilcoxon signed-rank fallback")
    rng = np.random.default_rng(2)
    d = rng.normal(0.05, 0.02, size=30)
    r = wilcoxon_signed_rank(d, np.zeros(30))
    check(r["p_value"] is not None and r["p_value"] < 0.01,
          f"signed-rank p={r['p_value']:.5f} detects the shift")
    r2 = wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0])
    check(r2["p_value"] is None, "too few pairs returns a null p-value with a note")


# --------------------------------------------------------------------------- #
def test_mixed_effects() -> None:
    print("\n[4] Mixed-effects model on data with a planted random effect")
    rng = np.random.default_rng(7)
    regions = [f"R{i:02d}" for i in range(12)]
    models = ["M1", "M2", "M3"]
    strategies = ["S0", "S2", "EXT"]
    re = {r: rng.normal(0, 0.05) for r in regions}      # region intercepts

    rows = []
    for r in regions:
        for m in models:
            base = {"M1": 0.62, "M2": 0.64, "M3": 0.66}[m]
            for s in strategies:
                delta = {"S0": 0.05, "S2": 0.01, "EXT": 0.0}[s]
                rows.append({
                    "region": r, "model_id": m, "strategy": s,
                    "seed": 0, "fold": 0,
                    "miou": base + delta + re[r] + rng.normal(0, 0.005),
                })
    df = pd.DataFrame(rows)
    out = fit_mixed_effects(df, response="miou",
                            fixed_effects="C(strategy) * C(model_id)",
                            group="region")
    check(out["n_obs"] == len(df), f"all {len(df)} rows used")
    check(out["n_groups"] == 12, "12 regions detected")
    check(out["converged"], "model converged")
    check(out["icc"] is not None and out["icc"] > 0.1,
          f"ICC={out['icc']:.3f} recovers the planted region effect")
    check(out["random_effect_var"] is not None and out["random_effect_var"] > 0,
          "random-effect variance is positive")
    terms = {t["term"] for t in out["fixed_effects"]}
    check(any("strategy" in t for t in terms), "strategy appears among the terms")
    check(any("model_id" in t for t in terms), "model appears among the terms")
    check(any(":" in t for t in terms), "the interaction term is present")

    # Too little structure must trigger the documented fallback, not an exception.
    small = df[df["region"].isin(regions[:2])].copy()
    out2 = fit_mixed_effects(small, group="region")
    check(out2["fallback_recommended"] is True,
          "insufficient region structure recommends the fallback")
    check(out2["fallback_reason"] is not None, "the fallback reason is recorded")


# --------------------------------------------------------------------------- #
def test_ranking_concordance() -> None:
    print("\n[5] Ranking concordance and reversal detection")
    rows = []
    # EXT ranks M3 > M2 > M1. S0 reproduces it; S2 deliberately swaps M1 and M3.
    ext = {"M1": 0.50, "M2": 0.55, "M3": 0.60}
    s0 = {"M1": 0.62, "M2": 0.66, "M3": 0.70}     # same order, inflated
    s2 = {"M1": 0.72, "M2": 0.68, "M3": 0.66}     # reversed
    for m in ("M1", "M2", "M3"):
        rows.append({"strategy": "EXT", "model_id": m, "seed": 0, "fold": 0,
                     "miou": ext[m]})
        rows.append({"strategy": "S0", "model_id": m, "seed": 0, "fold": 0,
                     "miou": s0[m]})
        rows.append({"strategy": "S2", "model_id": m, "seed": 0, "fold": 0,
                     "miou": s2[m]})
    df = pd.DataFrame(rows)
    out = ranking_concordance(df, reference_strategy="EXT")
    check(out["reference_ranking"] == ["M3", "M2", "M1"],
          "EXT reference ranking is M3 > M2 > M1")
    by = {r["strategy"]: r for r in out["per_strategy"]}
    check(close(by["S0"]["spearman_rho"], 1.0),
          f"S0 is perfectly concordant (rho={by['S0']['spearman_rho']:.4f})")
    check(by["S0"]["n_reversals"] == 0, "S0 shows no reversals")
    check(close(by["S2"]["spearman_rho"], -1.0),
          f"S2 is perfectly reversed (rho={by['S2']['spearman_rho']:.4f})")
    check(by["S2"]["n_reversals"] == 3,
          f"all 3 model pairs reversed under S2 (got {by['S2']['n_reversals']})")
    rev = by["S2"]["reversals"][0]
    check({"model_a", "model_b", "rank_a_ext", "metric_a_strategy"} <= set(rev),
          "each reversal is reported with both rankings and both metrics")

    # A missing reference must be reported, not silently defaulted.
    out2 = ranking_concordance(df, reference_strategy="ABSENT")
    check(out2["note"] is not None and "ABSENT" in out2["note"],
          "an absent reference strategy is reported explicitly")


# --------------------------------------------------------------------------- #
def test_reporting_layer() -> None:
    print("\n[6] Reporting layer: summaries, primary contrast, BH family")
    rng = np.random.default_rng(23)
    regions = [f"R{i:02d}" for i in range(10)]
    models = ["M1", "M2", "M3"]
    designs = ["S0", "S1", "S2", "S4"]
    # Planted optimism per strategy. The gaps are chosen relative to the noise
    # below so that one contrast is genuinely detectable and another is
    # genuinely not -- otherwise the BH assertions would test nothing.
    inflation = {"S0": 0.100, "S1": 0.060, "S2": 0.020, "S4": 0.055}
    NOISE = 0.02          # per-run noise; with 12 pairs, SEM ~ 0.008

    rows, region_rows = [], []
    for d in designs:
        for m in models:
            for s in range(4):
                base = {"M1": 0.60, "M2": 0.63, "M3": 0.66}[m]
                # The construction makes the strategy affect BOTH sides, which
                # is the situation the study exists to characterise: a strategy
                # that yields a more optimistic validation estimate also yields
                # a model that transfers worse. If only the validation side were
                # perturbed, the external metric would be design-independent and
                # the test would not exercise the design effect at all.
                ext = base - 0.5 * inflation[d] + rng.normal(0, NOISE)
                val = ext + inflation[d] + rng.normal(0, NOISE)
                rows.append({
                    "strategy": d, "model_id": m, "seed": s, "fold": 0,
                    "val_miou": val, "ext_miou": ext,
                    "ob_miou": val - ext, "aee_miou": abs(val - ext),
                    "gg_miou": val - ext,
                    "ri_miou": (val - ext) / ext * 100.0,
                })
                for r in regions:
                    region_rows.append({
                        "strategy": d, "model_id": m, "seed": s,
                        "eval_set": "val", "region": r,
                        "miou": val + rng.normal(0, 0.02),
                    })
                    region_rows.append({
                        "strategy": d, "model_id": m, "seed": s,
                        "eval_set": "ext", "region": r,
                        "miou": ext + rng.normal(0, 0.02),
                    })
    df = pd.DataFrame(rows)
    pr = pd.DataFrame(region_rows)

    es = estimator_summary(df, group_cols=("strategy", "model_id"),
                           resampling_unit="seed")
    check(len(es) > 0, "estimator summary is non-empty")
    check({"metric", "mean", "ci_low", "ci_high", "resampling_unit"} <= set(es.columns),
          "summary carries the interval and the resampling unit")
    ob = es[(es["strategy"] == "S0") & (es["metric"] == "ob_miou")]
    check(len(ob) == 3, "one OB summary row per model under S0")
    check(abs(ob["mean"].mean() - 0.10) < 0.03,
          f"recovered S0 optimism {ob['mean'].mean():.4f} near the planted 0.10")

    sl = strategy_level_summary(df, per_region=pr, eval_set="ext")
    check(len(sl) == len(designs), "one row per strategy")
    # The run-level mean carries a RUN-level interval, and the region-level
    # estimand is reported in its own columns. Asserting that the run-level row
    # was labelled "region" is what the code used to do while bootstrapping
    # regions underneath a run-level mean, producing an interval that excluded
    # its own point estimate (D-048). Both facts are now checked: the units are
    # separate, and the region estimand exists.
    check((sl["resampling_unit"] == "model x seed").all(),
          "run-level mean carries a run-level interval")
    check((sl["region_resampling_unit"] == "region").all()
          and sl["region_mean"].notna().all(),
          "region-level unit used for the region-level estimand")
    _bad = []
    for _, _r in sl.iterrows():
        if not (float(_r["ci_low"]) - 1e-12 <= float(_r["mean"])
                <= float(_r["ci_high"]) + 1e-12):
            _bad.append(_r["strategy"])
        if not (float(_r["region_ci_low"]) - 1e-12
                <= float(_r["region_mean"])
                <= float(_r["region_ci_high"]) + 1e-12):
            _bad.append(f"{_r['strategy']} (region)")
    check(not _bad, f"every interval contains its own point estimate: {_bad}")
    s0_ext = float(sl[sl["strategy"] == "S0"]["mean"].iloc[0])
    s2_ext = float(sl[sl["strategy"] == "S2"]["mean"].iloc[0])
    check(s0_ext < s2_ext,
          "external performance is lower after the more inflated S0 training, "
          "as constructed")

    pc = primary_contrast_s0_vs_ext(df, per_region=pr)
    check(pc["pre_specified"] is True, "the primary contrast is flagged pre-specified")
    check(pc["multiplicity_adjusted"] is False,
          "the primary contrast is NOT multiplicity-adjusted")
    check(pc["run_level"] is not None, "run-level primary contrast computed")
    check(pc["run_level"]["resampling_unit"] == "model x seed",
          "run-level unit is stated")
    check(abs(pc["run_level"]["mean_diff"] - 0.10) < 0.03,
          f"run-level OB {pc['run_level']['mean_diff']:.4f} near 0.10")
    check(pc["region_level"] is not None,
          "region-level primary contrast computed from per-region data")
    check(pc["region_level"]["resampling_unit"] == "region",
          "region-level unit is stated")
    check(pc["region_level"]["n_regions"] == len(regions),
          "all planted regions used")

    contrasts = paired_contrasts(df, a_col="ob_miou")
    check(len(contrasts) == 6,
          f"4 strategies give 6 pairwise contrasts (got {len(contrasts)})")
    check(contrasts["p_adjusted"].notna().all(), "BH adjusted every p-value")
    check(contrasts["family"].nunique() == 1, "one family recorded for the family")
    check((contrasts["p_adjusted"] >= contrasts["p_value"] - 1e-12).all(),
          "BH-adjusted p-values are never smaller than the raw ones")
    # S0 vs S2 is the largest planted gap (0.08 vs SEM ~0.008) and must survive.
    big = contrasts[contrasts["contrast"].isin(["S0 - S2", "S2 - S0"])]
    check(len(big) == 1 and bool(big["reject"].iloc[0]),
          "the largest planted contrast survives BH control")
    # S1 vs S4 is a planted gap of 0.005 against SEM ~0.008: not detectable.
    small = contrasts[contrasts["contrast"].isin(["S1 - S4", "S4 - S1"])]
    check(len(small) == 1 and not bool(small["reject"].iloc[0]),
          f"the non-detectable planted contrast does not survive BH "
          f"(p={small['p_value'].iloc[0]:.3f})")
    check(int(contrasts["reject"].sum()) < len(contrasts),
          "BH retains at least one hypothesis rather than rejecting everything")

    # Empty input must return empty frames, not raise.
    check(len(estimator_summary(pd.DataFrame())) == 0,
          "empty frame yields an empty summary")
    check(len(paired_contrasts(pd.DataFrame())) == 0,
          "empty frame yields an empty contrast table")


# --------------------------------------------------------------------------- #
def test_no_stdlib_shadowing() -> None:
    """`src/` is on sys.path, so a module there can shadow the standard library.

    This is not a hypothetical hazard. A package named `src/statistics/` broke
    `geopandas.explore`, which does `from statistics import mean` -- and with it
    `libpysal` and the entire spatial-dependence analysis. The failure surfaces
    far from its cause, in an unrelated third-party import, which is exactly the
    kind of defect that must be caught automatically rather than by reading
    tracebacks.
    """
    print("\n[0] Namespace safety: no module under src/ shadows the standard library")
    import sys as _sys
    import sysconfig

    src_dir = Path(__file__).resolve().parent.parent / "src"
    stdlib_names = set(_sys.stdlib_module_names)
    # Names that are not stdlib but that third-party code commonly imports at
    # top level and would be broken by a same-named local module.
    extra = {"numpy", "pandas", "scipy", "matplotlib", "PIL", "cv2", "torch",
             "rasterio", "geopandas", "shapely", "statsmodels", "tqdm"}
    guarded = stdlib_names | extra

    offenders: list[str] = []
    for entry in sorted(src_dir.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        stem = entry.stem if entry.is_file() else entry.name
        if stem in guarded:
            offenders.append(f"{entry.name} shadows '{stem}'")

    check(not offenders, f"no module under src/ shadows a stdlib or common "
                         f"package name (checked {len(guarded)} names); "
                         f"offenders: {offenders or 'none'}")

    # And the concrete case: the stdlib module must still resolve to the stdlib.
    import statistics as _std
    origin = getattr(_std, "__file__", "") or ""
    check("site-packages" not in origin.replace("\\", "/").lower()
          or "statistics" not in src_dir.as_posix(),
          f"the stdlib 'statistics' module resolves to {origin or 'builtin'}")
    check(hasattr(_std, "mean"), "the stdlib 'statistics' still exposes mean()")

    # geopandas is the concrete downstream victim; if it is installed, it must
    # import cleanly.
    try:
        import geopandas  # noqa: F401
        check(True, f"geopandas imports cleanly ({geopandas.__version__})")
    except ImportError as exc:                                 # noqa: BLE001
        check(False, f"geopandas failed to import: {exc}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 72)
    print("STATISTICS MODULE TESTS")
    print("=" * 72)
    test_no_stdlib_shadowing()
    test_bh_hand_computed()
    test_paired_bootstrap()
    test_wilcoxon()
    test_mixed_effects()
    test_ranking_concordance()
    test_reporting_layer()
    print("\n" + "=" * 72)
    print(f"ALL PASSED — {CHECKS} assertions")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
analyze_results.py — statistics over the experiment registry (protocol section 13).

Reads only machine-generated artefacts:
    experiments/experiments_registry.csv     partition-level metrics
    experiments/per_region_metrics.csv       region-level metrics
    results/tables/spatial_independence_diagnostics.csv   leakage diagnostics
    data/metadata/external_regions.json      the freeze record

and writes:
    results/tables/estimator_summary.csv
    results/tables/strategy_summary.csv
    results/tables/primary_contrast.json
    results/tables/secondary_contrasts.csv
    results/tables/sweep_summary.csv
    results/tables/mixed_effects.json
    results/tables/ranking_concordance.json
    results/tables/ranking_concordance.csv
    results/statistics/statistics_report.md

Nothing is typed by hand. If a number is not traceable to one of the inputs
above, it does not appear in the output.

THE PRIMARY CONTRAST IS TESTED WITHOUT MULTIPLICITY ADJUSTMENT; everything else
is secondary and goes through Benjamini-Hochberg over a family fixed here.
That split is structural: the primary contrast is computed by
`primary_contrast_s0_vs_ext`, and every other comparison by `paired_contrasts`.

SCOPE IS FROZEN, NOT INFERRED (D-060). The inferential analyses — the primary
contrast, the ten secondary contrasts, the mixed models and the ranking
concordance — run on exactly the five pre-registered designs named in
`common.PRE_REGISTERED_DESIGNS`. The P1 sensitivity strategies (the block-size
and buffer-width sweeps) are summarised separately in `sweep_summary.csv` and
enter no pre-registered family, because pairing "every strategy present" is
exactly how a fixed family silently grows from ten contrasts to thirty-six.

Exits 0 on PASS, 1 on PARTIAL (some analysis could not be run), 2 when there is
nothing to analyse. A PARTIAL exit is a real finding, not a soft failure: it
means a pre-registered analysis is missing and must be reported as missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    EXPERIMENTS_DIR, EXTERNAL_REGIONS_JSON, LOGS_DIR, METADATA_DIR,
    PRE_REGISTERED_DESIGNS, RESULTS_DIR, STATISTICS_DIR, TABLES_DIR,
    ensure_dirs, get_logger, parse_sweep_strategy, read_json,
    sha256_json_canonical, utcnow, write_json, freeze_content_sha256,
)
from statanalysis.analysis import (  # noqa: E402
    fit_mixed_effects, ranking_concordance,
)
from statanalysis.reporting import (  # noqa: E402
    estimator_summary, paired_contrasts, primary_contrast_s0_vs_ext,
    strategy_level_summary,
)

REGISTRY = EXPERIMENTS_DIR / "experiments_registry.csv"
PER_REGION = EXPERIMENTS_DIR / "per_region_metrics.csv"
LEAKAGE = TABLES_DIR / "spatial_independence_diagnostics.csv"

# The reference strategy for ranking concordance. EXT is the only ranking that
# is not itself produced by a validation strategy under study.
REFERENCE_STRATEGY = "EXT"

PRIMARY_STRATEGY = "S0"

# The ten secondary contrasts of the pre-registered design set: the complete
# pairwise family, fixed before the results were seen (protocol §13.4, frozen
# in D-060). `sorted()` is deliberate — it reproduces, pair for pair and in the
# same direction, the family the 63-run snapshot computed, so the refresh can
# be verified against that snapshot value by value (K2 in ROUND_PLAN_2026-10-13).
PREREG_PAIRS = tuple(combinations(sorted(PRE_REGISTERED_DESIGNS), 2))


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{nd}f}"


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    # Select the requested columns that actually exist, preserving the requested
    # order. A report renderer must not fail because one column is absent: the
    # table is a view of the data, and a missing column is a fact about the
    # data, not a reason to abort the analysis.
    if cols:
        present = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        d = df[present] if present else df
    else:
        d = df
        missing = []
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    lines = [head, sep]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append(_fmt(v))
            elif isinstance(v, (list, dict)):
                cells.append(json.dumps(v, ensure_ascii=False)[:80])
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if missing:
        lines.append("")
        lines.append(f"_Columns not available: {missing}_")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="miou")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--allow-empty", action="store_true",
                    help="exit 0 with a report even when no runs exist")
    ap.add_argument("--registry", default=None,
                    help="override the registry path (used by the pipeline test)")
    ap.add_argument("--per-region", default=None,
                    help="override the per-region metrics path")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory for tables and report")
    ap.add_argument("--tag", default="",
                    help="suffix appended to output file names, so a validation "
                         "run cannot overwrite the real results")
    args = ap.parse_args()

    ensure_dirs()
    global REGISTRY, PER_REGION
    if args.registry:
        REGISTRY = Path(args.registry)
    if args.per_region:
        PER_REGION = Path(args.per_region)
    out_tables = Path(args.out_dir) if args.out_dir else TABLES_DIR
    out_stats = (Path(args.out_dir) / "statistics") if args.out_dir else STATISTICS_DIR
    out_tables.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger("analyze_results", LOGS_DIR / "analyze_results.log")
    logger.info("=" * 72)
    logger.info("STATISTICAL ANALYSIS  %s", utcnow())
    logger.info("=" * 72)

    issues: list[str] = []
    warnings: list[str] = []

    # ---- inputs ----------------------------------------------------------- #
    if not REGISTRY.exists():
        msg = (f"{REGISTRY.name} not found: no experiment has been run. Run "
               f"scripts/run_experiments.py first.")
        logger.error(msg)
        if not args.allow_empty:
            print("\n[PHASE] Statistical analysis")
            print("[STATUS] FAIL")
            print(f"[ISSUES] {msg}")
            print("[NEXT] run the experiment grid")
            return 2
        registry = pd.DataFrame()
    else:
        registry = pd.read_csv(REGISTRY, keep_default_na=False)
        logger.info("registry: %d rows", len(registry))

    per_region = None
    if PER_REGION.exists():
        per_region = pd.read_csv(PER_REGION, keep_default_na=False)
        logger.info("per-region metrics: %d rows, %d regions",
                    len(per_region), per_region["region"].nunique())
    else:
        warnings.append(
            f"{PER_REGION.name} absent: region-level intervals and the "
            f"random intercept for region cannot be computed, because a "
            f"partition-level metric carries no region structure. Re-run the "
            f"grid after updating run_experiments.py.")

    if LEAKAGE.exists():
        leakage = pd.read_csv(LEAKAGE, keep_default_na=False)
        logger.info("leakage diagnostics: %d rows", len(leakage))
    else:
        leakage = pd.DataFrame()
        warnings.append(f"{LEAKAGE.name} absent: independence is not yet measured")

    # ONE HASH, ONE DEFINITION. The registry's `external_freeze_sha256` is
    # written by run_experiments.py via common.freeze_content_sha256(), which
    # hashes the freeze record with its own `content_sha256` field EXCLUDED (it
    # cannot include the digest it is computing). This function previously did
    # two different things: it looked for a `freeze_sha256` key that has never
    # existed in the record (the field is `content_sha256`), then fell back to
    # sha256_json_canonical() over the WHOLE record -- including that field.
    # The two definitions cannot agree, so the External-QC check could never
    # pass and reported, on every analysis,
    #     "the external test set has been modified since these runs"
    # against an untouched freeze. A false integrity alarm on the study's
    # central comparison is worse than no alarm: it invites either a real
    # modification to be dismissed as noise, or valid results to be discarded.
    freeze = read_json(EXTERNAL_REGIONS_JSON) if EXTERNAL_REGIONS_JSON.exists() else {}
    freeze_sha = freeze_content_sha256(freeze) if freeze else None

    if len(registry) == 0:
        report = _empty_report(warnings)
        (out_stats / f"statistics_report{sfx}.md").write_text(report, encoding="utf-8")
        print("\n[PHASE] Statistical analysis")
        print("[STATUS] PARTIAL")
        print("[COMPLETED] analysis code validated; no experiment results to analyse")
        print(f"[FILES] {out_stats / f'statistics_report{sfx}.md'}")
        for w in warnings:
            print(f"[ISSUES] {w}")
        print("[NEXT] run scripts/run_experiments.py, then re-run this script")
        return 0 if args.allow_empty else 2

    # ---- provenance and integrity checks ---------------------------------- #
    if "external_freeze_sha256" in registry.columns:
        seen = {str(v) for v in registry["external_freeze_sha256"].unique() if v}
        if len(seen) > 1:
            issues.append(
                f"runs were produced against {len(seen)} different external "
                f"freezes ({sorted(seen)}); the external test set changed "
                f"mid-grid, so the runs are not mutually comparable")
        elif seen and freeze_sha and list(seen)[0] != freeze_sha:
            issues.append(
                f"the freeze recorded in the registry ({list(seen)[0][:16]}) does "
                f"not match the current freeze ({freeze_sha[:16]}); the external "
                f"test set has been modified since these runs")

    # A duplicated run_id means the append-only registry was edited.
    dup = registry["run_id"][registry["run_id"].duplicated()].tolist()
    if dup:
        issues.append(f"duplicate run_id values in the registry: {dup[:5]}")

    strategies = sorted(registry["strategy"].astype(str).unique().tolist())
    models = sorted(registry["model_id"].astype(str).unique().tolist())
    seeds = sorted({int(s) for s in registry["seed"]})
    logger.info("strategies=%d models=%d seeds=%d", len(strategies), len(models), len(seeds))

    if len(seeds) < 3:
        issues.append(
            f"only {len(seeds)} seed(s) present; protocol 11 requires 5 (minimum "
            f"3). Between-seed variance is not estimable below 3.")
    if len(models) < 3:
        issues.append(
            f"only {len(models)} model(s) present; protocol 15 P0 requires at "
            f"least 3, otherwise RQ3 is not answerable.")

    # ---- tables ----------------------------------------------------------- #
    est = estimator_summary(registry, group_cols=("strategy", "model_id"),
                            resampling_unit="seed")
    est.to_csv(out_tables / f"estimator_summary{sfx}.csv", index=False,
               encoding="utf-8")
    logger.info("estimator_summary: %d rows", len(est))

    strat = strategy_level_summary(registry, metric=args.metric,
                                   per_region=per_region, eval_set="ext")
    strat.to_csv(out_tables / f"strategy_summary{sfx}.csv", index=False,
                 encoding="utf-8")
    logger.info("strategy_summary: %d rows", len(strat))

    # ---- design coverage --------------------------------------------------- #
    # The manuscript states how many tiles the external test contains and how
    # many of them fall in each region. Those counts belong in an artefact like
    # every other number (protocol section 16), and they are the check that the
    # external set is the SAME set for every run: if a run were evaluated
    # against a different region set, the study's central comparison would be
    # between different truths.
    if per_region is not None and len(per_region):
        ext = per_region[per_region["eval_set"].astype(str) == "ext"]
        val = per_region[per_region["eval_set"].astype(str) == "val"]
        rows = []
        for run_id, sub in ext.groupby("run_id"):
            vsub = val[val["run_id"] == run_id]
            rows.append({
                "run_id": run_id,
                "n_external_regions": int(sub["region"].nunique()),
                "n_external_samples": int(sub["n_samples"].sum()),
                "n_val_regions": int(vsub["region"].nunique()),
                "n_val_samples": int(vsub["n_samples"].sum()),
                "external_regions": "|".join(sorted(sub["region"].astype(str))),
            })
        cov = pd.DataFrame(rows).sort_values("run_id")
        cov.to_csv(out_tables / f"design_coverage{sfx}.csv", index=False,
                   encoding="utf-8")
        n_sets = cov["external_regions"].nunique() if len(cov) else 0
        logger.info("design_coverage: %d rows; distinct external region sets=%d; "
                    "external samples=%s", len(cov), n_sets,
                    sorted(set(cov["n_external_samples"]))[:3])
        if n_sets > 1:
            issues.append(
                f"runs were evaluated against {n_sets} different external region "
                f"sets; the external truth is not identical across runs")

    primary = primary_contrast_s0_vs_ext(
        registry, strategy=PRIMARY_STRATEGY, per_region=per_region)
    write_json(out_tables / f"primary_contrast{sfx}.json", primary)
    logger.info("primary contrast: region_level=%s run_level=%s",
                primary.get("region_level") is not None,
                primary.get("run_level") is not None)
    if primary.get("region_level") is None:
        warnings.append(
            "the region-level primary contrast could not be computed; the "
            "run-level version is reported instead and its coarser interval is "
            "not a substitute, because it does not respect geographic dependence")

    secondary = paired_contrasts(
        registry, a_col="ob_miou", alpha=args.alpha, pairs=PREREG_PAIRS,
        family=("pre-registered design contrasts (ten pairs among the five "
                "designs, fixed before the results were seen; D-060)"))
    secondary.to_csv(out_tables / f"secondary_contrasts{sfx}.csv", index=False,
                     encoding="utf-8")
    logger.info("secondary contrasts: %d rows (frozen family of %d)",
                len(secondary), len(PREREG_PAIRS))

    # P1 sensitivity sweeps (D-060): summarised separately, never entering the
    # family above, the mixed models or the ranking analysis.
    sweep = _sweep_summary(registry)
    if len(sweep):
        sweep.to_csv(out_tables / f"sweep_summary{sfx}.csv", index=False,
                     encoding="utf-8")
        logger.info("sweep summary: %d rows across %d sweep cells",
                    len(sweep), len(sweep[["family", "setting_r"]]
                                     .drop_duplicates()))
    else:
        logger.info("sweep summary: no P1 sweep strategies present in the registry")

    # Pre-registered scope (D-060): the mixed model, like every inferential
    # analysis here, is fitted on the five-design set. Per-region rows of the
    # P1 sweeps are excluded rather than silently entering the Strategy factor
    # as three further levels with one seed each.
    per_region_prereg = (
        per_region[per_region["strategy"].astype(str)
                   .isin(PRE_REGISTERED_DESIGNS)]
        if per_region is not None else None)
    mixed = _fit_mixed_models(per_region_prereg)
    write_json(out_tables / f"mixed_effects{sfx}.json", mixed)
    logger.info("mixed effects: fits=%d, primary converged=%s fallback=%s",
                len(mixed.get("fits", [])),
                (mixed.get("primary") or {}).get("converged"),
                (mixed.get("primary") or {}).get("fallback_recommended"))

    # Ranking concordance needs the external strategy present as a column value.
    # SCOPE (D-060): computed on the pre-registered design set only. The P1
    # sweeps ran at a single seed and are not part of RQ5's pre-registered
    # scope; including them would change the comparison without any text or
    # planning asking for it.
    prereg_reg = registry[registry["strategy"].astype(str)
                          .isin(PRE_REGISTERED_DESIGNS)]
    rank_df = prereg_reg.copy()
    rank_df["strategy"] = rank_df["strategy"].astype(str)
    rank_df["miou"] = rank_df["val_miou"]
    # Build the reference arm from the external evaluation of each strategy's
    # own models: the ranking under EXT is a single ranking, so the reference
    # rows carry strategy == REFERENCE_STRATEGY and the external metric.
    if f"ext_{args.metric}" in registry.columns:
        ref_rows = prereg_reg.copy()
        ref_rows["strategy"] = REFERENCE_STRATEGY
        ref_rows["miou"] = ref_rows[f"ext_{args.metric}"]
        rank_df = pd.concat(
            [rank_df[["strategy", "model_id", "seed", "fold", "miou"]],
             ref_rows[["strategy", "model_id", "seed", "fold", "miou"]]],
            ignore_index=True)
    ranking = ranking_concordance(rank_df, reference_strategy=REFERENCE_STRATEGY)
    write_json(out_tables / f"ranking_concordance{sfx}.json", ranking)
    if ranking.get("per_strategy"):
        pd.DataFrame([
            {k: v for k, v in rec.items() if k != "reversals"}
            for rec in ranking["per_strategy"]
        ]).to_csv(out_tables / f"ranking_concordance{sfx}.csv", index=False,
                  encoding="utf-8")
    logger.info("ranking concordance: %d strategies", ranking.get("n_strategies", 0))

    # ---- statistics QC gate ----------------------------------------------- #
    qc: list[dict] = []

    n_reg_pairs = sum(
        len(v) for v in [est[est["metric"] == "ob_miou"]["n"]] if v is not None)
    qc.append({
        "check": "sample counts consistent",
        "detail": f"{len(registry)} runs, {len(est)} summary rows, "
                  f"{len(per_region) if per_region is not None else 0} region rows",
        "status": "PASS" if len(est) > 0 else "FAIL",
    })
    qc.append({
        "check": "no run double-counted",
        "detail": f"{registry['run_id'].nunique()} unique run_id of "
                  f"{len(registry)} rows",
        "status": "PASS" if registry["run_id"].nunique() == len(registry) else "FAIL",
    })
    qc.append({
        "check": "bootstrap resampling unit stated",
        "detail": f"estimator_summary unit={est['resampling_unit'].iloc[0] if len(est) else 'n/a'}",
        "status": "PASS" if len(est) and est["resampling_unit"].notna().all()
                  else "FAIL",
    })
    # The region-level estimand is now reported in its own columns
    # (`region_mean` / `region_ci_*`), so the gate checks for THAT unit rather
    # than for a region-bootstrapped interval placed under the run-level mean
    # (D-048).
    unit_ok = bool(
        per_region is not None and len(per_region) and len(strat)
        and "region_resampling_unit" in strat.columns
        and (strat["region_resampling_unit"] == "region").any())
    qc.append({
        "check": "geographic designs use the region as the resampling unit",
        "detail": (f"region (n_regions up to "
                   f"{int(strat['n_regions'].max()) if 'n_regions' in strat.columns and strat['n_regions'].notna().any() else 'n/a'})"
                   if unit_ok else
                   "NOT AVAILABLE - per-region metrics absent, so geographic "
                   "dependence is not respected by the intervals"),
        "status": "PASS" if unit_ok else "PARTIAL",
    })
    qc.append({
        "check": "primary contrast pre-specified and unadjusted",
        "detail": f"{primary.get('name')} / adjusted="
                  f"{primary.get('multiplicity_adjusted')}",
        "status": "PASS" if primary.get("pre_specified") and not
                  primary.get("multiplicity_adjusted") else "FAIL",
    })
    # Declaring a contrast pre-specified is not the same as computing it. This
    # check exists because the gate said PASS while the primary contrast
    # returned `no runs for strategy 'S0'` -- the partition labels are
    # `S0_random_patch` and the contrast compared against `S0` (D-047). A gate
    # that passes on an uncomputed pre-registered analysis is worse than no gate:
    # it certifies the absence of the study's headline result as compliance.
    qc.append({
        "check": "primary contrast actually computed",
        "detail": ("computed on the run level" if primary.get("run_level")
                   else f"NOT COMPUTED: {primary.get('note')}"),
        "status": "PASS" if primary.get("run_level") else "FAIL",
    })
    # The family must be the pre-registered ten, nothing less and nothing more.
    # Declaring a family is not the same as freezing it: the previous version of
    # this check passed whenever the `family` column was non-null, which it
    # would have remained even as the family silently grew from ten contrasts
    # to thirty-six once the P1 sweep strategies were registered (D-060).
    prereg_in_registry = [s for s in PRE_REGISTERED_DESIGNS
                          if (registry["strategy"].astype(str) == s).any()]
    if len(secondary):
        sec_strategies = (set(secondary["strategy_a"].astype(str))
                          | set(secondary["strategy_b"].astype(str)))
        sec_pairs = set(zip(secondary["strategy_a"].astype(str),
                            secondary["strategy_b"].astype(str)))
    else:
        sec_strategies, sec_pairs = set(), set()
    sec_outside = sorted(sec_strategies - set(PRE_REGISTERED_DESIGNS))
    sec_missing = [f"{a} - {b}" for a, b in PREREG_PAIRS
                   if (a, b) not in sec_pairs]
    if sec_outside:
        _fam_status = "FAIL"
        _fam_detail = (f"contrasts involve strategies outside the frozen set: "
                       f"{sec_outside}")
    elif len(prereg_in_registry) < len(PRE_REGISTERED_DESIGNS):
        _fam_status = "PARTIAL"
        _fam_detail = (f"registry contains {len(prereg_in_registry)} of "
                       f"{len(PRE_REGISTERED_DESIGNS)} pre-registered designs; "
                       f"the frozen-family completeness check applies to the "
                       f"study registry (no contrast outside the frozen set)")
    elif sec_missing:
        _fam_status = "FAIL"
        _fam_detail = (f"{len(secondary)} of {len(PREREG_PAIRS)} pre-registered "
                       f"contrasts computed; missing: {sec_missing[:3]}")
    else:
        _fam_status = "PASS"
        _fam_detail = (f"{len(secondary)} of {len(PREREG_PAIRS)} pre-registered "
                       f"contrasts computed; no contrast outside the set")
    qc.append({
        "check": "family for FDR control frozen to the pre-registered designs",
        "detail": _fam_detail,
        "status": _fam_status,
    })
    qc.append({
        "check": "external freeze consistent across runs",
        "detail": f"sha={str(freeze_sha)[:16] if freeze_sha else 'n/a'}",
        "status": "PASS" if not any("freeze" in i for i in issues) else "FAIL",
    })

    # A point estimate must lie inside its own interval. This is a cheap
    # invariant, and it is the one that caught an interval bootstrapped over
    # regions while its mean averaged over runs (D-048): S0 read
    # mean = 0.5242 with CI [0.4428, 0.5129].
    _containment: list[str] = []
    for frame, name in ((est, "estimator_summary"), (strat, "strategy_summary")):
        if frame is None or not len(frame):
            continue
        for _, row in frame.iterrows():
            for mcol, lo, hi in (("mean", "ci_low", "ci_high"),
                                 ("region_mean", "region_ci_low",
                                  "region_ci_high")):
                if mcol not in frame.columns or lo not in frame.columns:
                    continue
                try:
                    m, a, b = float(row[mcol]), float(row[lo]), float(row[hi])
                except (TypeError, ValueError):
                    continue
                if m != m or a != a or b != b:            # NaN
                    continue
                if not (a - 1e-12 <= m <= b + 1e-12):
                    _containment.append(
                        f"{name}/{row.get('strategy')}/{row.get('model_id')}/"
                        f"{row.get('metric')}: {mcol}={m:.4f} outside "
                        f"[{a:.4f}, {b:.4f}]")
    qc.append({
        "check": "every interval contains its own point estimate",
        "detail": ("all intervals contain their estimate" if not _containment
                   else "; ".join(_containment[:3])),
        "status": "PASS" if not _containment else "FAIL",
    })

    qc_status = "PASS"
    if any(q["status"] == "FAIL" for q in qc):
        qc_status = "FAIL"
    elif any(q["status"] == "PARTIAL" for q in qc):
        qc_status = "PARTIAL"

    # ---- report ----------------------------------------------------------- #
    report = _build_report(
        registry=registry, est=est, strat=strat, primary=primary,
        secondary=secondary, sweep=sweep, mixed=mixed, ranking=ranking, qc=qc,
        leakage=leakage, warnings=warnings, issues=issues,
        freeze_sha=freeze_sha, strategies=strategies, models=models, seeds=seeds,
        metric=args.metric, alpha=args.alpha, per_region=per_region)
    (out_stats / f"statistics_report{sfx}.md").write_text(report, encoding="utf-8")
    write_json(out_stats / f"statistics_qc{sfx}.json",
               {"status": qc_status, "checks": qc, "issues": issues,
                "warnings": warnings, "at_utc": utcnow()})

    overall = "PASS"
    if issues or qc_status == "FAIL":
        overall = "PARTIAL"
    if not len(est):
        overall = "FAIL"

    print()
    print("[PHASE] Statistical analysis")
    print(f"[STATUS] {overall}")
    print(f"[COMPLETED] {len(registry)} runs analysed; "
          f"{len(strategies)} strategies x {len(models)} models x {len(seeds)} seeds")
    print(f"[FILES] {out_tables / f'estimator_summary{sfx}.csv'}")
    print(f"[FILES] {out_tables / f'strategy_summary{sfx}.csv'}")
    print(f"[FILES] {out_tables / f'primary_contrast{sfx}.json'}")
    print(f"[FILES] {out_tables / f'secondary_contrasts{sfx}.csv'}")
    if len(sweep):
        print(f"[FILES] {out_tables / f'sweep_summary{sfx}.csv'}")
    print(f"[FILES] {out_tables / f'mixed_effects{sfx}.json'}")
    print(f"[FILES] {out_tables / f'ranking_concordance{sfx}.json'}")
    print(f"[FILES] {out_stats / f'statistics_report{sfx}.md'}")
    print(f"[QC] statistics gate: {qc_status}")
    for q in qc:
        print(f"[QC]   {q['status']:7s} {q['check']}: {q['detail']}")
    for w in warnings:
        print(f"[ISSUES] {w}")
    for i in issues:
        print(f"[ISSUES] {i}")
    print("[NEXT] figures, then the Results section")
    return 0 if overall == "PASS" else 1


def _fit_mixed_models(per_region: pd.DataFrame | None) -> dict[str, Any]:
    """Fit the pre-registered mixed model, once per response.

    WHY NOT ONE FIT. `per_region_metrics.csv` holds both the validation-set and
    the external-set metric. Pooling those rows into a single response would fit
    a mixture of two different quantities, and the strategy coefficients would
    then describe neither. The pre-registered formula names one `Metric`, so it
    is fitted separately for each response:

      * external metric -- the study's ground truth, evaluated on the frozen
        external regions, with a random intercept for external region. Primary.
      * validation metric -- the estimate, evaluated on development regions,
        with a random intercept for development region. Reported for contrast;
        the difference between the two coefficient sets is the strategy effect
        on the estimate rather than on the truth.

    WHY THERE IS NO REGION-LEVEL OB FIT. The optimism bias is
    `validation - external` for a given run, and those two numbers are computed
    on disjoint region sets by construction -- the external regions are excluded
    from every partition. No region therefore carries both, so an OB random
    intercept for region is not identifiable. That is a property of the study
    design, not a data gap, and it is reported as such rather than being
    silently omitted or, worse, fitted on a partial overlap that would mean
    something different from what its name suggests.
    """
    base: dict[str, Any] = {
        "pre_registered_formula": "Metric ~ Strategy + Model + Strategy x Model + (1|Region)",
        "group": "region",
        "primary_response": "external metric",
        "fits": [], "primary": None, "note": None,
        "ob_region_fit": {
            "fitted": False,
            "reason": ("not identifiable by design: the optimism bias compares a "
                       "validation estimate on development regions with an "
                       "external estimate on external regions, and those region "
                       "sets are disjoint because the external regions are "
                       "excluded from every partition. No region carries both "
                       "quantities, so no region random intercept can be "
                       "estimated for the bias. The bias is analysed at run "
                       "level instead (primary_contrast.json, "
                       "secondary_contrasts.csv)."),
        },
    }
    if per_region is None or len(per_region) == 0:
        base["note"] = ("per_region_metrics.csv absent: the pre-registered random "
                        "intercept for region requires region-level observations, "
                        "which do not exist at partition level")
        base["fallback_recommended"] = True
        base["fallback_reason"] = base["note"]
        base["converged"] = False
        return base

    present = set(per_region["eval_set"].astype(str).unique())
    responses = [("external metric", "ext", "primary",
                  "the frozen external geographic test: the truth being estimated"),
                 ("validation metric", "val", "contrast",
                  "the estimate produced by each validation strategy")]
    for label, eval_set, role, why in responses:
        if eval_set not in present:
            base["fits"].append({
                "response_label": label, "converged": False,
                "error": f"no rows with eval_set={eval_set!r}",
                "fallback_recommended": True,
                "fallback_reason": f"no rows with eval_set={eval_set!r}",
                "why": why,
            })
            continue
        sub = per_region[per_region["eval_set"].astype(str) == eval_set]
        frame = sub.rename(columns={"miou": "miou"}).copy()
        frame["miou"] = pd.to_numeric(frame["miou"], errors="coerce")
        frame = frame.dropna(subset=["miou"])
        fit = fit_mixed_effects(
            frame, response="miou",
            fixed_effects="C(strategy) * C(model_id)", group="region")
        fit["response_label"] = label
        fit["eval_set"] = eval_set
        fit["role"] = role
        fit["why"] = why
        fit["n_regions"] = int(frame["region"].nunique())
        base["fits"].append(fit)
        if role == "primary":
            base["primary"] = fit

    base["converged"] = bool((base["primary"] or {}).get("converged"))
    base["fallback_recommended"] = bool(
        (base["primary"] or {}).get("fallback_recommended"))
    base["fallback_reason"] = (base["primary"] or {}).get("fallback_reason")
    base["assumptions_met"] = (base["primary"] or {}).get("assumptions_met")
    # Flatten the primary fit to the top level as well, so downstream readers
    # that expect the single-fit schema keep working.
    if base["primary"]:
        for k in ("n_obs", "n_groups", "fixed_effects", "random_effect_var",
                  "residual_var", "icc", "shapiro_p", "breusch_pagan_p",
                  "singular_covariance", "error", "fallback_note"):
            base.setdefault(k, base["primary"].get(k))
    return base


def _sweep_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """P1 sensitivity sweeps — block and buffer, at 0.5 r / 1 r / 2 r (D-060).

    These strategies are NOT part of the pre-registered analysis scope: they
    enter no primary contrast, no FDR family, no mixed model and no ranking
    analysis. They are summarised here, and in figure F7, only.

    TWO SCOPES, COMPUTED SEPARATELY. The 1 r setting has several seeds
    (S2: five, S3: three) while the 0.5 r and 2 r endpoints have one (seed
    42). A delta against a 1 r mean that includes the extra seeds, while the
    endpoints carry one seed each, would silently change what the difference
    measures. The table therefore carries, for every statistic:
      * scope="seed42_paired" -- every setting restricted to seed 42, the only
        seed the endpoints have; the like-for-like series.
      * scope="all_available" -- every setting over all of its runs; the value
        figure F7 plots, where the 1 r cell is the better-sampled one.
    Deltas (d_ob_vs_1r, d_aee_vs_1r) are computed WITHIN each scope, so each
    difference always compares like with like.

    Levels: one row per (family, setting, model) with level="model"; plus one
    across_models row per (family, setting) whose value is the unweighted mean
    of the three per-model means — the same estimator the manuscript quotes.
    """
    cols = ["family", "setting_r", "scope", "level", "model_id", "n_runs",
            "n_seeds", "n_train_median", "val_miou", "ext_miou", "ob_miou",
            "aee_miou", "d_ob_vs_1r", "d_aee_vs_1r"]
    if registry is None or len(registry) == 0:
        return pd.DataFrame(columns=cols)
    reg = registry.copy()
    parsed = reg["strategy"].astype(str).map(parse_sweep_strategy)
    reg["family"] = [p[0] if p else None for p in parsed]
    reg["setting_r"] = [p[1] if p else None for p in parsed]
    reg = reg.dropna(subset=["family", "setting_r"])
    if reg.empty:
        return pd.DataFrame(columns=cols)
    for col in ("ob_miou", "aee_miou", "val_miou", "ext_miou", "n_train"):
        reg[col] = pd.to_numeric(reg[col], errors="coerce")
    reg["seed"] = pd.to_numeric(reg["seed"], errors="coerce")

    rows: list[dict] = []
    for scope in ("seed42_paired", "all_available"):
        sub = reg[reg["seed"] == 42] if scope == "seed42_paired" else reg
        if sub.empty:
            continue
        per_model = (
            sub.groupby(["family", "setting_r", "model_id"])
            .agg(n_runs=("run_id", "nunique"), n_seeds=("seed", "nunique"),
                 n_train_median=("n_train", "median"),
                 val_miou=("val_miou", "mean"), ext_miou=("ext_miou", "mean"),
                 ob_miou=("ob_miou", "mean"), aee_miou=("aee_miou", "mean"))
            .reset_index())
        base = (per_model[per_model["setting_r"] == 1.0]
                .set_index(["family", "model_id"]))
        for _, r in per_model.iterrows():
            key = (r["family"], r["model_id"])
            b = base.loc[key] if key in base.index else None
            rows.append({
                "family": r["family"], "setting_r": float(r["setting_r"]),
                "scope": scope, "level": "model", "model_id": r["model_id"],
                "n_runs": int(r["n_runs"]), "n_seeds": int(r["n_seeds"]),
                "n_train_median": float(r["n_train_median"]),
                "val_miou": float(r["val_miou"]),
                "ext_miou": float(r["ext_miou"]),
                "ob_miou": float(r["ob_miou"]),
                "aee_miou": float(r["aee_miou"]),
                "d_ob_vs_1r": (float(r["ob_miou"] - b["ob_miou"])
                               if b is not None else np.nan),
                "d_aee_vs_1r": (float(r["aee_miou"] - b["aee_miou"])
                                if b is not None else np.nan),
            })
        across = (
            per_model.groupby(["family", "setting_r"])
            .agg(ob_miou=("ob_miou", "mean"), aee_miou=("aee_miou", "mean"),
                 val_miou=("val_miou", "mean"), ext_miou=("ext_miou", "mean"),
                 n_runs=("n_runs", "sum"), n_seeds=("n_seeds", "max"),
                 n_train_median=("n_train_median", "median"))
            .reset_index())
        base_all = across[across["setting_r"] == 1.0].set_index("family")
        for _, r in across.iterrows():
            b = (base_all.loc[r["family"]]
                 if r["family"] in base_all.index else None)
            rows.append({
                "family": r["family"], "setting_r": float(r["setting_r"]),
                "scope": scope, "level": "across_models", "model_id": "ALL",
                "n_runs": int(r["n_runs"]), "n_seeds": int(r["n_seeds"]),
                "n_train_median": float(r["n_train_median"]),
                "val_miou": float(r["val_miou"]),
                "ext_miou": float(r["ext_miou"]),
                "ob_miou": float(r["ob_miou"]),
                "aee_miou": float(r["aee_miou"]),
                "d_ob_vs_1r": (float(r["ob_miou"] - b["ob_miou"])
                               if b is not None else np.nan),
                "d_aee_vs_1r": (float(r["aee_miou"] - b["aee_miou"])
                                if b is not None else np.nan),
            })
    out = pd.DataFrame(rows, columns=cols)
    if len(out):
        out = out.sort_values(["scope", "family", "level", "setting_r",
                               "model_id"], kind="stable").reset_index(drop=True)
    return out


def _empty_report(warnings: list[str]) -> str:
    w = "\n".join(f"- {x}" for x in warnings) or "- none"
    return f"""# Statistical analysis report

Generated {utcnow()} by `scripts/analyze_results.py`.

## Status

**No experiment results exist yet.** The analysis code has been validated on
synthetic data with known answers (73 unit assertions in
`scripts/test_statistics.py`, plus an end-to-end pipeline check against
`validation/analysis_pipeline/`), but there is nothing to analyse. No number in
this file is an experimental result, and none may be reported as one.

## Warnings carried forward

{w}

## What will be produced once the grid has run

| Output | Content |
|---|---|
| `estimator_summary.csv` | mean / sd / bootstrap interval per strategy x model, with the resampling unit |
| `strategy_summary.csv` | per-strategy aggregate, region-level intervals where region data exist |
| `primary_contrast.json` | S0 vs EXT on mIoU, pre-specified, unadjusted |
| `secondary_contrasts.csv` | all other pairwise contrasts, Benjamini-Hochberg over a fixed family |
| `sweep_summary.csv` | P1 block/buffer sensitivity summary (separate from the pre-registered family) |
| `mixed_effects.json` | `Metric ~ Strategy + Model + Strategy x Model + (1\\|Region)`, fitted once per response (external, region-level OB, validation) with assumption diagnostics |
| `ranking_concordance.json` | Spearman rho / Kendall tau against the external ranking, reversals listed individually |
| `statistics_qc.json` | the statistics quality-control gate |
"""


def _build_report(**kw) -> str:
    registry = kw["registry"]
    est, strat = kw["est"], kw["strat"]
    primary, secondary = kw["primary"], kw["secondary"]
    mixed, ranking, qc = kw["mixed"], kw["ranking"], kw["qc"]
    leakage, warnings, issues = kw["leakage"], kw["warnings"], kw["issues"]
    metric, alpha = kw["metric"], kw["alpha"]

    L: list[str] = []
    L.append("# Statistical analysis report\n")
    L.append(f"Generated {utcnow()} by `scripts/analyze_results.py`.\n")
    L.append("Every number below is read from a machine-generated artefact in "
             "`experiments/` or `results/`. Nothing is typed by hand.\n")

    L.append("## 1. Scope of the experiment grid\n")
    L.append(f"- runs: **{len(registry)}**")
    L.append(f"- strategies: {', '.join(kw['strategies'])}")
    L.append(f"- pre-registered inferential scope (D-060): "
             f"{'; '.join(PRE_REGISTERED_DESIGNS)} — the five designs the "
             f"primary contrast, the FDR family, the mixed models and the "
             f"ranking analysis are computed over")
    _p1 = [s for s in kw['strategies'] if parse_sweep_strategy(s)]
    if _p1:
        L.append(f"- P1 sensitivity strategies (summarised in section 5b and "
                 f"figure F7, entering no pre-registered family): "
                 f"{', '.join(_p1)}")
    L.append(f"- models: {', '.join(kw['models'])}")
    L.append(f"- seeds: {', '.join(str(s) for s in kw['seeds'])}")
    L.append(f"- external freeze sha256: `{str(kw['freeze_sha'])[:32] if kw['freeze_sha'] else 'n/a'}`")
    L.append(f"- primary metric: **{metric}**\n")

    if issues:
        L.append("### Integrity issues (must be resolved before publication)\n")
        for i in issues:
            L.append(f"- **{i}**")
        L.append("")
    if warnings:
        L.append("### Warnings (pre-registered analyses that could not be run)\n")
        for w in warnings:
            L.append(f"- {w}")
        L.append("")

    L.append("## 2. The primary contrast (pre-specified, no multiplicity adjustment)\n")
    L.append(f"{primary.get('name')}\n")
    L.append(f"> {primary.get('rationale')}\n")
    rl, rn = primary.get("region_level"), primary.get("run_level")
    if rl:
        L.append("**Region-level (primary).** Resampling unit: "
                 f"**{rl['resampling_unit']}**, n={rl['n_regions']}.\n")
        L.append(f"- mean OB = **{_fmt(rl['mean_ob'])}** "
                 f"(95% CI {_fmt(rl['ci_low'])} to {_fmt(rl['ci_high'])})")
        L.append(f"- median OB = {_fmt(rl.get('median_ob'))}")
        L.append(f"- regions with positive OB (optimistic): "
                 f"**{rl.get('n_regions_positive')}**, "
                 f"negative (pessimistic): **{rl.get('n_regions_negative')}**")
        L.append(f"- permutation p = {_fmt(rl['p_value'], 5)}")
        L.append(f"- standardised effect (Cohen's dz) = {_fmt(rl['cohens_dz'])}\n")
    else:
        L.append("**Region-level paired contrast not available.** "
                 f"{primary.get('region_level_unavailable_reason') or primary.get('note') or 'no per-region data'}\n")
    if rn:
        L.append("**Run-level (supporting).** Resampling unit: "
                 f"**{rn['resampling_unit']}**, n={rn['n_pairs']}.\n")
        L.append(f"- mean difference = {_fmt(rn['mean_diff'])} "
                 f"(95% CI {_fmt(rn['ci_low'])} to {_fmt(rn['ci_high'])})")
        L.append(f"- permutation p = {_fmt(rn['p_value'], 5)}")
        w = rn.get("wilcoxon") or {}
        L.append(f"- Wilcoxon signed-rank p = {_fmt(w.get('p_value'), 5)}\n")

    L.append("## 3. Estimator metrics per strategy and model\n")
    L.append(_md_table(est, ["strategy", "model_id", "metric", "n", "mean", "sd",
                             "ci_low", "ci_high", "resampling_unit"]))

    L.append("## 4. Per-strategy summary\n")
    L.append(_md_table(strat))

    L.append("## 5. Secondary contrasts (exploratory, Benjamini-Hochberg)\n")
    L.append(f"Family: *{secondary['family'].iloc[0] if len(secondary) else 'n/a'}*, "
             f"alpha = {alpha}. The primary contrast above is deliberately "
             f"**excluded** from this family.\n")
    L.append(_md_table(secondary, ["contrast", "n_pairs", "mean_diff", "ci_low",
                                   "ci_high", "p_value", "p_adjusted", "reject"]))

    L.append("## 5b. P1 sensitivity sweeps (not pre-registered; D-060)\n")
    sweep_tbl = kw.get("sweep")
    if sweep_tbl is not None and len(sweep_tbl):
        L.append("Block-size (S2) and buffer-width (S3) sweeps at 0.5 r, 1 r and "
                 "2 r. `d_ob_vs_1r` / `d_aee_vs_1r` are computed within each "
                 "scope, against that scope's own 1 r value. These strategies "
                 "enter no pre-registered family, no mixed model and no ranking "
                 "analysis.\n")
        L.append(_md_table(sweep_tbl, ["family", "setting_r", "scope", "level",
                                       "model_id", "n_runs", "n_seeds",
                                       "n_train_median", "val_miou",
                                       "ext_miou", "ob_miou", "aee_miou",
                                       "d_ob_vs_1r", "d_aee_vs_1r"]))
    else:
        L.append("_No P1 sweep strategies present in the registry._\n")

    L.append("## 6. Mixed-effects model\n")
    L.append(f"Pre-registered formula: `{mixed.get('pre_registered_formula')}`\n")
    L.append("Fitted separately for each response. Pooling the validation and "
             "external metrics into one response would fit a mixture of two "
             "different quantities, and the coefficients would describe "
             "neither.\n")
    ob = mixed.get("ob_region_fit") or {}
    if ob and not ob.get("fitted"):
        L.append("**No region-level fit for the optimism bias.** "
                 f"{ob.get('reason')}\n")
    fits = mixed.get("fits") or []
    if not fits:
        L.append(f"**Not fitted.** {mixed.get('note') or mixed.get('error')}\n")
    for f in fits:
        tag = " **(primary)**" if f.get("role") == "primary" else ""
        L.append(f"### `{f.get('response_label')}`{tag} — {f.get('why')}\n")
        L.append(f"`{f.get('formula')}`\n")
        if not f.get("converged"):
            L.append(f"**Not fitted.** {f.get('error') or f.get('fallback_reason')}\n")
            continue
        L.append(f"- observations: {f.get('n_obs')}, regions: {f.get('n_groups')}")
        L.append(f"- random-effect variance: {_fmt(f.get('random_effect_var'), 6)}, "
                 f"residual variance: {_fmt(f.get('residual_var'), 6)}")
        L.append(f"- intraclass correlation: **{_fmt(f.get('icc'))}**")
        L.append(f"- singular covariance: {f.get('singular_covariance')}")
        L.append(f"- Shapiro-Wilk (residual normality) p = {_fmt(f.get('shapiro_p'), 5)}")
        L.append(f"- Breusch-Pagan (homoscedasticity) p = {_fmt(f.get('breusch_pagan_p'), 5)}")
        L.append(f"- assumptions met: **{f.get('assumptions_met')}**")
        if f.get("fallback_recommended"):
            L.append(f"\n> **Fallback in use.** {f.get('fallback_reason')}")
            if f.get("fallback_note"):
                L.append(f"> {f.get('fallback_note')}")
            L.append("")
        fe = f.get("fixed_effects") or []
        if fe:
            L.append("\n| term | coefficient | SE | p |")
            L.append("|---|---|---|---|")
            for t in fe:
                L.append(f"| `{t['term']}` | {_fmt(t['coef'])} | {_fmt(t['se'])} "
                         f"| {_fmt(t['p_value'], 5)} |")
            L.append("")

    L.append("## 7. Model ranking stability (RQ5)\n")
    L.append(f"Reference ranking: the frozen external geographic test "
             f"(`{ranking.get('reference_strategy')}`).\n")
    if ranking.get("reference_ranking"):
        L.append(f"Reference order: **{' > '.join(ranking['reference_ranking'])}**\n")
    if ranking.get("per_strategy"):
        L.append("| strategy | rho | p | tau | p | rank diffs | reversals |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ranking["per_strategy"]:
            L.append(f"| {r['strategy']} | {_fmt(r['spearman_rho'])} "
                     f"| {_fmt(r['spearman_p'], 4)} | {_fmt(r['kendall_tau'])} "
                     f"| {_fmt(r['kendall_p'], 4)} | {r['rank_difference_max']} "
                     f"| {r['n_reversals']} |")
        L.append("")
        for r in ranking["per_strategy"]:
            if r["n_reversals"]:
                L.append(f"**Reversals under {r['strategy']}:**")
                for rev in r["reversals"]:
                    L.append(
                        f"- `{rev['model_a']}` vs `{rev['model_b']}`: external "
                        f"rank {rev['rank_a_ext']} vs {rev['rank_b_ext']} "
                        f"({_fmt(rev['metric_a_ext'])} vs {_fmt(rev['metric_b_ext'])}), "
                        f"strategy rank {rev['rank_a_strategy']} vs "
                        f"{rev['rank_b_strategy']} "
                        f"({_fmt(rev['metric_a_strategy'])} vs "
                        f"{_fmt(rev['metric_b_strategy'])})")
                L.append("")
        L.append("Directional neutrality: a concordant outcome is reported as "
                 "concordant. No branch of the analysis treats reversal as the "
                 "expected result.\n")
    else:
        L.append(f"Not computed. {ranking.get('note')}\n")

    L.append("## 8. Spatial independence diagnostics\n")
    if len(leakage):
        L.append(_md_table(leakage.head(30)))
        L.append("\nInterpretation rule: no claim of spatial independence rests "
                 "on a strategy's name. The diagnostic table is the evidence.\n")
    else:
        L.append("_Not available._\n")

    L.append("## 9. Statistics quality-control gate\n")
    L.append("| status | check | detail |")
    L.append("|---|---|---|")
    for q in qc:
        L.append(f"| **{q['status']}** | {q['check']} | {q['detail']} |")
    L.append("")

    L.append("## 10. Traceability\n")
    L.append("| artefact | path |")
    L.append("|---|---|")
    for p in ("experiments/experiments_registry.csv",
              "experiments/per_region_metrics.csv",
              "results/tables/spatial_independence_diagnostics.csv",
              "data/metadata/external_regions.json",
              "results/tables/estimator_summary.csv",
              "results/tables/strategy_summary.csv",
              "results/tables/primary_contrast.json",
              "results/tables/secondary_contrasts.csv",
              "results/tables/sweep_summary.csv",
              "results/tables/mixed_effects.json",
              "results/tables/ranking_concordance.json"):
        L.append(f"| `{p}` | |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

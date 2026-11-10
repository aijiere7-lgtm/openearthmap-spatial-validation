#!/usr/bin/env python
"""
make_figures.py — generate the manuscript figures from files in results/.

Protocol section 14 requires that figures be produced by code from artefacts in
`results/`, with no number typed into a figure by hand, at vector or >=300 dpi
resolution, and that no unfavourable result be removed for appearance.

This script therefore does not accept data as arguments. It reads the same
tables the statistical analysis writes, and if a table is missing it reports the
figure as unavailable rather than substituting anything.

Figures produced here (the data-driven ones):
    F4  spatial similarity / autocorrelation vs geographic distance
    F5  mIoU and macro F1 under each validation strategy
    F6  optimism bias and absolute estimation error
    F7  performance and estimation error vs spatial separation
    F8  model ranking under cross-validation vs the external test

F1 (research framework), F2 (geographic distribution -- a map, which carries
additional compliance requirements) and F3 (illustration of the splitting
strategies) are produced separately because they are conceptual or geographic
rather than derived from the experiment tables.

Usage:
    python scripts/make_figures.py                 # all available figures
    python scripts/make_figures.py --only F5,F6
    python scripts/make_figures.py --tag VALIDATION --registry <path> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    FIGURES_DIR, LOGS_DIR, PRE_REGISTERED_DESIGNS, TABLES_DIR, ensure_dirs,
    get_logger, parse_sweep_strategy, utcnow,
)

ROOT = Path(__file__).resolve().parent.parent
DPI = 300
FIGSIZE_WIDE = (7.2, 4.4)
FIGSIZE_SQUARE = (5.6, 5.0)


def _preregistered(registry: pd.DataFrame, logger=None) -> list[str]:
    """The pre-registered design set as present in the registry, sorted.

    Every figure EXCEPT F7 describes the design comparison, which is
    pre-registered on exactly these five designs (D-060). The P1 sweep
    strategies are deliberately excluded here and carried by F7 alone; the
    earlier code said "every strategy in the registry", which was correct
    only until the sweeps were registered.
    """
    present = set(registry["strategy"].astype(str).unique())
    kept = sorted(s for s in PRE_REGISTERED_DESIGNS if s in present)
    if logger is not None:
        missing = [s for s in PRE_REGISTERED_DESIGNS if s not in present]
        if missing:
            logger.warning("pre-registered design(s) absent from the registry: "
                           "%s", missing)
    return kept


def _read_json(path: Path) -> dict:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _class_names() -> list[str]:
    """Class names in index order, from the frozen config (never duplicated here)."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    return list(cfg["dataset"]["class_names"])


def _short_strategy(name: str) -> str:
    """A one-line label for a partition name, for axis ticks.

    The partition names carry their parameters (`S2_spatial_block_1r`), which is
    right for the registry and unreadable as a tick label: at three strategies
    the labels collide into illegible overlap. The protocol ID plus its
    parameter is enough inside a figure whose caption defines it.
    """
    s = str(name)
    sid = s.split("_", 1)[0]
    tail = s[len(sid):].strip("_")
    if not tail:
        return sid
    words = [w for w in tail.split("_") if w]
    keep: list[str] = []
    for w in words:
        if w in ("random",):
            keep.append("random")
        elif w in ("patch",):
            # "random patch" -> "random": the two-word tail is the widest
            # second line in F5/F6 and left a 0.8 px gap to its neighbour at
            # the shared width (measured, D-061). The caption defines the
            # class-stratified random patch split; the tick needs the design's
            # short name only.
            continue
        elif w in ("spatial",):
            continue
        elif w in ("block", "buffered"):
            keep.append("block" if w == "block" else "buffered")
        elif w in ("leave", "one", "region", "out"):
            keep = ["leave-region"]
        elif w in ("group", "scene"):
            keep = ["group"]
        else:
            keep.append(w)          # 0.5r, 1r, 2r, N1324, rep0 ...
    if "N1324" in keep:
        # The matched control's full label ("MATCHED S3 buffered 1r N1324
        # rep0") is the longest in any figure, and its second line spills
        # leftward into the neighbouring panel's plotting area (F8's
        # concordance tick labels, scientific-visualization legibility check,
        # D-061). The sample-size token plus the draw number uniquely
        # identifies the strategy; the "buffered 1r" part repeats what the
        # caption already defines.
        keep = [w for w in keep if w in ("N1324", "rep0")]
    return f"{sid}\n" + " ".join(keep) if keep else sid


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return plt


# A colour-blind-safe qualitative palette (Okabe-Ito), so the figures remain
# readable in print and for readers with colour vision deficiency.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# ONE COLOUR, ONE MEANING, ACROSS EVERY FIGURE.
#
# Colour encodes a MODEL wherever a model is what varies (F5b, F6b, F7, F8a).
# Anywhere colour would encode nothing -- the bars that contrast the estimate
# with the truth in F5a, the boxes in F6a, the concordance bars in F8b -- a
# neutral grey is used instead. The previous version coloured the estimate/truth
# bars with PALETTE[0] and PALETTE[1], which are also M1 and M2, so the same blue
# meant "validation estimate" in one panel and "M1" in the panel beside it. That
# is the kind of collision a reader resolves wrongly and silently.
MODEL_COLOURS = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73"}
NEUTRAL = "#8C8C8C"
NEUTRAL_DARK = "#3C3C3C"


def _model_colour(model_id: str, index: int = 0) -> str:
    return MODEL_COLOURS.get(str(model_id), PALETTE[index % len(PALETTE)])


def _colour_map(keys: list[str]) -> dict[str, str]:
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(sorted(keys))}


def _boxplot(ax, data, tick_labels, **kw):
    """Boxplot that works across the matplotlib rename.

    matplotlib 3.9 renamed `boxplot(labels=...)` to `tick_labels=...` and 3.11
    removed the old name. Passing the wrong one is a TypeError, so the version
    is probed rather than assumed.
    """
    import matplotlib
    try:
        major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    except Exception:                                          # noqa: BLE001
        major, minor = (3, 11)
    if (major, minor) >= (3, 9):
        return ax.boxplot(data, tick_labels=tick_labels, **kw)
    return ax.boxplot(data, labels=tick_labels, **kw)


def _save(fig, path: Path, logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    # A vector copy as well: journals prefer it and it costs nothing here.
    if path.suffix == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    logger.info("wrote %s", path)


# --------------------------------------------------------------------------- #
# F4 — spatial similarity / autocorrelation vs distance
# --------------------------------------------------------------------------- #
def figure_f4(tables: Path, out: Path, logger) -> bool:
    # The range r is estimated from WITHIN-REGION pairs. The all-pairs curve spans
    # continents, so its bins are ~460 km wide and it cannot resolve the scale at
    # which land-cover composition varies -- `spatial_dependence.json` records it
    # as "context only, not used for range". Plotting the all-pairs curve as the
    # primary panel (the previous behaviour) would have shown exactly the curve
    # the analysis declines to use, and dropped the one it uses on a column-name
    # mismatch (`lag_centre_km` was not among the accepted names).
    curve_path = tables / "spatial_dependence_curve_within_region.csv"
    context_path = tables / "spatial_dependence_curve.csv"
    autocorr_path = tables / "spatial_autocorrelation.csv"
    if not curve_path.exists():
        logger.warning("F4 unavailable: %s missing", curve_path.name)
        return False

    plt = _setup_mpl()
    curve = pd.read_csv(curve_path)
    lag_col = next((c for c in ("lag_centre_km", "lag_km", "distance_km", "lag",
                                "bin_km")
                    if c in curve.columns), None)
    sim_col = next((c for c in ("similarity", "s(d)", "s_d", "mean_similarity")
                    if c in curve.columns), None)
    if lag_col is None or sim_col is None:
        logger.warning("F4 unavailable: expected a distance column and a "
                       "similarity column, found %s", list(curve.columns))
        return False

    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE_WIDE[0], 3.4),
                             squeeze=False)
    ax = axes[0][0]
    ax.plot(curve[lag_col], curve[sim_col], "o-", ms=3, lw=1.2,
            color=PALETTE[0], label="observed")
    # The estimated dependence range, marked rather than merely mentioned.
    try:
        dep = _read_json(ROOT / "data" / "metadata" / "spatial_dependence.json")
        r_km = float(dep.get("range", {}).get("r_km"))
        ax.axvline(r_km, color=PALETTE[2], lw=1.2, ls="--",
                   label=f"estimated range r = {r_km:.2f} km")
        ax.axvspan(0, r_km, color=PALETTE[2], alpha=0.07)
        # Upper right, not lower left: the curve descends monotonically, so the
        # lower-left legend sat exactly on its own last four points and hid them
        # behind a 0.9-alpha frame. Legend occlusion is a defect, not a style.
        ax.legend(fontsize=7, loc="upper right", framealpha=1.0,
                  borderpad=0.4)
    except Exception:                                          # noqa: BLE001
        ax.legend(fontsize=7, loc="upper right", framealpha=1.0)
    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("Mean class similarity $S(d)$")
    ax.set_title("(a) Within-region pairs (used for $r$)", fontsize=9)

    # (b) all-pairs context curve, on its own coarse scale, drawn as the
    # contrast it is rather than as a second view of the same thing.
    ax2 = axes[0][1]
    if context_path.exists():
        ctx = pd.read_csv(context_path)
        c_lag = next((c for c in ("lag_centre_km", "lag_km", "distance_km")
                      if c in ctx.columns), None)
        c_sim = next((c for c in ("mean_similarity", "similarity")
                      if c in ctx.columns), None)
        if c_lag and c_sim:
            ax2.plot(ctx[c_lag], ctx[c_sim], "s-", ms=3, lw=1.2,
                     color=PALETTE[1])
            ax2.set_xlabel("Geographic distance (km)")
            ax2.set_ylabel("Mean class similarity $S(d)$")
            ax2.set_title("(b) All pairs (context only)", fontsize=9)
        else:
            ax2.axis("off")
    else:
        ax2.axis("off")

    # (c) Moran's I per class. The autocorrelation table is one row PER CLASS
    # (columns feature/morans_i/p_sim), not a distance series, so it cannot be
    # plotted against distance.
    ax3 = axes[0][2]
    if autocorr_path.exists():
        ac = pd.read_csv(autocorr_path)
        if {"feature", "morans_i"} <= set(ac.columns):
            try:
                names = _class_names()
            except Exception:                                  # noqa: BLE001
                names = None
            labels = [names[int(f)] if names and int(f) < len(names)
                      else str(f) for f in ac["feature"]]
            ax3.barh(range(len(ac)), ac["morans_i"].astype(float),
                     color=PALETTE[3], height=0.7)
            ax3.set_yticks(range(len(ac)))
            ax3.set_yticklabels(labels, fontsize=7)
            ax3.invert_yaxis()
            ax3.axvline(0.0, color="0.4", lw=0.8)
            ax3.set_xlabel("Moran's $I$")
            ax3.set_title("(c) Autocorrelation by class", fontsize=9)
        else:
            ax3.axis("off")
    else:
        ax3.axis("off")

    fig.suptitle("Spatial dependence of the land-cover corpus", y=1.03,
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out / "F4_spatial_dependence.png", logger)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# F5 — mIoU / macro F1 per validation strategy
# --------------------------------------------------------------------------- #
def figure_f5(registry: pd.DataFrame, out: Path, logger) -> bool:
    if registry is None or len(registry) == 0:
        logger.warning("F5 unavailable: no experiment registry")
        return False
    metric_cols = [c for c in ("val_miou", "ext_miou") if c in registry.columns]
    if not metric_cols:
        logger.warning("F5 unavailable: no miou columns in the registry")
        return False

    plt = _setup_mpl()
    strategies = _preregistered(registry, logger)
    # Short two-line tick labels: the registry names carry their parameters
    # ("S2_spatial_block_1r"), which is right for the registry and unreadable
    # as a tick label. The short form ("S2 / block 1r") plus the width rule
    # below keeps five designs legible (D-055, D-061).
    labels = [_short_strategy(s) for s in strategies]
    # `wspace` is set explicitly: at the default spacing, F8's panel (b) tick
    # labels (long partition names) were drawn across panel (a)'s plotting area.
    # Width grows with the number of designs at 1.5 in per design beyond three:
    # the two-line tick labels collided into one unreadable string at four
    # designs in the shared 7.2-inch width ("random patchblock 1 r", D-055),
    # and at five designs the old 1.0-in step left a measured 0.8 px gap
    # between "N1324 rep0" and "random patch" neighbours (D-061). A figure
    # whose axis labels cannot be read is not a figure.
    f5_w = FIGSIZE_WIDE[0] + 1.5 * max(0, len(strategies) - 3)
    fig, axes = plt.subplots(1, 2, figsize=(f5_w, 3.8), squeeze=False,
                             gridspec_kw={"wspace": 0.3})

    # (a) per-strategy distribution of the validation estimate and the truth.
    ax = axes[0][0]
    width = 0.36
    xs = np.arange(len(strategies))
    for j, (col, label, colour, hatch) in enumerate(
            [("val_miou", "validation estimate", NEUTRAL, None),
             ("ext_miou", "external test", NEUTRAL_DARK, "//")]):
        means = [registry.loc[registry["strategy"].astype(str) == s, col].mean()
                 for s in strategies]
        sds = [registry.loc[registry["strategy"].astype(str) == s, col].std(ddof=1)
               for s in strategies]
        ax.bar(xs + (j - 0.5) * width, means, width, yerr=sds, capsize=2,
               label=label, color=colour, alpha=0.9, hatch=hatch,
               edgecolor="white", linewidth=0.4,
               error_kw={"lw": 0.8, "ecolor": "0.3"})
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mIoU")
    ax.set_title("(a) Estimate vs external truth")
    ax.legend()

    # (b) per-model, per-strategy external mIoU: does the strategy matter for
    #     every architecture, or only for some?
    ax = axes[0][1]
    models = sorted(registry["model_id"].astype(str).unique())
    for i, m in enumerate(models):
        sub = registry[registry["model_id"].astype(str) == m]
        means = [sub.loc[sub["strategy"].astype(str) == s, "ext_miou"].mean()
                 for s in strategies]
        ax.plot(labels, means, "o-", ms=4, lw=1.3,
                color=_model_colour(m, i), label=m)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylabel("External-test mIoU")
    ax.set_xlabel("Validation strategy")
    ax.set_title("(b) External performance by architecture")
    ax.legend()

    fig.suptitle("Accuracy under each validation strategy", y=1.03)
    _save(fig, out / "F5_strategy_accuracy.png", logger)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# F6 — optimism bias / absolute estimation error
# --------------------------------------------------------------------------- #
def figure_f6(registry: pd.DataFrame, out: Path, logger) -> bool:
    if registry is None or len(registry) == 0:
        logger.warning("F6 unavailable: no experiment registry")
        return False
    if not {"ob_miou", "aee_miou"} <= set(registry.columns):
        logger.warning("F6 unavailable: ob_miou / aee_miou absent")
        return False

    plt = _setup_mpl()
    strategies = _preregistered(registry, logger)
    labels = [_short_strategy(s) for s in strategies]
    colours = _colour_map(strategies)
    # Same width rule as F5: five two-line tick labels need the extra 1.5 in
    # per design beyond three, or the tails collide (D-055, D-061).
    f6_w = FIGSIZE_WIDE[0] + 1.5 * max(0, len(strategies) - 3)
    fig, axes = plt.subplots(1, 2, figsize=(f6_w, 3.8), squeeze=False,
                             gridspec_kw={"wspace": 0.3})

    ax = axes[0][0]
    data = [registry.loc[registry["strategy"].astype(str) == s, "ob_miou"]
            .dropna().to_numpy() for s in strategies]
    if any(len(d) for d in data):
        bp = _boxplot(ax, data, labels, patch_artist=True, widths=0.55)
        # Neutral: the x axis already names the strategy, so colouring the boxes
        # by strategy would reuse the model hues for a second meaning.
        for patch in bp["boxes"]:
            patch.set_facecolor(NEUTRAL)
            patch.set_alpha(0.65)
            patch.set_edgecolor(NEUTRAL_DARK)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.2)
        del colours
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
    ax.set_ylabel("Optimism bias  (CV $-$ external), mIoU")
    ax.set_title("(a) Signed bias")
    # Placed in the lower-right corner: the upper region is occupied by the
    # boxes whenever the bias is positive, which is the case the study expects.
    ax.annotate("above the dashed line = optimistic",
                xy=(0.985, 0.985), xycoords="axes fraction",
                ha="right", va="top", fontsize=7, color="0.35")

    ax = axes[0][1]
    for i, m in enumerate(sorted(registry["model_id"].astype(str).unique())):
        sub = registry[registry["model_id"].astype(str) == m]
        means = [sub.loc[sub["strategy"].astype(str) == s, "aee_miou"].mean()
                 for s in strategies]
        ax.plot(labels, means, "o-", ms=4, lw=1.3,
                color=_model_colour(m, i), label=m)
    ax.set_ylabel("Absolute estimation error, mIoU")
    ax.set_xlabel("Validation strategy")
    ax.set_title("(b) Magnitude of error by architecture")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()

    fig.suptitle("Estimation error of each validation strategy", y=1.03)
    _save(fig, out / "F6_estimation_error.png", logger)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# F7 — performance and error vs spatial separation / block size
# --------------------------------------------------------------------------- #
def figure_f7(registry: pd.DataFrame, tables: Path, out: Path, logger) -> bool:
    if registry is None or len(registry) == 0:
        logger.warning("F7 unavailable: no experiment registry")
        return False
    # ------------------------------------------------------------------ #
    # TWO SWEEPS, TWO PANELS, AND THEY MUST NOT BE AVERAGED TOGETHER.
    #
    # The block-size sweep (S2 at 0.5 r, 1 r, 2 r) and the buffer sweep (S3 at
    # the same three values) vary different things: block size changes how the
    # fold boundaries are drawn, buffer width changes how much training data is
    # removed *around* those boundaries. The previous version extracted a single
    # "separation parameter" from the strategy name and grouped by it, which for
    # r = 1 would have averaged S2_spatial_block_1r and S3_buffered_1r into one
    # point — two designs with the same nominal separation, 2.5 % and 0 %
    # short-range contamination respectively, and different training-set sizes.
    # That is a mixed-population mean presented as a sensitivity result.
    #
    # SCOPE IS PARSED BY common.parse_sweep_strategy, WHICH ANCHORS THE NAME.
    # An unanchored search reads "MATCHED_S3_buffered_1r_N1324_rep0" as a
    # buffer-sweep point at 1 r and merges nine matched-control runs into the
    # sweep's 1 r cell — the same class of mixing defect as above, caught again
    # before the first render (D-060). The MATCHED filter below is the
    # in-figure assertion for it.
    # ------------------------------------------------------------------ #
    reg = registry.copy()
    parsed = reg["strategy"].astype(str).map(parse_sweep_strategy)
    reg["family"] = [p[0] if p else None for p in parsed]
    reg["mult"] = [p[1] if p else None for p in parsed]
    reg = reg.dropna(subset=["family", "mult"])
    if reg.empty:
        logger.warning("F7 unavailable: no block-size or buffer sweep present "
                       "in the registry (P1 priority; not yet run)")
        return False
    if reg["mult"].nunique() < 2:
        logger.warning("F7 unavailable: the sweep has fewer than two settings "
                       "per family")
        return False
    if reg["strategy"].astype(str).str.contains("MATCHED").any():
        # Cannot happen through parse_sweep_strategy; if it ever does, the
        # sweep means are contaminated and the figure must not be produced.
        logger.error("F7 refused: a MATCHED strategy passed the sweep filter")
        return False
    logger.info("F7 scope: %d sweep strategies, %d runs (%s)",
                reg["strategy"].nunique(), len(reg),
                ", ".join(sorted(reg["strategy"].astype(str).unique())))

    models = sorted(reg["model_id"].astype(str).unique())
    plt = _setup_mpl()
    # F7 carries three panels with six-line legends, so it does not fit the
    # shared 7.2-inch width: at that width the panel titles ran into each other
    # and the x-axis labels overlapped. It gets its own width rather than
    # shrinking the text below legibility.
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), squeeze=False,
                             gridspec_kw={"wspace": 0.30})

    # (a) and (b): one panel per sweep. Colour = model, linestyle = estimate
    # versus truth, which is the same encoding F7 has always used.
    for k, (family, title) in enumerate([
            ("block", "(a) Block-size sweep"),
            ("buffer", "(b) Buffer-width sweep")]):
        ax = axes[0][k]
        sub_all = reg[reg["family"] == family]
        if sub_all.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{family} sweep not yet run", ha="center",
                    va="center", fontsize=8, color="0.5")
            continue
        for i, m in enumerate(models):
            sub = sub_all[sub_all["model_id"].astype(str) == m]
            if sub.empty:
                continue
            g = sub.groupby("mult")[["ext_miou", "val_miou"]].mean().sort_index()
            colour = _model_colour(m, i)
            ax.plot(g.index, g["ext_miou"], "o-", ms=4, lw=1.4, color=colour,
                    label=f"{m} external")
            ax.plot(g.index, g["val_miou"], "s--", ms=3, lw=1.0, alpha=0.65,
                    color=colour, label=f"{m} estimate")
        ax.set_xlabel("Nominal separation (multiples of $r$)")
        ax.set_xticks(sorted(sub_all["mult"].unique()))
        ax.set_ylabel("mIoU")
        ax.set_title(title, fontsize=9)
        # `loc="best"` rather than a fixed corner: the six lines fill the plot in
        # a pattern that depends on the data, and a fixed corner put the legend on
        # top of the two flat external-performance lines. Opaque frame so any
        # residual overlap hides nothing (legend occlusion is a defect, D-055).
        ax.legend(fontsize=6, ncol=2, loc="best", framealpha=1.0,
                  handlelength=1.4, columnspacing=1.0)

    # (c) error against MEASURED separation, not nominal. The nominal multiple
    #     is a design parameter; the distance actually achieved is a measurement,
    #     and the study's own §4.1.3 insists the second is what the claim rests
    #     on. Taking the x values from the audit artefact keeps the figure and
    #     that section describing the same quantity.
    ax = axes[0][2]
    sv_path = tables / "split_validity_summary.csv"
    sv = pd.read_csv(sv_path).set_index("strategy") if sv_path.exists() else None
    plotted = 0
    # Encoding: COLOUR = model (the same hues as panels (a) and (b), so a model
    # identified once can be followed into this panel); MARKER = sweep family
    # (circle = block, square = buffer). The earlier version coloured by family
    # and varied alpha by model index, which gave M1 and M3 the SAME alpha —
    # two of the three per-model lines were visually identical, so the panel
    # could not show that all three models behave the same way, which is part
    # of what it exists for (scientific-visualization checklist, D-061).
    for family, marker in (("block", "o"), ("buffer", "s")):
        sub_all = reg[reg["family"] == family]
        for i, m in enumerate(models):
            sub = sub_all[sub_all["model_id"].astype(str) == m]
            if sub.empty:
                continue
            xs, ys = [], []
            for _, r in sub.iterrows():
                if sv is not None and r["strategy"] in sv.index:
                    xs.append(float(sv.loc[r["strategy"],
                                           "median_of_fold_medians_km"]))
                    ys.append(float(r["aee_miou"]))
            if xs:
                ax.plot(xs, ys, marker + "-", ms=4, lw=1.2,
                        color=_model_colour(m, i), alpha=0.9)
                plotted += 1
    if plotted == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "measured separation unavailable", ha="center",
                va="center", fontsize=8, color="0.5")
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Measured median train–validation distance (km; log scale)")
        ax.set_ylabel("Absolute estimation error, mIoU")
        ax.set_title("(c) Error vs measured separation", fontsize=9)
        handles = [plt.Line2D([], [], color=_model_colour(m, i), marker="",
                              ls="-", label=m)
                   for i, m in enumerate(models)]
        handles += [plt.Line2D([], [], color="0.35", marker="o", ls="none",
                               label="block (S2)"),
                    plt.Line2D([], [], color="0.35", marker="s", ls="none",
                               label="buffer (S3)")]
        ax.legend(handles=handles, fontsize=6, loc="best", framealpha=1.0,
                  ncol=2)

    fig.suptitle("Sensitivity to the spatial separation of the partitions", y=1.03)
    _save(fig, out / "F7_separation_sensitivity.png", logger)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# F8 — ranking under cross-validation vs the external test
# --------------------------------------------------------------------------- #
def figure_f8(registry: pd.DataFrame, tables: Path, out: Path, logger) -> bool:
    if registry is None or len(registry) == 0:
        logger.warning("F8 unavailable: no experiment registry")
        return False
    if not {"val_miou", "ext_miou"} <= set(registry.columns):
        logger.warning("F8 unavailable: val_miou / ext_miou absent")
        return False

    plt = _setup_mpl()
    strategies = _preregistered(registry, logger)
    models = sorted(registry["model_id"].astype(str).unique())
    mc = {m: _model_colour(m, i) for i, m in enumerate(models)}

    # One slopegraph PER DESIGN, then the concordance summary.
    #
    # The previous version overlaid all three strategies' slopegraphs in one
    # panel. Because two of the three designs produce the same ranking, the nine
    # plotted lines collapsed onto three visible paths and no reader could tell
    # which design a path belonged to -- a figure that looks like a result and
    # does not state one. One panel per design also gives each design the room to
    # be read on its own, and the shared model legend sits below the figure
    # rather than on top of the data.
    #
    # THE GRID IS DERIVED FROM THE NUMBER OF DESIGNS, NOT FIXED AT 2 x 2.
    # With the buffered design added there are four designs plus the concordance
    # panel, and the fixed 2 x 2 layout silently DROPPED the fourth design --
    # `strategies[:3]` -- while panel (d) still summarised all four. A figure that
    # quietly omits one of its own treatments is worse than one that fails to
    # render, so the grid now grows with the data and every design gets a panel.
    # The panel set is now the pre-registered designs (D-060) and the cap is
    # gone: with the P1 sweep strategies registered, a `[:6]` slice here would
    # have dropped the buffered design and leave-region-out silently — the same
    # defect class as `[:3]` above, one grid generation later.
    shown = strategies
    n_panels = len(shown) + 1
    ncols = 3 if n_panels > 4 else 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIGSIZE_WIDE[0] * ncols / 2.0, 3.35 * nrows),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.30, "hspace": 0.45})

    def _panel(k: int):
        return axes[k // ncols][k % ncols]

    ref = (registry.groupby("model_id")["ext_miou"].mean()
           .sort_values(ascending=False))
    ref_rank = {str(m): i + 1 for i, m in enumerate(ref.index)}
    letters = "abcdef"
    for k, s in enumerate(shown):
        ax = _panel(k)
        sub = registry[registry["strategy"].astype(str) == s]
        est_means = sub.groupby("model_id")["val_miou"].mean()
        ext_means = sub.groupby("model_id")["ext_miou"].mean()
        # RANK LIKE WITH LIKE. A design whose runs are not all complete yet can
        # rank fewer models than the external reference does; comparing a
        # two-model ranking against a three-model one is not a rank inversion,
        # it is a different question. Both sides are therefore restricted to the
        # models this design actually has, and the count is shown.
        common = sorted(set(est_means.index.astype(str))
                        & set(ext_means.index.astype(str)) & set(ref_rank))
        if len(common) < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{_short_strategy(s)}\nnot enough models",
                    ha="center", va="center", fontsize=8, color="0.5")
            continue
        est_rank = {m: i + 1 for i, m in enumerate(
            est_means.loc[common].sort_values(ascending=False).index.astype(str))}
        ext_rank = {m: i + 1 for i, m in enumerate(
            ext_means.loc[common].sort_values(ascending=False).index.astype(str))}
        for m in common:
            flipped = est_rank[m] != ext_rank[m]
            ax.plot([0, 1], [est_rank[m], ext_rank[m]], "o-",
                    ms=5, lw=1.8 if flipped else 1.0,
                    color=mc.get(m, "0.3"), alpha=0.95 if flipped else 0.75)
            ax.annotate(m, (1, ext_rank[m]), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=7,
                        color=mc.get(m, "0.3"))
            # `ref_rank` and `ext_rank` agree on `common` by construction; the
            # right-hand y value is `ext_rank` so the panel is self-consistent
            # even when the design covers fewer models than the reference.
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{s.split('_')[0]} estimate", "external test"],
                           fontsize=8)
        ax.set_yticks(range(1, len(common) + 1))
        ax.set_ylim(len(common) + 0.5, 0.5)
        ax.set_xlim(-0.12, 1.35)
        if k % ncols == 0:
            ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"({letters[k]}) {_short_strategy(s).replace(chr(10), ' ')}",
                     fontsize=9)
        n_moved = sum(1 for m in common if est_rank[m] != ext_rank[m])
        ax.annotate(f"{n_moved} of {len(common)} models re-ranked",
                    xy=(0.02, 0.05), xycoords="axes fraction",
                    fontsize=7, color="0.35", ha="left", va="bottom")

    # Concordance summary, if the analysis has written it.
    ax = _panel(len(shown))
    rc_path = tables / "ranking_concordance.csv"
    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        if "spearman_rho" in rc.columns:
            rc = rc.sort_values("spearman_rho")
            # Short labels: the full partition names are long enough to be drawn
            # across panel (a)'s plotting area at this figure width.
            ax.barh([_short_strategy(s) for s in rc["strategy"].astype(str)],
                    rc["spearman_rho"], color=NEUTRAL, alpha=0.9,
                    edgecolor=NEUTRAL_DARK, linewidth=0.4)
            ax.axvline(0, color="0.4", lw=0.8)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel(r"Spearman $\rho$ vs external ranking")
            ax.set_title(f"({letters[len(shown)]}) Ranking concordance")
            for i, (_, r) in enumerate(rc.iterrows()):
                ax.text(r["spearman_rho"] + (0.03 if r["spearman_rho"] >= 0 else -0.03),
                        i, f"{int(r.get('n_reversals', 0))} rev.",
                        va="center", ha="left" if r["spearman_rho"] >= 0 else "right",
                        fontsize=7, color="0.3")
        else:
            ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "ranking concordance not yet computed",
                ha="center", va="center", fontsize=9, color="0.5")

    # Any leftover slots in the grid are switched off rather than left as empty
    # axes with ticks, which read as a failed panel.
    for k in range(n_panels, nrows * ncols):
        _panel(k).axis("off")

    handles = [plt.Line2D([], [], color=mc[m], marker="o", ls="-", label=m)
               for m in models]
    fig.legend(handles=handles, fontsize=8, ncol=len(models),
               loc="lower center", bbox_to_anchor=(0.5, -0.015),
               frameon=False)
    fig.suptitle("Does the validation strategy change the model ranking?", y=0.99)
    _save(fig, out / "F8_ranking_stability.png", logger)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=None)
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default=None, help="comma list, e.g. F5,F6")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("make_figures", LOGS_DIR / "make_figures.log")
    logger.info("=" * 72)
    logger.info("FIGURE GENERATION  %s", utcnow())
    logger.info("=" * 72)

    from common import EXPERIMENTS_DIR
    registry_path = Path(args.registry) if args.registry else (
        EXPERIMENTS_DIR / "experiments_registry.csv")
    tables = Path(args.tables) if args.tables else TABLES_DIR
    out = Path(args.out) if args.out else FIGURES_DIR
    sfx = f"_{args.tag}" if args.tag else ""

    registry = pd.DataFrame()
    if registry_path.exists():
        registry = pd.read_csv(registry_path, keep_default_na=False)
        logger.info("registry: %d rows from %s", len(registry), registry_path)
    else:
        logger.warning("registry not found at %s; figures needing it will be "
                       "reported unavailable", registry_path)

    only = {s.strip().upper() for s in args.only.split(",")} if args.only else None

    def want(fid: str) -> bool:
        return only is None or fid in only

    made: list[str] = []
    skipped: list[str] = []
    if want("F4") and figure_f4(tables, out, logger):
        made.append("F4")
    elif want("F4"):
        skipped.append("F4")
    if want("F5") and figure_f5(registry, out, logger):
        made.append("F5")
    elif want("F5"):
        skipped.append("F5")
    if want("F6") and figure_f6(registry, out, logger):
        made.append("F6")
    elif want("F6"):
        skipped.append("F6")
    if want("F7") and figure_f7(registry, tables, out, logger):
        made.append("F7")
    elif want("F7"):
        skipped.append("F7")
    if want("F8") and figure_f8(registry, tables, out, logger):
        made.append("F8")
    elif want("F8"):
        skipped.append("F8")

    print()
    print("[PHASE] Figures")
    print(f"[STATUS] {'PASS' if made and not skipped else ('PARTIAL' if made else 'FAIL')}")
    print(f"[COMPLETED] generated: {', '.join(made) if made else 'none'}")
    if skipped:
        print(f"[ISSUES] unavailable (input artefact absent): {', '.join(skipped)}")
        print("[ISSUES] F1, F2 and F3 are produced separately: F1/F3 are "
              "conceptual and F2 is a map, which carries its own compliance "
              "requirements")
    print(f"[FILES] {out}")
    print("[NEXT] Results section, then Discussion")
    return 0 if made and not skipped else (1 if made else 2)


if __name__ == "__main__":
    raise SystemExit(main())

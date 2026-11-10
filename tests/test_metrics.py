#!/usr/bin/env python
"""
test_metrics.py — validates the metric implementations against values computed
BY HAND.

This is the "metric sanity check against a hand-computed reference" that
research_protocol.md §11.2 requires before any full training run. A silently
wrong metric would corrupt every downstream number, so the reference values
below are derived by hand and stated explicitly in the assertions.

Reference case (8-class index space, only 3 classes present):

    pred = [0, 0, 1, 1, 2, 2, 0, 1]
    true = [0, 0, 1, 2, 2, 1, 1, 0]

    confusion (rows = predicted, cols = true):
        row 0: [2, 1, 0]
        row 1: [1, 1, 1]
        row 2: [0, 1, 1]

    pred totals = [3, 3, 2]     true totals = [3, 3, 2]     tp = [2, 1, 1]

    IoU   = [2/4, 1/5, 1/3]      = [0.5,     0.2,     0.33333]
    prec  = [2/3, 1/3, 1/2]      = [0.66667, 0.33333, 0.5]
    rec   = [2/3, 1/3, 1/2]      = [0.66667, 0.33333, 0.5]
    F1    = [2/3, 1/3, 1/2]      = [0.66667, 0.33333, 0.5]
    OA    = (2+1+1)/8            = 0.5
    mIoU (skip absent)   = mean([0.5, 0.2, 0.33333]) = 0.344444
    mIoU (include absent)= (0.5+0.2+0.33333)/8       = 0.129166
    macro F1             = mean([0.66667, 0.33333, 0.5]) = 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.metrics import (  # noqa: E402
    ConfusionMatrix, bootstrap_ci, estimator_metrics, optimism_bias,
    relative_inflation,
)

TOL = 1e-5
FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(f"{label} {detail}".strip())


def approx(a, b, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    print("=" * 72)
    print("METRIC VALIDATION against hand-computed reference values")
    print("=" * 72)

    pred = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    true = np.array([0, 0, 1, 2, 2, 1, 1, 0])

    cm = ConfusionMatrix()
    cm.update(pred, true)

    print("\n--- confusion matrix ---")
    expected = np.array([[2, 1, 0], [1, 1, 1], [0, 1, 1]])
    got = cm.counts[:3, :3]
    print(f"  got:\n{got}")
    check(np.array_equal(got, expected), "confusion matrix matches hand computation",
          f"expected {expected.tolist()}")

    print("\n--- per-class metrics ---")
    iou = cm.iou_per_class()
    pr = cm.precision_per_class()
    rc = cm.recall_per_class()
    f1 = cm.f1_per_class()

    for c, (ei, ep, er, ef) in enumerate([
        (0.5, 2 / 3, 2 / 3, 2 / 3),
        (0.2, 1 / 3, 1 / 3, 1 / 3),
        (1 / 3, 0.5, 0.5, 0.5),
    ]):
        check(approx(iou[c], ei), f"class {c} IoU", f"{iou[c]:.6f} vs {ei:.6f}")
        check(approx(pr[c], ep), f"class {c} precision", f"{pr[c]:.6f} vs {ep:.6f}")
        check(approx(rc[c], er), f"class {c} recall", f"{rc[c]:.6f} vs {er:.6f}")
        check(approx(f1[c], ef), f"class {c} F1", f"{f1[c]:.6f} vs {ef:.6f}")

    print("\n--- aggregate metrics ---")
    check(approx(cm.overall_accuracy(), 0.5), "overall accuracy",
          f"{cm.overall_accuracy():.6f} vs 0.5")
    check(approx(cm.miou(), 0.3444444444), "mIoU (absent=skip)",
          f"{cm.miou():.6f} vs 0.344444")
    check(approx(cm.miou(absent="include"), 0.1291666667), "mIoU (absent=include)",
          f"{cm.miou(absent='include'):.6f} vs 0.129167")
    check(approx(cm.macro_f1(), 0.5), "macro F1", f"{cm.macro_f1():.6f} vs 0.5")
    check(cm.miou() != cm.miou(absent="include"),
          "absent='skip' and absent='include' differ (not silently identical)")

    print("\n--- absent-class handling ---")
    check(np.isnan(iou[3]), "absent class has NaN IoU, not 0",
          f"iou[3]={iou[3]}")
    check(cm.per_class()["true_tot"][3] == 0, "absent class has zero support")

    # A class that is PRESENT in the ground truth but never predicted must score
    # F1 = 0, not NaN. If it were dropped as NaN, macro-F1 would average over a
    # different class set than mIoU (which counts it as a finite 0), and
    # macro-F1 would become NaN for a whole run whose model predicts nothing
    # correctly -- observed on M2 in the real-data smoke test.
    print("\n--- present-but-never-predicted class ---")
    cm3 = ConfusionMatrix()
    # class 0 predicted correctly; class 1 present in truth, never predicted.
    cm3.update(np.array([0, 0, 0, 0]), np.array([0, 1, 1, 1]))
    f3 = cm3.f1_per_class()
    print(f"      f1 = {f3[:3].tolist()}")
    check(f3[1] == 0.0, "present-but-unpredicted class has F1 = 0, not NaN",
          f"f1[1]={f3[1]}")
    check(np.isnan(f3[2]), "a class appearing nowhere keeps NaN F1",
          f"f1[2]={f3[2]}")
    check(np.isfinite(cm3.macro_f1()),
          "macro F1 stays finite when a class is never predicted",
          f"macro_f1={cm3.macro_f1()}")
    # mIoU and macro-F1 must cover the same classes under absent='skip'.
    i3 = cm3.iou_per_class()
    n_iou = int(np.sum(np.isfinite(i3)))
    n_f1 = int(np.sum(np.isfinite(f3)))
    check(n_iou == n_f1,
          "mIoU and macro F1 average over the same class set",
          f"finite IoU classes={n_iou}, finite F1 classes={n_f1}")

    print("\n--- ignore_index handling ---")
    cm2 = ConfusionMatrix()
    cm2.update(np.array([0, 1, 1]), np.array([0, 1, 255]), ignore_index=255)
    check(int(cm2.counts.sum()) == 2, "ignored pixels excluded from the matrix",
          f"{int(cm2.counts.sum())} counted of 3")

    print("\n--- out-of-range predictions ---")
    cm3 = ConfusionMatrix()
    cm3.update(np.array([0, 99, 1]), np.array([0, 1, 1]))
    check(int(cm3.mat[8].sum()) == 1,
          "out-of-range prediction recorded in the ignore row, not coerced",
          f"ignore-row count = {int(cm3.mat[8].sum())}")
    check(int(cm3.counts.sum()) == 2, "in-range predictions still counted",
          f"{int(cm3.counts.sum())}")

    print("\n--- estimator metrics ---")
    em = estimator_metrics(0.80, 0.60)
    check(approx(em["OB"], 0.20), "OB = CV - Ext", f"{em['OB']}")
    check(approx(em["AEE"], 0.20), "AEE = |CV - Ext|", f"{em['AEE']}")
    check(approx(em["RI_pct"], 33.3333333), "RI = (CV-Ext)/Ext*100",
          f"{em['RI_pct']:.4f}")
    check(em["GG_equals_OB"] is True, "GG flagged as coinciding with OB when no "
          "separate validation score is given")

    neg = optimism_bias(0.60, 0.80)
    check(neg < 0, "negative OB is returned signed, not folded to positive",
          f"OB = {neg}")
    check(approx(abs(neg), absolute_estimation_error_safe(0.60, 0.80)),
          "AEE is positive for a negative OB")

    check(relative_inflation(0.5, 0.0) is None,
          "RI is None (undefined) when the external metric is 0, not infinite")

    em2 = estimator_metrics(0.75, 0.60, validation_metric=0.78)
    check(em2["GG_equals_OB"] is False,
          "GG distinguished from OB when a separate selection score is given",
          f"GG={em2['GG']} OB={em2['OB']}")
    check(approx(em2["GG"], 0.18), "GG = Val - Ext", f"{em2['GG']}")

    print("\n--- bootstrap CI ---")
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.05, 60)
    ci = bootstrap_ci(vals, n_resamples=2000, method="percentile", seed=1)
    check(ci["ci_low"] < ci["mean"] < ci["ci_high"],
          "CI brackets the mean", f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
    check(approx(ci["mean"], float(np.mean(vals)), 1e-9), "bootstrap mean matches")
    ci_bca = bootstrap_ci(vals, n_resamples=2000, method="bca", seed=1)
    check(ci_bca["ci_low"] < ci_bca["mean"] < ci_bca["ci_high"],
          "BCa CI brackets the mean", f"[{ci_bca['ci_low']:.4f}, {ci_bca['ci_high']:.4f}]")
    ci_single = bootstrap_ci([0.7])
    check(ci_single["n"] == 1 and ci_single["ci_low"] == 0.7,
          "single-value input handled without error")
    ci_empty = bootstrap_ci([])
    check(ci_empty["n"] == 0 and ci_empty["mean"] is None,
          "empty input returns None rather than raising")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} assertion(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("RESULT: PASS — all metric assertions hold")
    print("=" * 72)
    return 0


def absolute_estimation_error_safe(a: float, b: float) -> float:
    from evaluation.metrics import absolute_estimation_error
    return absolute_estimation_error(a, b)


if __name__ == "__main__":
    sys.exit(main())

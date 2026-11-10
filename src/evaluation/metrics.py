"""
metrics.py — performance metrics and estimator-evaluation metrics.

Two distinct groups live here, and conflating them would be a design error:

  PERFORMANCE metrics (§12.1) describe how well a model segments: mIoU,
  macro F1, OA, and per-class precision/recall/F1/IoU.

  ESTIMATOR metrics (§12.2) describe how well a validation strategy predicts
  the external geographic test: OB, AEE, GG and RI.

All confusion-matrix arithmetic is accumulated in a single integer matrix so
that metrics are computed once, consistently, from the same counts.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from common import CLASS_NAMES, NUM_CLASSES


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
class ConfusionMatrix:
    """Accumulates a (K+1) x K confusion matrix with an ignore row.

    Row K is reserved for ignored pixels, which are excluded from every metric
    rather than being treated as a class. Accumulating in integers and deriving
    all metrics from this one matrix keeps the metrics mutually consistent.
    """

    def __init__(self, n_classes: int = NUM_CLASSES) -> None:
        self.k = n_classes
        self.mat = np.zeros((n_classes + 1, n_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray,
               ignore_index: int | None = None) -> None:
        pred = np.asarray(pred).ravel()
        target = np.asarray(target).ravel()
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")

        if ignore_index is not None:
            keep = target != ignore_index
            pred, target = pred[keep], target[keep]

        # Anything outside 0..K-1 is ignored, not coerced.
        valid = (target >= 0) & (target < self.k)
        pred_v, targ_v = pred[valid], target[valid]

        # Row K collects pixels whose prediction fell outside the label range.
        oob = (pred_v < 0) | (pred_v >= self.k)
        if oob.any():
            np.add.at(self.mat[self.k], targ_v[oob], 1)
        inb = ~oob
        if inb.any():
            np.add.at(self.mat, (pred_v[inb], targ_v[inb]), 1)

    @property
    def counts(self) -> np.ndarray:
        return self.mat[: self.k]

    def per_class(self) -> dict[str, np.ndarray]:
        m = self.counts.astype(np.float64)
        tp = np.diag(m)
        pred_tot = m.sum(axis=1)          # predicted as class i
        true_tot = m.sum(axis=0)          # actually class i
        return {"tp": tp, "pred_tot": pred_tot, "true_tot": true_tot, "mat": m}

    # -- metrics ----------------------------------------------------------- #
    def iou_per_class(self) -> np.ndarray:
        p = self.per_class()
        union = p["pred_tot"] + p["true_tot"] - p["tp"]
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, p["tp"] / union, np.nan)
        return iou

    def precision_per_class(self) -> np.ndarray:
        p = self.per_class()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p["pred_tot"] > 0, p["tp"] / p["pred_tot"], np.nan)

    def recall_per_class(self) -> np.ndarray:
        p = self.per_class()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(p["true_tot"] > 0, p["tp"] / p["true_tot"], np.nan)

    def f1_per_class(self) -> np.ndarray:
        p = self.per_class()
        pr, rc = self.precision_per_class(), self.recall_per_class()
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc), np.nan)
        # A class that OCCURS -- in the ground truth, in the prediction, or both
        # -- has a defined F1 of 0 when it is never matched, because precision
        # and recall are both genuinely 0 there. NaN is reserved for a class that
        # appears nowhere at all, where the score is undefined.
        #
        # This is not cosmetic. Under the `absent='skip'` convention that mIoU
        # uses, a class present in the ground truth but never predicted
        # contributes a finite IoU of 0; if F1 dropped that same class as NaN,
        # mIoU and macro-F1 would be averaged over DIFFERENT class sets and could
        # not be reported side by side. It also made macro-F1 NaN for an entire
        # run whenever a model predicted nothing correctly -- observed on M2 in
        # the real-data smoke test.
        appears = (p["pred_tot"] > 0) | (p["true_tot"] > 0)
        return np.where(np.isfinite(f1), f1, np.where(appears, 0.0, np.nan))

    def miou(self, absent: str = "skip") -> float:
        """Mean IoU.

        `absent='skip'` averages over classes present in the ground truth. This
        is stated explicitly wherever mIoU is reported, because averaging over
        absent classes (counting them as 0) yields a different and lower number.
        """
        iou = self.iou_per_class()
        if absent == "skip":
            present = self.per_class()["true_tot"] > 0
            vals = iou[present]
            vals = vals[np.isfinite(vals)]
        elif absent == "include":
            # Absent classes contribute 0, they are not dropped. (Dropping them
            # would make this branch identical to 'skip'.)
            vals = np.where(np.isfinite(iou), iou, 0.0)
        else:
            raise ValueError(f"absent must be 'skip' or 'include', got {absent!r}")
        return float(np.mean(vals)) if len(vals) else float("nan")

    def macro_f1(self) -> float:
        f = self.f1_per_class()
        f = f[np.isfinite(f)]
        return float(np.mean(f)) if len(f) else float("nan")

    def overall_accuracy(self) -> float:
        m = self.counts.astype(np.float64)
        tot = m.sum()
        return float(np.trace(m) / tot) if tot > 0 else float("nan")

    def all_metrics(self) -> dict[str, Any]:
        iou = self.iou_per_class()
        pr = self.precision_per_class()
        rc = self.recall_per_class()
        f1 = self.f1_per_class()
        true_tot = self.per_class()["true_tot"]

        out: dict[str, Any] = {
            "miou": self.miou(),
            "macro_f1": self.macro_f1(),
            "oa": self.overall_accuracy(),
            "n_pixels": int(self.counts.sum()),
        }
        for c in range(self.k):
            name = CLASS_NAMES[c].replace(" ", "_").lower()
            out[f"iou_{name}"] = float(iou[c]) if np.isfinite(iou[c]) else None
            out[f"precision_{name}"] = float(pr[c]) if np.isfinite(pr[c]) else None
            out[f"recall_{name}"] = float(rc[c]) if np.isfinite(rc[c]) else None
            out[f"f1_{name}"] = float(f1[c]) if np.isfinite(f1[c]) else None
            out[f"support_{name}"] = int(true_tot[c])
        return out

    def to_dataframe(self):
        import pandas as pd
        iou = self.iou_per_class()
        pr = self.precision_per_class()
        rc = self.recall_per_class()
        f1 = self.f1_per_class()
        true_tot = self.per_class()["true_tot"]
        return pd.DataFrame({
            "class_id": range(self.k),
            "class_name": CLASS_NAMES,
            "iou": iou, "precision": pr, "recall": rc, "f1": f1,
            "support_px": true_tot.astype(np.int64),
        })


# --------------------------------------------------------------------------- #
# Estimator-evaluation metrics (§12.2)
# --------------------------------------------------------------------------- #
def optimism_bias(cv_metric: float, external_metric: float) -> float:
    """OB = CV - External. SIGNED: negative means the CV estimate is pessimistic."""
    return float(cv_metric - external_metric)


def absolute_estimation_error(cv_metric: float, external_metric: float) -> float:
    """AEE = |CV - External|."""
    return float(abs(cv_metric - external_metric))


def generalization_gap(validation_metric: float, external_metric: float) -> float:
    """GG = Validation - External.

    `validation_metric` is the score on the selection validation set used for
    early stopping. For hold-out designs GG coincides with OB; for k-fold designs
    CV is fold-averaged while this is a single split, so they differ. The
    distinction must be stated wherever both appear.
    """
    return float(validation_metric - external_metric)


def relative_inflation(cv_metric: float, external_metric: float,
                       eps: float = 1e-12) -> float | None:
    """RI = (CV - External) / External * 100.

    Returns None when External is ~0, because the quantity is undefined there
    rather than infinite. Callers must report it as undefined, not as a number.
    """
    if abs(external_metric) < eps:
        return None
    return float((cv_metric - external_metric) / external_metric * 100.0)


def estimator_metrics(cv_metric: float, external_metric: float,
                      validation_metric: float | None = None) -> dict[str, Any]:
    """All four estimator metrics, with the GG coincidence flagged."""
    gg_val = validation_metric if validation_metric is not None else cv_metric
    return {
        "cv_metric": float(cv_metric),
        "external_metric": float(external_metric),
        "OB": optimism_bias(cv_metric, external_metric),
        "AEE": absolute_estimation_error(cv_metric, external_metric),
        "GG": generalization_gap(gg_val, external_metric),
        "RI_pct": relative_inflation(cv_metric, external_metric),
        "GG_equals_OB": bool(validation_metric is None
                             or abs(gg_val - cv_metric) < 1e-12),
    }


# --------------------------------------------------------------------------- #
# Bootstrap uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Iterable[float],
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    method: str = "bca",
    seed: int = 0,
) -> dict[str, float | None]:
    """Bootstrap confidence interval for the mean of `values`.

    The resampling unit is the caller's responsibility and MUST be stated
    wherever an interval is reported (protocol §13.1): regions for geographic
    designs, patches for random partitions. Passing patch-level values for a
    region-level design would understate the interval, because samples within a
    region are not independent.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": None, "sd": None, "ci_low": None, "ci_high": None, "n": 0}
    if n == 1:
        return {"mean": float(v[0]), "sd": 0.0,
                "ci_low": float(v[0]), "ci_high": float(v[0]), "n": 1}

    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    alpha = 1.0 - confidence_level

    if method == "percentile":
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    elif method == "bca":
        # Bias-corrected and accelerated, via the jackknife.
        z0 = _norm_ppf(float(np.mean(boot < v.mean())))
        jack = np.array([np.delete(v, i).mean() for i in range(n)])
        jm = jack.mean()
        num = ((jm - jack) ** 3).sum()
        den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
        a = num / den if den != 0 else 0.0
        zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
        lo_p = _norm_cdf(z0 + (z0 + zl) / max(1e-12, 1 - a * (z0 + zl)))
        hi_p = _norm_cdf(z0 + (z0 + zu) / max(1e-12, 1 - a * (z0 + zu)))
        lo, hi = np.percentile(boot, [100 * lo_p, 100 * hi_p])
    else:
        raise ValueError(f"unknown bootstrap method: {method}")

    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": float(lo), "ci_high": float(hi),
        "n": n,
        "method": method,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
    }


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(min(max(p, 1e-12), 1 - 1e-12)))


def _norm_cdf(z: float) -> float:
    from scipy.stats import norm
    return float(norm.cdf(z))


def paired_effect_size(a: Iterable[float], b: Iterable[float]) -> dict[str, float | None]:
    """Paired effect sizes for two aligned sets of measurements."""
    av = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    bv = np.asarray([x for x in b if x is not None and np.isfinite(x)], float)
    n = min(len(av), len(bv))
    if n < 2:
        return {"cohens_dz": None, "mean_diff": None, "cliffs_delta": None, "n": n}
    av, bv = av[:n], bv[:n]
    d = av - bv
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else None

    # Cliff's delta (nonparametric, rank-based)
    gt = sum((x > y) for x in av for y in bv)
    lt = sum((x < y) for x in av for y in bv)
    tot = n * n
    delta = float((gt - lt) / tot) if tot else None

    return {"cohens_dz": dz, "mean_diff": float(d.mean()),
            "cliffs_delta": delta, "n": n}

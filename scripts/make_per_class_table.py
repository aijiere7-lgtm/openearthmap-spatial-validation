#!/usr/bin/env python
"""make_per_class_table.py — per-class metrics from the stored run artefacts.

WHY THIS EXISTS
---------------
The acceptance checklist's row 13 ("mIoU / F1 / per-class metrics complete") was
PARTIAL because only mIoU and macro-F1 had been tabulated. The per-class numbers
were never missing: `ConfusionMatrix.all_metrics()` writes IoU, precision, recall,
F1 and support for all eight classes into every `experiments/runs/<run_id>.json`,
for both the validation fold and the external test. This script tabulates what is
already on disk, so nothing is recomputed and no model is re-evaluated.

Two tables are written:

* `results/tables/per_class_metrics.csv` — one row per (run, evaluation set,
  class), which is the supplementary table.
* `results/tables/per_class_bias.csv` — one row per (run, class) with the
  validation-minus-external difference, i.e. the per-class optimism that the
  aggregate mIoU difference is the sum of.

Run:  python scripts/make_per_class_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from common import get_logger, load_config  # noqa: E402

RUNS = PROJ / "experiments" / "runs"
TABLES = PROJ / "results" / "tables"
LOGS = PROJ / "logs"


def main() -> int:
    log = get_logger("make_per_class_table", LOGS / "make_per_class_table.log")
    cfg = load_config()
    classes = list(cfg["dataset"]["class_names"])
    slug = [c.lower().replace(" ", "_") for c in classes]

    rows: list[dict] = []
    run_files = sorted(p for p in RUNS.glob("*.json")
                       if not p.name.endswith(".error.json"))
    log.info("run artefacts: %d", len(run_files))

    for path in run_files:
        d = json.loads(path.read_text(encoding="utf-8"))
        row = d.get("row", {})
        for eval_set, key in (("validation_fold", "val_metrics"),
                              ("external_test", "ext_metrics")):
            metrics = d.get(key) or {}
            for cls, s in zip(classes, slug):
                rows.append({
                    "run_id": row.get("run_id", path.stem),
                    "strategy": row.get("strategy"),
                    "model_id": row.get("model_id"),
                    "model_name": row.get("model_name"),
                    "seed": row.get("seed"),
                    "fold": row.get("fold"),
                    "eval_set": eval_set,
                    "class": cls,
                    "iou": metrics.get(f"iou_{s}"),
                    "precision": metrics.get(f"precision_{s}"),
                    "recall": metrics.get(f"recall_{s}"),
                    "f1": metrics.get(f"f1_{s}"),
                    "support_px": metrics.get(f"support_{s}"),
                })

    table = pd.DataFrame(rows)
    if table.empty:
        log.error("no per-class metrics found; nothing written")
        return 1
    table = table.sort_values(["strategy", "model_id", "seed",
                               "eval_set", "class"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "per_class_metrics.csv", index=False)

    wide = table.pivot_table(
        index=["run_id", "strategy", "model_id", "model_name", "seed", "fold",
               "class"],
        columns="eval_set", values="iou", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"validation_fold": "val_iou",
                                "external_test": "ext_iou"})
    if {"val_iou", "ext_iou"} <= set(wide.columns):
        wide["ob_iou"] = wide["val_iou"] - wide["ext_iou"]
    wide = wide.sort_values(["strategy", "model_id", "seed", "class"])
    wide.to_csv(TABLES / "per_class_bias.csv", index=False)

    log.info("wrote %d rows to %s", len(table), TABLES / "per_class_metrics.csv")
    log.info("wrote %d rows to %s", len(wide), TABLES / "per_class_bias.csv")
    if {"val_iou", "ext_iou"} <= set(wide.columns):
        # Pooling the strategies would average a design effect into a class
        # effect, so the breakdown is per strategy. A class whose optimism is
        # large under one design and negative under another is a statement about
        # the design, not about the class.
        log.info("mean per-class OB (validation IoU - external IoU):")
        piv = wide.pivot_table(index="class", columns="strategy",
                               values="ob_iou", aggfunc="mean")
        for cls, r in piv.iterrows():
            cells = "  ".join(f"{s}={v:+.3f}" for s, v in r.items())
            log.info("  %-18s %s", cls, cells)
    print(f"[PHASE] Per-class supplementary table")
    print(f"[STATUS] PASS")
    print(f"[FILES] {TABLES / 'per_class_metrics.csv'}")
    print(f"[FILES] {TABLES / 'per_class_bias.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

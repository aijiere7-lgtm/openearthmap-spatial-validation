#!/usr/bin/env python
"""
run_smoke_test.py — protocol section 11.2: prove the training pipeline works
before committing to the experiment grid.

WHY THIS IS MANDATORY
---------------------
A defect in the data path, the loss, the evaluation or the checkpointing is
invisible until a run completes. On this host a single training run is measured
in hours, so discovering a shape mismatch after the grid has started would cost
the whole grid. The smoke test runs one tiny epoch per model and asserts the
things that would otherwise fail silently:

  * the dataset yields correctly shaped, correctly ranged tensors
  * the loss is finite and backpropagates
  * evaluation produces a usable confusion matrix and finite metrics
  * a checkpoint is written and can be reloaded

It also checks the label value set actually present in the data. A label code
outside [0, 8) would be a data-integrity finding, not something to paper over.

TWO MODES
---------
--synthetic   Generate small synthetic tiles and exercise the full pipeline.
              This validates the CODE and can run before the archive is
              downloaded. It says nothing about the real data.
(default)     Run against the real inventory. This validates the DATA path.

Both modes are needed: the first catches code defects now, the second catches
data defects later. A pass in synthetic mode must never be reported as evidence
that the real corpus loads.

Output: data/metadata/smoke_test.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    ALL_LABEL_VALUES_INCLUDING_UNKNOWN, DATA_DIR, LABEL_VALUE_TO_CLASS_INDEX,
    LABEL_VALUE_UNKNOWN, LOGS_DIR, METADATA_DIR, NUM_CLASSES, ensure_dirs,
    get_logger, read_json, utcnow, write_json,
)
from data.dataset import (  # noqa: E402
    IGNORE_INDEX, SegDataset, read_image_rgb, read_label,
)
from evaluation.metrics import ConfusionMatrix  # noqa: E402,F401  (API sanity)
from models.builders import SPECS, build_model, model_metadata  # noqa: E402
from training.engine import fit, make_loader, predict_and_score  # noqa: E402

SYNTH_DIR = DATA_DIR / "interim" / "SYNTHETIC_smoke_test"


# --------------------------------------------------------------------------- #
# Synthetic corpus
# --------------------------------------------------------------------------- #
def make_synthetic_corpus(root: Path, n_tiles: int = 12, size: int = 256,
                          seed: int = 0) -> pd.DataFrame:
    """Write small synthetic tiles with learnable structure.

    The label is derived from the image (brightness thresholds) so the task is
    genuinely learnable. A purely random label would let a broken loss still
    appear to "train", which would defeat the purpose of the smoke test.
    """
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    img_dir = root / "images"
    lab_dir = root / "labels"
    img_dir.mkdir(exist_ok=True)
    lab_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_tiles):
        # A smooth random field plus a gradient gives non-trivial structure.
        base = rng.normal(0, 1, (size // 16, size // 16)).astype(np.float32)
        up = np.array(Image.fromarray(base).resize((size, size), Image.BICUBIC))
        grad = np.linspace(0, 60, size, dtype=np.float32)[None, :]
        img = 128 + 40 * up + grad
        img = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack([img, np.clip(img * 0.9, 0, 255),
                        np.clip(img * 1.1, 0, 255)], axis=-1).astype(np.uint8)

        # Labels from brightness quantiles -> classes 0..7, learnable from RGB.
        q = np.quantile(img, np.linspace(0, 1, NUM_CLASSES + 1)[1:-1])
        lab = np.digitize(img, q).astype(np.uint8)

        sid = f"synth_{i:03d}"
        ip = img_dir / f"{sid}.png"
        lp = lab_dir / f"{sid}.png"
        Image.fromarray(rgb).save(ip)
        Image.fromarray(lab).save(lp)
        rows.append({
            "sample_id": sid,
            "region": f"synth_region_{i % 3}",
            "image_path": str(ip.relative_to(root)).replace("\\", "/"),
            "label_path": str(lp.relative_to(root)).replace("\\", "/"),
        })
    return pd.DataFrame(rows)


def real_index(root: Path, limit: int, rinfo: dict | None = None) -> pd.DataFrame:
    inv = METADATA_DIR / "dataset_inventory.csv"
    if not inv.exists():
        raise FileNotFoundError(
            f"{inv} not found; run scripts/build_inventory.py first, or use "
            f"--synthetic")
    df = pd.read_csv(inv, keep_default_na=False)
    if "label_readable" in df.columns:
        df = df[df["label_readable"].astype(str).str.lower()
                .isin({"true", "1", "yes"})]
    df = df.head(limit)
    out = df[["sample_id", "region", "image_path", "label_path"]].copy()

    # Verify the chosen root actually resolves the inventory's relative paths.
    # A root that is one level too high (or too low) still "looks" like a corpus
    # -- discover_layout reports the same 3838/3500 counts for both the wrapper
    # directory and the real corpus -- so the only reliable check is whether the
    # paths resolve. This is the exact failure mode that silently produced zero
    # results in extract_class_composition.py.
    from data.inventory import resolve_inventory_path
    checked = 0
    missing: list[str] = []
    for rel in out["label_path"].head(25):
        rel = str(rel)
        if not rel:
            continue
        checked += 1
        if not resolve_inventory_path(root, rel).exists():
            missing.append(rel)
    if checked and missing:
        raise FileNotFoundError(
            f"the resolved corpus root {root} does not resolve the inventory's "
            f"label paths ({len(missing)} of {checked} sampled paths missing, "
            f"e.g. {missing[0]!r}). The root is at the wrong nesting level; "
            f"refusing to build a dataset that would read nothing. "
            f"root resolution method was {rinfo.get('method') if rinfo else 'n/a'}.")
    return out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_dataset(ds: SegDataset, logger, label_values: set) -> list[str]:
    problems: list[str] = []
    item = ds[0]
    x, y = item["image"], item["mask"]
    logger.info("  sample %s: image %s %s  mask %s %s",
                item["sample_id"], tuple(x.shape), x.dtype,
                tuple(y.shape), y.dtype)
    if x.ndim != 3 or x.shape[0] != 3:
        problems.append(f"image is not (3,H,W): got {tuple(x.shape)}")
    if y.ndim != 2:
        problems.append(f"mask is not (H,W): got {tuple(y.shape)}")
    if x.shape[-2:] != y.shape[-2:]:
        problems.append(f"image/mask spatial mismatch: {tuple(x.shape[-2:])} "
                        f"vs {tuple(y.shape)}")
    if not torch.isfinite(x).all():
        problems.append("image contains non-finite values")
    # Normalised ImageNet inputs legitimately go negative; check the range is
    # plausible rather than assuming [0,1].
    if float(x.min()) < -5.0 or float(x.max()) > 5.0:
        problems.append(f"normalised image range implausible: "
                        f"[{float(x.min()):.2f},{float(x.max()):.2f}]")
    uniq = set(torch.unique(y).tolist())
    if not uniq:
        problems.append("mask is empty")
    # After remap_labels the mask holds class indices in [0, NUM_CLASSES) plus the
    # ignore index. Anything else means the remap table was bypassed.
    bad = {v for v in uniq if v != IGNORE_INDEX and not (0 <= v < NUM_CLASSES)}
    if bad:
        problems.append(f"mask contains out-of-range class codes: {sorted(bad)}")
    if label_values:
        # RAW raster codes are checked against the AUDITED encoding, not against
        # [0, NUM_CLASSES). The release stores 0 = "unknown" and 1..8 for the
        # eight classes, so a raw check against [0,8) would flag both the residual
        # category and Building as errors -- see DECISIONS D-022.
        allowed_raw = set(ALL_LABEL_VALUES_INCLUDING_UNKNOWN)
        unknown_raw = {v for v in label_values if v not in allowed_raw}
        if unknown_raw:
            problems.append(
                f"RAW label codes outside the audited encoding "
                f"{sorted(allowed_raw)} found: {sorted(unknown_raw)} -- "
                f"investigate before training")
        logger.info("  raw label codes present: %s (residual 'unknown' = %d)",
                    sorted(label_values), LABEL_VALUE_UNKNOWN)
        logger.info("  classes actually present in the raw scan: %d of %d",
                    len([v for v in label_values if v in LABEL_VALUE_TO_CLASS_INDEX]),
                    NUM_CLASSES)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated tiles instead of the real inventory")
    ap.add_argument("--n-tiles", type=int, default=12)
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=24,
                    help="real mode: how many samples to load")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--models", default="M1,M2,M3")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("smoke_test", LOGS_DIR / "smoke_test.log")
    logger.info("=" * 72)
    logger.info("TRAINING PIPELINE SMOKE TEST  %s", utcnow())
    logger.info("mode: %s", "SYNTHETIC (validates code only)" if args.synthetic
                else "REAL INVENTORY (validates the data path)")
    logger.info("=" * 72)

    device = torch.device(args.device) if args.device else torch.device("cpu")
    logger.info("device: %s", device)

    result: dict = {
        "run_at_utc": utcnow(),
        "mode": "synthetic" if args.synthetic else "real",
        "device": str(device),
        "models": {},
        "label_value_set": [],
        "ok": False,
        "problems": [],
    }
    problems: list[str] = []

    # ---- build the index -------------------------------------------------- #
    if args.synthetic:
        root = SYNTH_DIR
        if root.exists():
            shutil.rmtree(root)
        idx = make_synthetic_corpus(root, n_tiles=args.n_tiles,
                                    size=args.tile_size, seed=1)
        logger.info("synthetic tiles written to %s (%d tiles, %dpx)",
                    root, len(idx), args.tile_size)
        logger.warning("SYNTHETIC DATA - this validates code paths only and is "
                       "NOT evidence that the real corpus loads")
    else:
        # Root resolution is delegated to the shared evidence-driven resolver.
        # The previous local implementation walked data/{interim,raw,processed}
        # and accepted the first directory whose `discover_layout` reported
        # labels, requiring `image_dir` to be truthy -- but `image_dir` is None
        # for this corpus at BOTH nesting levels, so nothing ever matched and the
        # smoke test could not run against real data at all. Worse, had it
        # matched, `discover_layout` is recursive and reports identical counts
        # (3838 images / 3500 labels) for the wrapper and the real corpus, so the
        # first-listed directory would have won and every relative path would
        # have been off by one level.
        from data.inventory import (read_inventory_relpaths,
                                    resolve_inventory_root)
        root, rinfo = resolve_inventory_root(logger=logger)
        if root is None:
            logger.error("no extracted imagery found; rerun with --synthetic "
                         "to validate code paths meanwhile")
            return 2
        logger.info("real data root: %s (method=%s)", root, rinfo.get("method"))
        for n in rinfo.get("notes", []):
            logger.info("  root note: %s", n)
        idx = real_index(root, args.limit, rinfo=rinfo)

    result["n_index_rows"] = int(len(idx))
    logger.info("index rows: %d", len(idx))

    # ---- raw label value survey ------------------------------------------- #
    raw_values: set[int] = set()
    for p in idx["label_path"].head(50):
        arr = read_label(root / str(p))
        if arr is not None:
            raw_values.update(int(v) for v in np.unique(arr).tolist())
    result["label_value_set"] = sorted(raw_values)
    logger.info("raw label values observed (first %d samples): %s",
                min(50, len(idx)), sorted(raw_values))

    # ---- dataset construction -------------------------------------------- #
    n_val = max(1, len(idx) // 4)
    val_idx = idx.tail(n_val).reset_index(drop=True)
    train_idx = idx.head(len(idx) - n_val).reset_index(drop=True)
    logger.info("train=%d val=%d", len(train_idx), len(val_idx))

    train_ds = SegDataset(train_idx, root, crop_size=args.crop, train=True,
                          base_seed=0, epoch=0)
    val_ds = SegDataset(val_idx, root, crop_size=args.crop, train=False,
                        base_seed=0, epoch=0)

    ds_problems = check_dataset(train_ds, logger, raw_values)
    problems.extend(f"dataset: {p}" for p in ds_problems)
    result["dataset_checks"] = {"problems": ds_problems}

    # loader + collate
    try:
        batch = next(iter(make_loader(train_ds, args.batch_size, True, 0, 0)))
        logger.info("  batch: image %s mask %s ids=%s",
                    tuple(batch["image"].shape), tuple(batch["mask"].shape),
                    batch["sample_id"][:2])
        result["batch_shape"] = list(batch["image"].shape)
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"dataloader: {type(exc).__name__}: {exc}")
        logger.error("dataloader FAILED: %s", exc)

    # ---- per-model smoke -------------------------------------------------- #
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    for mid in wanted:
        if mid not in SPECS:
            problems.append(f"unknown model id {mid}")
            continue
        spec = SPECS[mid]
        entry: dict = {"name": spec.name, "arch": spec.arch}
        logger.info("[%s] %s", mid, spec.name)
        t0 = time.time()
        try:
            model, _ = build_model(mid, NUM_CLASSES, pretrained=True)
            entry.update(model_metadata(model, spec))
        except Exception as exc:                              # noqa: BLE001
            entry["built"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            problems.append(f"{mid}: build failed: {exc}")
            result["models"][mid] = entry
            continue
        entry["built"] = True

        ckpt = LOGS_DIR / "smoke_checkpoints" / f"{mid}_smoke.pt"
        try:
            res = fit(model, train_ds, val_ds, device=device, epochs=args.epochs,
                      batch_size=args.batch_size, lr=1e-4, patience=99,
                      num_workers=0, seed=0, checkpoint_path=ckpt, logger=logger)
            entry["fit"] = {
                "epochs_run": res.epochs_run,
                "train_loss": res.history[0]["train_loss"] if res.history else None,
                "val_miou": res.best_val_metrics.get("miou"),
                "val_macro_f1": res.best_val_metrics.get("macro_f1"),
                "checkpoint": res.best_checkpoint,
                "seconds": res.total_seconds,
            }
            tl = entry["fit"]["train_loss"]
            if tl is None or not np.isfinite(tl):
                problems.append(f"{mid}: training loss is not finite ({tl})")
            mi = entry["fit"]["val_miou"]
            if mi is None or not np.isfinite(mi):
                problems.append(f"{mid}: validation mIoU is not finite ({mi})")
            if not res.best_checkpoint:
                problems.append(f"{mid}: no checkpoint was written")
            else:
                state = torch.load(res.best_checkpoint, map_location="cpu",
                                   weights_only=False)
                if "model_state" not in state:
                    problems.append(f"{mid}: checkpoint lacks model_state")
                entry["checkpoint_reloadable"] = True
            logger.info("  %s OK: loss=%.4f val_mIoU=%.4f in %.1fs",
                        mid, tl if tl is not None else float("nan"),
                        mi if mi is not None else float("nan"), res.total_seconds)
        except Exception as exc:                              # noqa: BLE001
            import traceback
            entry["fit_error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback_tail"] = traceback.format_exc()[-1200:]
            problems.append(f"{mid}: fit failed: {type(exc).__name__}: {exc}")
            logger.error("  %s fit FAILED: %s", mid, exc)
        entry["elapsed_seconds"] = round(time.time() - t0, 1)
        result["models"][mid] = entry

    # ---- verdict ---------------------------------------------------------- #
    result["problems"] = problems
    result["ok"] = (len(problems) == 0)
    write_json(METADATA_DIR / "smoke_test.json", result)

    logger.info("-" * 72)
    print()
    print("[PHASE] Training pipeline smoke test")
    print(f"[MODE] {result['mode']}")
    if result["ok"]:
        print("[STATUS] PASS")
        print(f"[COMPLETED] {len(result['models'])} models: build -> 1 epoch -> "
              f"evaluate -> checkpoint reload, all finite")
        if args.synthetic:
            print("[CAVEAT] Synthetic mode validates CODE ONLY. The real data "
                  "path is still unverified.")
    else:
        print("[STATUS] FAIL")
        for p in problems:
            print(f"  - {p}")
    print(f"[FILES] {METADATA_DIR / 'smoke_test.json'}")
    logger.info("=" * 72)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

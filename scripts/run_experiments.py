#!/usr/bin/env python
"""
run_experiments.py — the experiment grid.

Runs (partition x model x seed) and records, for each run, the validation-fold
estimate and the external geographic estimate side by side. The difference
between those two numbers IS the study's dependent variable.

PROTOCOL GATES ENFORCED IN CODE, NOT BY CONVENTION
--------------------------------------------------
1. REAL-DATA SMOKE TEST FIRST. Refuses to start unless
   data/metadata/smoke_test.json shows a PASS in real-data mode. A synthetic
   pass does not satisfy this, because it says nothing about the corpus.

2. THE EXTERNAL TEST SET IS NEVER USED FOR SELECTION. `fit` receives only the
   partition's own train/val frames and has no parameter that could accept the
   external set. The external evaluation happens after training finishes and
   its result cannot influence the model. There is deliberately no code path
   that reads external metrics and feeds them back.

3. EXTERNAL SAMPLES MUST NOT APPEAR IN ANY PARTITION. Verified here against the
   freeze, independently of the check build_splits.py already performed, because
   a single leaked sample would invalidate the study's central comparison.

4. THE FREEZE IS HASH-VERIFIED before any run, so a silently edited freeze
   cannot pass unnoticed.

5. THE REGISTRY IS APPEND-ONLY, and completed runs are skipped, so an
   interrupted grid resumes without recomputing or overwriting results.

Output: experiments/experiments_registry.csv  (append-only)
        experiments/per_region_metrics.csv    (append-only, long format)
        experiments/runs/<strategy>__<model>__seed<seed>__fold<fold>.json

The column is named `strategy` rather than `design` deliberately: the
pre-registered mixed-effects model in the protocol is
`Metric ~ Strategy + Model + Strategy x Model + (1|Region)`, and a frame whose
strategy column is called something else cannot be fed to that formula. One
name for one concept, in the artefacts and in the analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import runlock  # noqa: E402  (scripts/ added to sys.path immediately above)

from common import (  # noqa: E402
    DATA_DIR, EXPERIMENTS_DIR, EXTERNAL_REGIONS_JSON, LOGS_DIR, METADATA_DIR,
    NUM_CLASSES, SPLITS_DIR, ensure_dirs, get_logger, load_config, read_json,
    freeze_content_sha256, freeze_development_regions,
    freeze_external_regions,
    set_seed, sha256_file, sha256_json_canonical, utcnow, write_json,
)
from data.dataset import IGNORE_INDEX, SegDataset  # noqa: E402
from evaluation.metrics import estimator_metrics  # noqa: E402
from models.builders import SPECS, build_model, model_metadata  # noqa: E402
from splits.diagnostics import fold_frames, representative_train_size  # noqa: E402
from training.engine import fit, make_loader, predict_and_score  # noqa: E402

REGISTRY = EXPERIMENTS_DIR / "experiments_registry.csv"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
PER_REGION = EXPERIMENTS_DIR / "per_region_metrics.csv"
SMOKE_JSON = METADATA_DIR / "smoke_test.json"
SPLIT_INDEX = SPLITS_DIR / "all_splits_index.csv"
# Per-`run_id` locks. One trainer per run, enforced at the moment training
# starts rather than only in the launcher: the grid is started by a repeating
# scheduled task, so a second launcher is a normal event (see scripts/runlock.py
# and DECISIONS D-052).
RUN_LOCKS_DIR = LOGS_DIR / "runlocks"

PER_REGION_COLUMNS = [
    "run_id", "strategy", "model_id", "model_name", "arch", "family", "seed",
    "fold", "eval_set", "region", "n_samples", "n_pixels",
    "miou", "macro_f1", "overall_accuracy",
]

REGISTRY_COLUMNS = [
    "run_id", "strategy", "model_id", "model_name", "arch", "family", "seed",
    "fold", "n_train", "n_val", "n_params_m", "epochs_run", "stopped_early",
    "train_seconds", "val_miou", "val_macro_f1", "val_oa",
    "ext_miou", "ext_macro_f1", "ext_oa",
    "ob_miou", "aee_miou", "gg_miou", "ri_miou",
    "coordinate_tier", "external_freeze_sha256", "config_sha256",
    "git_commit", "started_at_utc", "finished_at_utc", "notes",
]


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def check_smoke_gate(logger, override: bool) -> bool:
    if override:
        logger.warning("smoke-test gate OVERRIDDEN by --skip-smoke-check; the "
                       "real-data pipeline is UNVERIFIED for this grid")
        return True
    if not SMOKE_JSON.exists():
        logger.error("no %s: run scripts/run_smoke_test.py against the real "
                     "data first (protocol section 11.2)", SMOKE_JSON.name)
        return False
    d = read_json(SMOKE_JSON)
    if d.get("mode") != "real":
        logger.error("the only smoke test on record is mode=%r. A synthetic pass "
                     "validates code paths but NOT the corpus. Run the smoke test "
                     "against the real data.", d.get("mode"))
        return False
    if not d.get("ok"):
        logger.error("the real-data smoke test did not pass: %s", d.get("problems"))
        return False
    logger.info("smoke gate: PASS (real-data smoke test on record)")
    return True


# --------------------------------------------------------------------------- #
# Hyperparameters: read them, never default them
# --------------------------------------------------------------------------- #
def training_cfg(cfg: dict) -> dict:
    """Return the protocol-frozen `training:` section, or refuse to run.

    This helper exists because of a real defect, found 2026-09-18 before the grid
    was launched (DECISIONS D-035). The runner read `cfg.get("train", {})`, but
    `configs/default.yaml` defines `training:` — so every `cfg.get("train", {})`
    chain fell through to its hardcoded literal. Nothing raised, because `.get`
    cannot raise.

    The grid would therefore have trained at lr=6e-5, patience=8, batch_size=4
    and seeds 0,1,2 while both the frozen config and research_protocol.md
    section 11 specify lr=1e-4, patience=10, batch_size=8 and seeds
    42,1337,2024,7,17. A silent divergence of exactly this kind is what makes a
    pre-registered study unreproducible, so the lookup is loud from here on.
    """
    for key in ("training", "train"):        # `train` is a legacy alias only
        if isinstance(cfg.get(key), dict):
            return cfg[key]
    raise KeyError(
        "configs/default.yaml has no `training:` section (nor a legacy `train:`). "
        "The runner will not fall back to hardcoded hyperparameters: doing so "
        "would silently train a different model from the one the protocol and "
        "the frozen config describe.")


def cfg_required(section: dict, key: str, where: str):
    """Fetch a value that the training protocol fixes, or raise.

    Used for every hyperparameter that research_protocol.md section 11 pins
    down. A missing value is a configuration defect, not an invitation to
    substitute a literal.
    """
    if key not in section or section[key] is None:
        raise KeyError(
            f"{where}.{key} is absent from the config. research_protocol.md "
            f"section 11 fixes this value, so it must be read from the frozen "
            f"config rather than defaulted in code.")
    return section[key]


def load_freeze(logger) -> tuple[set[str], str]:
    if not EXTERNAL_REGIONS_JSON.exists():
        raise FileNotFoundError(
            f"{EXTERNAL_REGIONS_JSON} not found: the external test set must be "
            f"frozen before any training (protocol section 6)")
    raw = read_json(EXTERNAL_REGIONS_JSON)

    # Read through the shared accessor: the region lists live under "selection",
    # and reading only the top level returned an empty set. An empty set is
    # indistinguishable from "the study has no external test" and would have run
    # the whole grid against the reserved regions.
    names = freeze_external_regions(raw)
    if not names:
        raise RuntimeError(
            f"{EXTERNAL_REGIONS_JSON.name} yielded no external regions. Refusing "
            f"to launch the grid: every run would be evaluated against a "
            f"development region while the registry reported otherwise.")
    regions = set(names)

    # Verify the stored hash rather than trusting it. `freeze_content_sha256`
    # excludes the hash field itself, which is how the freeze was computed;
    # hashing the raw record would include it and never match.
    stored = raw.get("content_sha256") or ""
    recomputed = freeze_content_sha256(raw)
    if stored and stored != recomputed:
        raise RuntimeError(
            f"external_regions.json hash mismatch: stored {stored[:16]}... but "
            f"recomputed {recomputed[:16]}... The freeze was modified after it "
            f"was recorded, which invalidates the study's central comparison.")
    logger.info("external freeze: %d regions, sha256=%s (verified)",
                len(regions), (stored or recomputed)[:16])
    return regions, (stored or recomputed)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_csv(REGISTRY, keep_default_na=False)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def append_registry(row: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: row.get(c) for c in REGISTRY_COLUMNS}])
    header = not REGISTRY.exists()
    df.to_csv(REGISTRY, mode="a", header=header, index=False, encoding="utf-8")


def run_key(strategy: str, model_id: str, seed: int, fold: int) -> str:
    return f"{strategy}__{model_id}__seed{seed}__fold{fold}"


def append_per_region(rows: list[dict]) -> None:
    """Append region-level metric rows (protocol section 13.2).

    Long format, append-only, so the mixed-effects model
    `Metric ~ Strategy + Model + Strategy x Model + (1|Region)` can be fitted
    without re-running inference.
    """
    if not rows:
        return
    PER_REGION.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: r.get(c) for c in PER_REGION_COLUMNS} for r in rows])
    header = not PER_REGION.exists()
    df.to_csv(PER_REGION, mode="a", header=header, index=False, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    # The grid is launched through `pythonw.exe` (D-054: a GUI interpreter never
    # attaches to a console, and three runs were killed by a console-close
    # event). stdio is redirected to the grid log by the launcher, so it is
    # normally valid; this guard covers the case where it is not, because
    # `logging.StreamHandler(None)` would otherwise fail on the first record.
    if sys.stdout is None or sys.stderr is None:
        console_log = (LOGS_DIR / "run_experiments_console.log").open(
            "a", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = console_log
        if sys.stderr is None:
            sys.stderr = console_log

    ap.add_argument("--strategies", "--designs", dest="strategies", default=None,
                    help="comma list of split strategies; default all in the index")
    ap.add_argument("--models", default="M1,M2,M3")
    ap.add_argument("--seeds", default=None,
                    help="comma list of seeds; default = training.seeds from the "
                         "frozen config (research_protocol.md section 11 fixes 5)")
    ap.add_argument("--fold", type=int, default=0,
                    help="which fold to train/evaluate (fold 0 is the fixed "
                         "validation fold used by the matched-size control)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.epochs (default: read from config)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override training.batch_size (default: read from config)")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-smoke-check", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger = get_logger("run_experiments", LOGS_DIR / "run_experiments.log")
    cfg = load_config()
    logger.info("=" * 72)
    logger.info("EXPERIMENT GRID  %s", utcnow())
    logger.info("=" * 72)

    if not check_smoke_gate(logger, args.skip_smoke_check):
        return 3

    # ---- training protocol: READ the frozen config, never default ---------- #
    # Every value below is fixed by research_protocol.md section 11 and mirrored
    # in configs/default.yaml. They were previously read from a `train` key that
    # does not exist, so all of them silently fell back to literals in this file
    # (DECISIONS D-035). See training_cfg() for the full account.
    tcfg = training_cfg(cfg)
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else
             [int(s) for s in cfg_required(tcfg, "seeds", "training")])
    batch_size = int(args.batch_size or cfg_required(tcfg, "batch_size", "training"))
    epochs = int(args.epochs or cfg_required(tcfg, "epochs", "training"))
    crop_size = int(cfg_required(tcfg, "crop_size", "training"))
    lr = float(cfg_required(tcfg, "lr", "training"))
    weight_decay = float(cfg_required(tcfg, "weight_decay", "training"))
    num_workers = int(cfg_required(tcfg, "num_workers", "training"))
    warmup_epochs = int(cfg_required(tcfg, "warmup_epochs", "training"))
    es_cfg = cfg_required(tcfg, "early_stopping", "training")
    patience = int(cfg_required(es_cfg, "patience", "training.early_stopping"))
    loss_cfg = cfg_required(tcfg, "loss", "training")
    ce_weight = float(cfg_required(loss_cfg, "ce_weight", "training.loss"))
    dice_weight = float(cfg_required(loss_cfg, "dice_weight", "training.loss"))

    logger.info("training protocol, read from %s:", cfg.get("_config_path"))
    logger.info("  seeds=%s (%d)", seeds, len(seeds))
    logger.info("  batch_size=%d  crop_size=%d  epochs=%d  warmup_epochs=%d",
                batch_size, crop_size, epochs, warmup_epochs)
    logger.info("  lr=%g  weight_decay=%g  patience=%d  num_workers=%d",
                lr, weight_decay, patience, num_workers)
    logger.info("  loss=ce_dice  ce_weight=%g  dice_weight=%g",
                ce_weight, dice_weight)
    if len(seeds) < 3:
        logger.warning("only %d seed(s): research_protocol.md section 11 requires "
                       "at least 3 (5 targeted).", len(seeds))
    elif len(seeds) < 5:
        logger.warning("%d seeds: meets the protocol minimum of 3 but is below "
                       "the target of 5.", len(seeds))

    hyperparameters = {
        "seeds": seeds, "batch_size": batch_size, "crop_size": crop_size,
        "epochs": epochs, "warmup_epochs": warmup_epochs, "lr": lr,
        "weight_decay": weight_decay, "patience": patience,
        "num_workers": num_workers, "ce_weight": ce_weight,
        "dice_weight": dice_weight,
    }

    # ---- resolve inputs --------------------------------------------------- #
    inv_path = METADATA_DIR / "dataset_inventory.csv"
    if not inv_path.exists():
        logger.error("inventory missing; run scripts/build_inventory.py")
        return 2
    inv = pd.read_csv(inv_path, keep_default_na=False)
    if "label_readable" in inv.columns:
        inv = inv[inv["label_readable"].astype(str).str.lower()
                  .isin({"true", "1", "yes"})].copy()
    logger.info("inventory: %d usable samples", len(inv))

    # Root resolution is delegated to the shared evidence-driven resolver. The
    # local implementation that used to live here walked data/{interim,raw,
    # processed}, required `image_dir` to be truthy (it is None for this corpus
    # at BOTH nesting levels, so nothing ever matched), and would otherwise have
    # accepted the wrapper directory, since discover_layout is recursive and
    # reports identical 3838/3500 counts for the wrapper and the real corpus.
    from data.inventory import resolve_inventory_root, resolve_inventory_path
    root, rinfo = resolve_inventory_root(logger=logger)
    if root is None:
        logger.error("no extracted imagery found under data/; run "
                     "scripts/fetch_openearthmap.py then re-run the data QC")
        return 2
    logger.info("data root: %s (method=%s)", root, rinfo.get("method"))
    for n in rinfo.get("notes", []):
        logger.info("  root note: %s", n)

    # Verify the root resolves the inventory's relative paths before a multi-hour
    # grid is launched. A root one level off still "looks" like a corpus, and the
    # only symptom would be every tile failing to load deep inside the loop.
    _sample = inv["image_path"].astype(str).head(25)
    _missing = [r for r in _sample if r and not resolve_inventory_path(root, r).exists()]
    if _missing:
        logger.error("the resolved root does not resolve the inventory image "
                     "paths (%d of %d sampled missing, e.g. %r). Refusing to "
                     "launch the grid.", len(_missing), len(_sample), _missing[0])
        return 2

    external_regions, freeze_sha = load_freeze(logger)

    if not SPLIT_INDEX.exists():
        logger.error("%s missing; run scripts/build_splits.py first", SPLIT_INDEX)
        return 2
    index = pd.read_csv(SPLIT_INDEX)
    strategies = ([d.strip() for d in args.strategies.split(",") if d.strip()]
              if args.strategies else index["strategy"].tolist())
    strategies = [d for d in strategies if (SPLITS_DIR / f"{d}.csv").exists()]
    logger.info("strategies: %d", len(strategies))

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # ---- external test index (evaluated, never used for selection) -------- #
    ext_mask = inv["region"].astype(str).isin(external_regions)
    ext_index = inv.loc[ext_mask, ["sample_id", "region", "image_path",
                                   "label_path"]].reset_index(drop=True)
    logger.info("external geographic test set: %d samples in %d regions",
                len(ext_index), ext_index["region"].nunique())
    if len(ext_index) == 0:
        logger.error("the external test set is empty; nothing to evaluate "
                     "against. Check the freeze against the inventory regions.")
        return 2

    registry = load_registry()
    done = set(registry["run_id"].astype(str)) if len(registry) else set()
    logger.info("registry: %d completed runs", len(done))

    # Device resolution. The default used to be a hardcoded `torch.device("cpu")`,
    # which meant a working GPU would still be ignored and every run would take
    # CPU time. Precedence: explicit --device, then the config, then the best
    # device actually available.
    if args.device:
        device = torch.device(args.device)
    else:
        want = str(tcfg.get("device") or "auto").lower()
        if want == "auto":
            want = "cuda" if torch.cuda.is_available() else "cpu"
        if want.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("device %r requested but torch.cuda.is_available() is "
                           "False; falling back to CPU. Runs on this host will "
                           "be recorded as CPU-computed.", want)
            want = "cpu"
        device = torch.device(want)
    if device.type == "cuda":
        logger.info("device: %s (%s, %.1f GB)", device,
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    else:
        logger.info("device: %s (torch %s, cuda build %s, cuda available %s)",
                    device, torch.__version__, torch.version.cuda,
                    torch.cuda.is_available())
    logger.info("run provenance will record device=%s", device.type)

    # Path columns are joined here rather than stored in the partition files.
    # A partition file is an ASSIGNMENT (which sample is train/val in which
    # fold); the imagery paths are inventory facts that may legitimately change
    # when the inventory is rebuilt, so embedding them would create stale copies.
    # The runner previously assumed the paths were already present, which raised
    # KeyError: "['image_path', 'label_path'] not in index" on every run.
    path_lookup = inv.set_index("sample_id")[["image_path", "label_path"]]

    def with_paths(frame: pd.DataFrame, what: str) -> pd.DataFrame:
        missing = set(frame["sample_id"].astype(str)) - set(path_lookup.index)
        if missing:
            raise RuntimeError(
                f"{len(missing)} sample ids in the {what} frame are absent from "
                f"the inventory (e.g. {sorted(missing)[:3]}). The partition and "
                f"the inventory disagree; refusing to train on a partially "
                f"resolved set.")
        out = frame.copy()
        out["sample_id"] = out["sample_id"].astype(str)
        out = out.merge(path_lookup, left_on="sample_id", right_index=True,
                        how="left")
        return out

    plan = [(d, m, s, args.fold) for d in strategies for m in model_ids for s in seeds]
    if args.max_runs:
        plan = plan[: args.max_runs]
    logger.info("planned runs: %d", len(plan))

    if args.dry_run:
        for d, m, s, f in plan[:40]:
            logger.info("  would run %s", run_key(d, m, s, f))
        print(f"\n[DRY RUN] {len(plan)} runs planned; nothing executed.")
        return 0

    n_ok, n_skip, n_fail = 0, 0, 0
    for i, (strategy, model_id, seed, fold) in enumerate(plan, 1):
        key = run_key(strategy, model_id, seed, fold)
        if key in done:
            n_skip += 1
            continue

        # ---- ONE TRAINER PER run_id -------------------------------------- #
        # The registry is append-only and identifies a run by `run_id`, so two
        # processes training the same run would append two rows with the same
        # identity -- and would put two 6-9 GB working sets on one 15.82 GB host
        # and one 8 GB GPU. The lock is taken here, immediately before the work,
        # because the grid is started by a scheduled task that repeats every
        # 15 minutes, which makes a second launcher a normal event rather than
        # an exceptional one (scripts/runlock.py, DECISIONS D-052).
        run_lock = RUN_LOCKS_DIR / f"{key}.lock"
        acquired, holder = runlock.acquire(run_lock, run_id=key)
        if not acquired:
            logger.warning("  SKIP %s: pid %s is already training this run; "
                           "refusing to train it twice", key, holder)
            n_skip += 1
            continue

        logger.info("-" * 72)
        logger.info("[%d/%d] %s", i, len(plan), key)
        started = utcnow()
        t0 = time.time()
        notes: list[str] = []

        try:
            part = pd.read_csv(SPLITS_DIR / f"{strategy}.csv", keep_default_na=False)
            train_df, val_df = fold_frames(part, fold)

            # GUARD 3: the external set must not appear in the partition.
            used = set(train_df["sample_id"]) | set(val_df["sample_id"])
            leaked = used & set(ext_index["sample_id"])
            if leaked:
                raise RuntimeError(
                    f"{len(leaked)} external samples appear in partition "
                    f"{strategy} fold {fold}; refusing to train. The study's "
                    f"central comparison would be invalid.")

            meta_cols = ["sample_id", "region", "image_path", "label_path"]
            train_df = with_paths(train_df, f"{strategy} fold {fold} train")[meta_cols].reset_index(drop=True)
            val_df = with_paths(val_df, f"{strategy} fold {fold} val")[meta_cols].reset_index(drop=True)
            logger.info("  train=%d val=%d (external=%d held out)",
                        len(train_df), len(val_df), len(ext_index))

            train_ds = SegDataset(train_df, root, crop_size=crop_size, train=True,
                                  base_seed=seed, epoch=0)
            val_ds = SegDataset(val_df, root, crop_size=None, train=False,
                                base_seed=seed, epoch=0)
            ext_ds = SegDataset(ext_index, root, crop_size=None, train=False,
                                base_seed=seed, epoch=0)

            model, spec = build_model(model_id, NUM_CLASSES, pretrained=True)
            mmeta = model_metadata(model, spec)

            ckpt = RUNS_DIR / f"{key}.pt"
            res = fit(model, train_ds, val_ds, device=device, epochs=epochs,
                      batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                      patience=patience, warmup_epochs=warmup_epochs,
                      ce_weight=ce_weight, dice_weight=dice_weight,
                      seed=seed, checkpoint_path=ckpt, logger=logger,
                      num_workers=num_workers)

            # ---- evaluation: validation fold, then the external set ---------- #
            # Order matters for clarity only: neither result can influence the
            # model, which is already frozen at its best checkpoint.
            #
            # Both evaluations also accumulate a confusion matrix PER REGION.
            # That is what makes the pre-registered mixed-effects model
            # (random intercept for region) fittable; a partition-level metric
            # alone carries no region structure. The evaluation loaders run with
            # shuffle=False, so the dataset order -- and therefore the alignment
            # of `groups` with the samples -- is preserved.
            _, val_metrics, val_by_region = predict_and_score(
                model, make_loader(val_ds, 1, False, 0, seed), device,
                groups=val_df["region"].astype(str).tolist())
            _, ext_metrics, ext_by_region = predict_and_score(
                model, make_loader(ext_ds, 1, False, 0, seed), device,
                groups=ext_index["region"].astype(str).tolist())

            region_rows: list[dict] = []
            for eval_set, frame, by_region in (("val", val_df, val_by_region),
                                               ("ext", ext_index, ext_by_region)):
                n_by_region = (frame["region"].astype(str).value_counts()
                               .to_dict())
                for region, gm in sorted(by_region.items()):
                    region_rows.append({
                        "run_id": key, "strategy": strategy, "model_id": model_id,
                        "model_name": spec.name, "arch": spec.arch,
                        "family": spec.family, "seed": seed, "fold": fold,
                        "eval_set": eval_set, "region": region,
                        "n_samples": int(n_by_region.get(region, 0)),
                        "n_pixels": gm.get("n_pixels"),
                        "miou": gm.get("miou"),
                        "macro_f1": gm.get("macro_f1"),
                        "overall_accuracy": gm.get("overall_accuracy"),
                    })

            # `estimator_metrics` coerces with float(), so an undefined mIoU
            # (a set containing no valid pixels) must be handled before the call
            # rather than crashing the run after training has already completed.
            vm = val_metrics.get("miou")
            em = ext_metrics.get("miou")
            if vm is None or em is None:
                notes.append(
                    f"mIoU undefined (val={vm}, ext={em}); estimator metrics "
                    f"not computed")
                est = {}
            else:
                # For a single-fold design the validation metric IS the CV
                # estimate, so GG coincides with OB. Passing it explicitly makes
                # that coincidence explicit and flagged rather than assumed.
                est = estimator_metrics(vm, em, validation_metric=vm)
                if est.get("GG_equals_OB"):
                    notes.append("GG equals OB by construction (single-fold "
                                 "design); the two are not independent evidence")

            row = {
                "run_id": key, "strategy": strategy, "model_id": model_id,
                "model_name": spec.name, "arch": spec.arch, "family": spec.family,
                "seed": seed, "fold": fold,
                "n_train": len(train_df), "n_val": len(val_df),
                "n_params_m": mmeta["params_total_m"],
                "epochs_run": res.epochs_run, "stopped_early": res.stopped_early,
                "train_seconds": res.total_seconds,
                "val_miou": val_metrics.get("miou"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                "val_oa": val_metrics.get("overall_accuracy"),
                "ext_miou": ext_metrics.get("miou"),
                "ext_macro_f1": ext_metrics.get("macro_f1"),
                "ext_oa": ext_metrics.get("overall_accuracy"),
                "ob_miou": est.get("OB"),
                "aee_miou": est.get("AEE"),
                "gg_miou": est.get("GG"),
                "ri_miou": est.get("RI_pct"),
                "coordinate_tier": (read_json(METADATA_DIR / "spatial_metadata_audit.json").get("tier")
                                    if (METADATA_DIR / "spatial_metadata_audit.json").exists()
                                    else None),
                "external_freeze_sha256": freeze_sha,
                "config_sha256": cfg.get("_config_sha256"),
                "git_commit": None,
                "started_at_utc": started, "finished_at_utc": utcnow(),
                "notes": "; ".join(notes),
            }
            append_registry(row)
            append_per_region(region_rows)
            write_json(RUNS_DIR / f"{key}.json", {
                "row": row, "fit": res.to_dict(),
                # The resolved hyperparameters are persisted per run so that a
                # later reader can audit what actually trained without having to
                # re-derive it from the config hash (D-035).
                "hyperparameters": hyperparameters,
                "val_metrics": val_metrics, "ext_metrics": ext_metrics,
                "val_metrics_by_region": val_by_region,
                "ext_metrics_by_region": ext_by_region,
                "model_metadata": mmeta, "notes": notes,
            })
            done.add(key)
            n_ok += 1
            logger.info("  DONE val_mIoU=%.4f ext_mIoU=%.4f OB=%.4f epochs=%d",
                        vm if vm is not None else float("nan"),
                        em if em is not None else float("nan"),
                        est.get("OB") if est.get("OB") is not None else float("nan"),
                        res.epochs_run)
        except Exception as exc:                              # noqa: BLE001
            import traceback
            n_fail += 1
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            logger.error("  %s", traceback.format_exc()[-800:])
            write_json(RUNS_DIR / f"{key}.error.json", {
                "run_id": key, "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc()[-2000:],
                "at_utc": utcnow(),
            })
            continue
        finally:
            # Reached on success, on `continue` from the handler above, and on
            # any exception that escapes the handler. The lock is per run, so a
            # process that trains several runs in one call drops each lock as it
            # finishes that run rather than holding them all to the end.
            runlock.release(run_lock)

    logger.info("=" * 72)
    print()
    print("[PHASE] Experiment grid")
    print(f"[STATUS] {'PASS' if n_fail == 0 else 'PARTIAL'}")
    print(f"[COMPLETED] ok={n_ok} skipped={n_skip} failed={n_fail} "
          f"of {len(plan)} planned")
    print(f"[FILES] {REGISTRY}")
    print(f"[FILES] {RUNS_DIR}")
    if n_fail:
        print(f"[ISSUES] {n_fail} run(s) failed; see {RUNS_DIR}/*.error.json")
    print("[NEXT] statistics and figures")
    logger.info("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

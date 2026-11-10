"""
engine.py — training and evaluation, shared by every experiment.

The design goal is that a metric difference between two partitions can only come
from the partition. So the optimiser, schedule, loss, augmentation, batch size
and selection rule are all fixed here and identical for every run; only the
train/validation membership changes.

Two protocol requirements are enforced in code:

1. MODEL SELECTION USES THE PARTITION'S OWN VALIDATION FOLD. That is the
   treatment being studied, not a leak. The external geographic test set is
   never touched here: `fit` has no parameter that could accept it, and
   `evaluate` is a separate call made only at the end.

2. TRAINING-FOLD-ONLY STATISTICS. Class weights are computed from the training
   fold of the current partition. Computing them once from the whole corpus
   would leak validation and external information into every run.

Evaluation runs on the full tile (no crop), so the reported accuracy describes
the whole image rather than a central sub-window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import NUM_CLASSES, set_seed
from data.dataset import IGNORE_INDEX, SegDataset, collate
from evaluation.metrics import ConfusionMatrix

# Spatial size that an encoder's downsampling requires at inference time.
# 32 covers the strictest of the three architectures in the study (DeepLabV3+
# with a ResNet-50 backbone demands divisibility by 16 and pads internally to
# 32 for its atrous pyramid). Using one constant for all three models keeps the
# evaluation path identical across architectures.
ENCODER_PAD_MULTIPLE = 32


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """Cross-entropy plus soft Dice.

    CE alone under-weights rare classes; Dice alone is unstable early in
    training. The combination is standard for land-cover segmentation with
    strong class imbalance (Rangeland and Tree together are ~43% of pixels while
    Water is ~3%).
    """

    def __init__(self, weight: torch.Tensor | None = None,
                 ignore_index: int = IGNORE_INDEX,
                 ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             ignore_index=self.ignore_index)

        valid = (target != self.ignore_index)
        if not bool(valid.any()):
            return ce
        num_classes = logits.shape[1]
        tgt = target.clone()
        tgt[~valid] = 0
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(tgt, num_classes).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1).float()
        probs = probs * valid.unsqueeze(1).float()

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return self.ce_weight * ce + self.dice_weight * dice.mean()


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def make_loader(ds: SegDataset, batch_size: int, shuffle: bool,
                num_workers: int = 0, seed: int = 0,
                drop_last: bool | None = None) -> DataLoader:
    """Build a DataLoader.

    The training loader drops the final partial batch. This is not cosmetic: a
    trailing batch of size 1 makes DeepLabV3+ fail outright, because its
    BatchNorm layers receive a single value per channel and cannot compute batch
    statistics in training mode (`Expected more than 1 value per channel`).
    Evaluation is unaffected, since it runs under `model.eval()` and BatchNorm
    then uses running statistics.

    Dropping at most `batch_size - 1` samples per epoch is applied identically to
    every partition, so it cannot bias the strategy comparison. If the fold is
    smaller than one batch, this raises rather than silently training on a
    single sample.
    """
    if drop_last is None:
        drop_last = bool(shuffle)
    if drop_last and len(ds) < batch_size:
        raise ValueError(
            f"training fold has {len(ds)} samples but batch_size is "
            f"{batch_size}; reduce batch_size. Training on a partial batch would "
            f"either fail (BatchNorm) or silently use very few samples.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=False, drop_last=drop_last, generator=gen,
    )


# --------------------------------------------------------------------------- #
# Epoch primitives
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, grad_clip: float | None = 1.0,
                    logger=None, log_every: int = 0) -> dict[str, float]:
    model.train()
    total, n = 0.0, 0
    for step, batch in enumerate(loader, 1):
        x = batch["image"].to(device, non_blocking=False)
        y = batch["mask"].to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.shape[0]
        total += float(loss.detach()) * bs
        n += bs
        if logger and log_every and step % log_every == 0:
            logger.info("      step %d/%d loss=%.4f", step, len(loader),
                        total / max(n, 1))
    return {"loss": total / max(n, 1), "n_samples": n}


@torch.no_grad()
def predict_and_score(model, loader, device, num_classes: int = NUM_CLASSES,
                      ignore_index: int = IGNORE_INDEX,
                      collect_predictions: bool = False,
                      groups: "list[str] | None" = None
                      ) -> "tuple[ConfusionMatrix, dict] | tuple[ConfusionMatrix, dict, dict]":
    """Run inference and accumulate a confusion matrix.

    Returns (cm, metrics). The confusion matrix is the primary artefact: every
    metric is derived from it, so storing it allows any metric to be recomputed
    later without re-running inference.

    If `groups` is given, it must be a sequence aligned with the dataset order
    (the loader runs with shuffle=False for evaluation, so dataset order is
    preserved) naming the group each sample belongs to -- in practice the
    region. A separate confusion matrix is then accumulated per group and the
    function returns `(cm, metrics, per_group)`.

    WHY THIS EXISTS: the protocol pre-registers a mixed-effects model
    `Metric ~ Strategy + Model + Strategy x Model + (1 | Region)`, with a random
    intercept for region because samples inside a region are not independent.
    That model cannot be fitted from partition-level metrics alone -- one number
    per (strategy, model, seed) carries no region structure. Accumulating the
    confusion matrix per region during the single inference pass produces the
    region-level observations the model needs, at no extra inference cost.
    """
    model.eval()
    cm = ConfusionMatrix(n_classes=num_classes)
    per_group: dict[str, ConfusionMatrix] | None = {} if groups is not None else None
    n_pixels = 0
    t0 = time.time()

    if groups is not None and len(groups) != len(loader.dataset):
        raise ValueError(
            f"groups has {len(groups)} entries but the dataset has "
            f"{len(loader.dataset)}; per-group metrics would be misattributed")

    cursor = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=False)
        y = batch["mask"]
        # ---- encoder stride compatibility -------------------------------- #
        # Evaluation reads FULL tiles, and this corpus is not uniformly sized
        # (1000x1000, 900x900, 650x650, 958x1024, 439x406 all occur). Some
        # encoders require the spatial size to be divisible by their output
        # stride: DeepLabV3+ with an ImageNet ResNet backbone raises
        #     RuntimeError: Wrong input shape height=1000, width=1000.
        #     Expected image height and width divisible by 16.
        # which killed every M2 run of the grid. Padding to a multiple and
        # cropping the logits back is exact rather than approximate: the padded
        # region is removed BEFORE any pixel enters a confusion matrix, so it
        # cannot affect a metric. `replicate`/`constant` is used rather than
        # always `reflect`, which is undefined once the padding reaches the
        # dimension being padded.
        h, w = x.shape[-2:]
        ph, pw = (-h) % ENCODER_PAD_MULTIPLE, (-w) % ENCODER_PAD_MULTIPLE
        x_in = x
        if ph or pw:
            mode = ("reflect" if (ph < h and pw < w) else "constant")
            x_in = F.pad(x, (0, pw, 0, ph), mode=mode)

        logits = model(x_in)
        if logits.shape[-2:] != (h, w):
            logits = logits[..., :h, :w]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
        pred = logits.argmax(dim=1).cpu()
        pn, yn = pred.numpy(), y.numpy()
        cm.update(pn, yn, ignore_index=ignore_index)
        n_pixels += int(y.numel())

        if per_group is not None:
            bs = pn.shape[0]
            for i in range(bs):
                g = str(groups[cursor + i])
                if g not in per_group:
                    per_group[g] = ConfusionMatrix(n_classes=num_classes)
                per_group[g].update(pn[i: i + 1], yn[i: i + 1],
                                    ignore_index=ignore_index)
            cursor += bs

    metrics = cm.all_metrics()
    metrics["n_pixels"] = n_pixels
    metrics["inference_seconds"] = round(time.time() - t0, 2)

    if per_group is None:
        return cm, metrics

    group_metrics = {g: c.all_metrics() for g, c in per_group.items()}
    return cm, metrics, group_metrics


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_metrics: dict = field(default_factory=dict)
    best_checkpoint: str | None = None
    epochs_run: int = 0
    stopped_early: bool = False
    total_seconds: float = 0.0
    # Recorded so a run's provenance states which schedule actually ran. The
    # protocol fixes "cosine decay with linear warmup" (section 11), and a
    # schedule that is only asserted in a document cannot be audited later.
    warmup_epochs: int = 0
    lr: float = 0.0
    batch_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_metrics": self.best_val_metrics,
            "best_checkpoint": self.best_checkpoint,
            "epochs_run": self.epochs_run,
            "stopped_early": self.stopped_early,
            "total_seconds": self.total_seconds,
            "warmup_epochs": self.warmup_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


def fit(
    model,
    train_ds: SegDataset,
    val_ds: SegDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int = 4,
    lr: float = 6e-5,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    warmup_epochs: int = 0,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    seed: int = 0,
    patience: int = 8,
    min_delta: float = 1e-4,
    grad_clip: float | None = 1.0,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: Path | None = None,
    logger=None,
    log_every: int = 0,
    selection_metric: str = "miou",
    on_epoch: Callable[[int, dict], None] | None = None,
) -> FitResult:
    """Train with early stopping on the PARTITION'S OWN validation fold.

    Early stopping is deliberate and is part of the treatment: different
    validation strategies see different data, so they may legitimately stop at
    different epochs. `epochs_run` is therefore recorded per run and reported,
    because a strategy that stops much earlier is doing less training, and that
    must be visible rather than hidden behind a single accuracy number.
    """
    set_seed(seed)
    model = model.to(device)

    train_loader = make_loader(train_ds, batch_size, True, num_workers, seed)
    val_loader = make_loader(val_ds, batch_size, False, num_workers, seed)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = CombinedLoss(weight=class_weights, ce_weight=ce_weight,
                             dice_weight=dice_weight).to(device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    # Cosine decay with LINEAR WARMUP, as fixed by research_protocol.md section
    # 11 ("cosine decay with linear warmup; identical across models").
    #
    # This was previously a bare CosineAnnealingLR, i.e. the protocol declared a
    # warmup that the optimiser never performed. Warmup matters here beyond
    # convention: the three encoders are initialised from ImageNet weights and
    # fine-tuned at full learning rate from step one, and a transformer (M3)
    # tolerates that far less gracefully than a CNN. A schedule difference
    # between families would be a confound in RQ3, which compares architectures.
    #
    # LambdaLR multiplies the base lr. `scheduler.step()` is called after each
    # epoch, so during the first epoch `step` is 0; `epoch_no` converts that to
    # the 1-based epoch number about to run, which is the convention the standard
    # warmup+cosine implementation uses. The consequence is that the ramp and the
    # decay both land exactly: epoch `warmup_epochs` runs at the full base lr, and
    # the final epoch runs at eta_min.
    eta_min_ratio = 0.01

    def _lr_lambda(step: int) -> float:
        epoch_no = step + 1
        if warmup_epochs > 0 and epoch_no <= warmup_epochs:
            return float(epoch_no) / float(warmup_epochs)
        progress = (epoch_no - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        # np rather than math: numpy is already imported and verified in this
        # module, so the schedule adds no new import dependency.
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + float(np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    res = FitResult()
    best = -np.inf
    bad = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        # Per-phase wall clock, recorded in the run artefact. Without it, an
        # epoch that takes eight minutes instead of two is one number with no
        # explanation, and the two candidate causes -- a starved loader and slow
        # evaluation -- cannot be told apart after the fact. Measuring is not
        # optional when the diagnosis has already been guessed wrong twice.
        t_train0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=scaler, grad_clip=grad_clip, logger=logger,
                             log_every=log_every)
        train_seconds = time.time() - t_train0
        # VRAM after the TRAINING phase, before validation touches the GPU. The
        # card has 8,188 MiB and the desktop shares it, so this is the number that
        # decides whether the eight-minute epochs are a compute problem or a
        # memory-oversubscription problem: on WDDM, once the card is full the
        # driver migrates allocations to system RAM and every kernel launch waits
        # on PCIe transfers, which shows up as ~0 % GPU utilisation and a process
        # blocked in the driver with almost no CPU time.
        if device.type == "cuda":
            try:
                import torch as _t
                train_reserved_mb = round(_t.cuda.memory_reserved() / 1e6, 1)
            except Exception:                                  # noqa: BLE001
                train_reserved_mb = None
        else:
            train_reserved_mb = None
        t_val0 = time.time()
        _, val = predict_and_score(model, val_loader, device)
        val_seconds = time.time() - t_val0
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "val_loss_proxy_miou": val.get("miou"),
            "val_miou": val.get("miou"),
            "val_macro_f1": val.get("macro_f1"),
            "val_oa": val.get("overall_accuracy"),
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": round(train_seconds, 2),
            "val_seconds": round(val_seconds, 2),
            "val_n_pixels": val.get("n_pixels"),
            "val_inference_seconds": val.get("inference_seconds"),
            "cuda_reserved_after_train_mb": train_reserved_mb,
            "cuda_reserved_after_val_mb": (
                round(torch.cuda.memory_reserved() / 1e6, 1)
                if device.type == "cuda" else None),
            "cuda_peak_alloc_mb": (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if device.type == "cuda" else None),
        }
        res.history.append(row)
        res.epochs_run = epoch

        score = val.get(selection_metric)
        score = -np.inf if score is None else float(score)
        improved = score > best + min_delta
        if improved:
            best = score
            bad = 0
            res.best_epoch = epoch
            res.best_val_metrics = dict(val)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(),
                            "epoch": epoch, "val_metrics": val,
                            "seed": seed}, checkpoint_path)
                res.best_checkpoint = str(checkpoint_path)
        else:
            bad += 1

        if logger:
            # Do NOT write `val.get("macro_f1") or float("nan")`. A macro-F1 of
            # exactly 0.0 is a real, finite result (a model that matched no class)
            # and `0.0 or nan` silently rewrites it as nan, making a total
            # failure look like a missing value.
            mf1 = val.get("macro_f1")
            logger.info(
                "    epoch %3d/%d  train_loss=%.4f  val_mIoU=%.4f  "
                "val_macroF1=%.4f  train=%.0fs val=%.0fs  %s",
                epoch, epochs, tr["loss"],
                score if np.isfinite(score) else float("nan"),
                mf1 if (mf1 is not None and np.isfinite(mf1)) else float("nan"),
                train_seconds, val_seconds,
                "BEST" if improved else f"no-improve({bad}/{patience})")

        if on_epoch is not None:
            on_epoch(epoch, row)

        if bad >= patience:
            res.stopped_early = True
            if logger:
                logger.info("    early stop at epoch %d (best epoch %d)",
                            epoch, res.best_epoch)
            break

    res.total_seconds = round(time.time() - t_start, 1)
    res.warmup_epochs = int(warmup_epochs)
    res.lr = float(lr)
    res.batch_size = int(batch_size)

    # Restore the best checkpoint so the returned model matches the reported
    # metrics. Without this, the caller could evaluate a model in its final,
    # over-fitted state while believing the numbers describe the best epoch.
    if res.best_checkpoint and Path(res.best_checkpoint).exists():
        ckpt = torch.load(res.best_checkpoint, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    return res


def training_fold_class_counts(ds: SegDataset, num_classes: int = NUM_CLASSES,
                               max_samples: int | None = None) -> np.ndarray:
    """Class pixel counts over a TRAINING fold only.

    Reading every label is expensive, so `max_samples` allows a bounded estimate;
    the caller must record whether the counts were exact or estimated.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    for i in range(n):
        lab = ds[i]["mask"].numpy()
        vals, c = np.unique(lab, return_counts=True)
        for v, cc in zip(vals.tolist(), c.tolist()):
            if 0 <= v < num_classes:
                counts[v] += int(cc)
    return counts

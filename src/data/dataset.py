"""
dataset.py — the single dataset implementation used by every experiment.

Design constraints that follow from the protocol:

* ONE implementation. Every model, every partition and every metric must read
  data through this class. A second code path would mean the architecture
  comparison (RQ3) and the strategy comparison (RQ1/RQ2/RQ4) were not run on
  identical inputs, which is exactly the kind of difference that invalidates a
  comparison.

* Reproducible augmentation. The augmentation for a given (sample, epoch) is
  derived from a seed rather than from global RNG state, so results do not
  depend on DataLoader worker count or iteration order. Without this, a rerun
  with a different `num_workers` would silently produce different numbers.

* Label values are REMAPPED, not taken as class indices. The label rasters store
  0 = "unknown", 1 = Bareland, ..., 8 = buildings, so the paper's class k lives
  at integer k+1 and integer 0 is a residual category. `remap_labels` performs
  the conversion to class indices in [0, NUM_CLASSES) and sends both the residual
  value and any genuinely unexpected code to the ignore index. Any value the
  release does not define is a data-integrity signal and must be visible, not
  silently folded into a real class. Getting this mapping wrong mislabels every
  per-class metric while producing plausible numbers -- see DECISIONS D-022.

* Training crops; evaluation does not. Training uses random crops (a standard
  augmentation), but evaluation runs on the full tile so that the reported
  accuracy describes the whole image rather than a central sub-window. The corpus
  is NOT uniformly sized (measured: 2747 tiles at 1024x1024, 710 at 1000x1000,
  260 at 650x650, 50 at 900x900, and ~70 smaller than 512), so evaluation batches
  pad to the largest tile present and the padding is ignored by every metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from common import (
    IGNORE_INDEX, LABEL_VALUE_TO_CLASS_INDEX, NUM_CLASSES,
)

# ImageNet statistics: the encoders are ImageNet-pretrained, so inputs must be
# normalised the way those weights expect.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Vectorised lookup table: raster value -> class index, or IGNORE_INDEX.
# Built from the audited encoding rather than written out by hand, so the table
# and the documented mapping cannot drift apart.
_LOOKUP_SIZE = 256
_LABEL_VALUE_TO_INDEX_LUT = np.full(_LOOKUP_SIZE, IGNORE_INDEX, dtype=np.int64)
for _v, _k in LABEL_VALUE_TO_CLASS_INDEX.items():
    _LABEL_VALUE_TO_INDEX_LUT[_v] = _k


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def read_image_rgb(path: Path) -> np.ndarray | None:
    """Read a 3-band uint8 RGB array, (H, W, 3)."""
    suf = path.suffix.lower()
    try:
        if suf in {".tif", ".tiff"}:
            import rasterio
            with rasterio.open(path) as src:
                n = src.count
                idx = [1, 2, 3] if n >= 3 else [1, 1, 1]
                arr = src.read(idx)                    # (3, H, W)
                arr = np.moveaxis(arr, 0, -1)
                if n == 1:
                    arr = np.repeat(arr, 3, axis=-1)
        else:
            from PIL import Image
            with Image.open(path) as im:
                arr = np.array(im.convert("RGB"))
    except Exception:
        return None
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # Scale to 8-bit only when the range genuinely exceeds it; a 16-bit
        # image scaled by a wrong factor would corrupt the inputs silently.
        a = arr.astype(np.float32)
        mx = float(np.nanmax(a)) if a.size else 0.0
        if mx > 255.0:
            a = a / max(mx, 1e-6) * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def read_label(path: Path) -> np.ndarray | None:
    """Read an integer label array, (H, W)."""
    suf = path.suffix.lower()
    try:
        if suf in {".tif", ".tiff"}:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read(1)
        else:
            from PIL import Image
            with Image.open(path) as im:
                arr = np.array(im)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def remap_labels(arr: np.ndarray, num_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Convert raster label values to class indices, sending the rest to ignore.

    The label raster is NOT encoded as class indices. The authors' own code
    (`open_earth_map/utils.py`, `class_grey_oem`) defines

        unknown=0, Bareland=1, Grass=2, Pavement=3, Road=4,
        Tree=5, Water=6, Cropland=7, buildings=8

    so class index k corresponds to raster value k+1, and raster value 0 is a
    residual category that is not one of the eight land-cover classes. Treating
    the raster value as a class index would label every "unknown" pixel Bareland
    and would discard every Building pixel as out of range.

    Out-of-range values are NOT clamped to a neighbouring class: doing so would
    invent annotations the dataset does not contain. The caller checks whether
    anything unexpected was present via `count_unmapped_values`.
    """
    src = np.asarray(arr)
    if src.dtype != np.uint8:
        # Values outside 0..255 cannot be looked up; they are unexpected by
        # construction and go straight to ignore.
        out = np.full(src.shape, ignore_index, dtype=np.int64)
        inside = (src >= 0) & (src < _LOOKUP_SIZE)
        if inside.any():
            out[inside] = _LABEL_VALUE_TO_INDEX_LUT[src[inside].astype(np.uint8)]
    else:
        out = _LABEL_VALUE_TO_INDEX_LUT[src].astype(np.int64, copy=True)
    if ignore_index != IGNORE_INDEX:
        out[out == IGNORE_INDEX] = ignore_index
    return out


def count_unmapped_values(arr: np.ndarray) -> dict[int, int]:
    """Raster values present in `arr` that do not denote a land-cover class.

    Used by QC to make an unexpected code visible instead of letting it vanish
    into the ignore index.
    """
    src = np.asarray(arr)
    vals, counts = np.unique(src, return_counts=True)
    known = set(LABEL_VALUE_TO_CLASS_INDEX)
    return {
        int(v): int(c)
        for v, c in zip(vals.tolist(), counts.tolist())
        if int(v) not in known
    }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SegDataset(Dataset):
    """Segmentation dataset over an explicit list of samples.

    `index_df` must contain at least: sample_id, image_path, label_path. The set
    of rows IS the split, so the partition logic and the training loop never have
    to agree on anything beyond this table -- which removes a class of bug where
    a fold is defined one way in the split code and another way in the loader.
    """

    def __init__(
        self,
        index_df,
        root: Path,
        crop_size: int | None = 512,
        train: bool = False,
        base_seed: int = 0,
        epoch: int = 0,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        normalise: bool = True,
    ) -> None:
        self.df = index_df.reset_index(drop=True)
        self.root = Path(root)
        self.crop_size = crop_size
        self.train = bool(train)
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.normalise = bool(normalise)

        missing = {"sample_id", "image_path", "label_path"} - set(self.df.columns)
        if missing:
            raise ValueError(f"index_df is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation seed. Called once per epoch."""
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        # Seed per (sample, epoch) so augmentation is independent of worker count
        # and of iteration order.
        return np.random.default_rng(
            (self.base_seed * 1_000_003 + self.epoch * 10_007 + idx) % (2**32))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        sid = str(row["sample_id"])
        img_path = self.root / str(row["image_path"])
        lab_path = self.root / str(row["label_path"])

        img = read_image_rgb(img_path)
        lab = read_label(lab_path)
        if img is None or lab is None:
            raise RuntimeError(
                f"unreadable sample {sid}: image={img_path} label={lab_path}")

        # Align shapes; a mismatch would otherwise broadcast incorrectly later.
        if img.shape[:2] != lab.shape[:2]:
            h = min(img.shape[0], lab.shape[0])
            w = min(img.shape[1], lab.shape[1])
            img = img[:h, :w]
            lab = lab[:h, :w]

        lab = remap_labels(lab, self.num_classes, self.ignore_index)

        if self.crop_size:
            img, lab = self._crop(img, lab, idx)

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        x = x.float().div_(255.0)
        if self.normalise:
            mean = torch.from_numpy(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.from_numpy(IMAGENET_STD).view(3, 1, 1)
            x = (x - mean) / std
        y = torch.from_numpy(np.ascontiguousarray(lab)).long()

        return {"image": x, "mask": y, "sample_id": sid}

    def _crop(self, img: np.ndarray, lab: np.ndarray,
              idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = self._rng(idx)
        h, w = lab.shape[:2]
        cs = self.crop_size

        if self.train:
            # Rotate by a multiple of 90 degrees: safe for a label raster, and it
            # avoids inventing interpolated class values that a free rotation
            # would create on the mask.
            k = int(rng.integers(0, 4))
            if k:
                img = np.rot90(img, k)
                lab = np.rot90(lab, k)
            h, w = lab.shape[:2]
            if h < cs or w < cs:
                pad_h, pad_w = max(0, cs - h), max(0, cs - w)
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                lab = np.pad(lab, ((0, pad_h), (0, pad_w)),
                             mode="constant", constant_values=self.ignore_index)
                h, w = lab.shape[:2]
            top = int(rng.integers(0, h - cs + 1))
            left = int(rng.integers(0, w - cs + 1))
        else:
            # Deterministic centre crop for evaluation: the same pixels every
            # time, so metric differences cannot come from crop placement.
            top = max(0, (h - cs) // 2)
            left = max(0, (w - cs) // 2)

        return (img[top:top + cs, left:left + cs],
                lab[top:top + cs, left:left + cs])


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #
def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack a batch, padding to the largest H/W present.

    The corpus is NOT uniformly sized: the release contains 1024x1024, 1000x1000,
    650x650, 900x900, 958x1024 and ~439x406 tiles. Padding therefore is not a
    defensive nicety -- it is on the normal path whenever an evaluation batch
    mixes sizes, which is why evaluation reads full tiles. Padded mask positions
    carry the ignore index and are excluded from every metric.
    """
    max_h = max(b["image"].shape[-2] for b in batch)
    max_w = max(b["image"].shape[-1] for b in batch)

    imgs, masks, ids = [], [], []
    for b in batch:
        x, y = b["image"], b["mask"]
        ph, pw = max_h - x.shape[-2], max_w - x.shape[-1]
        if ph or pw:
            # `reflect` is only defined while the padding is strictly smaller
            # than the dimension being padded. This corpus is genuinely
            # non-uniform (1,091 of 3,838 tiles are not 1024x1024; 439x406 and
            # 1024x1024 coexist), so an evaluation batch can pair a 406-row tile
            # with a 1024-row one and ask for 586 rows of reflection, which
            # raises
            #     RuntimeError: Padding size should be less than the
            #     corresponding input dimension
            # on the real fold-0 validation -- before any metric is produced,
            # so every run of the grid died at the same line.
            # Reflection is retained where it is defined, because it does not
            # introduce a constant edge into the receptive field; past that
            # limit a constant pad (0, i.e. the ImageNet mean after
            # normalisation) is used. Either way the MASK's padded positions
            # carry IGNORE_INDEX, so they are excluded from every metric and the
            # choice of image padding cannot bias a score.
            mode = ("reflect"
                    if (ph < x.shape[-2] and pw < x.shape[-1]) else "constant")
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)
            y = torch.nn.functional.pad(
                y, (0, pw, 0, ph), mode="constant", value=IGNORE_INDEX)
        imgs.append(x)
        masks.append(y)
        ids.append(b["sample_id"])

    return {"image": torch.stack(imgs), "mask": torch.stack(masks),
            "sample_id": ids}


def class_weights_from_counts(counts: np.ndarray, mode: str = "none") -> torch.Tensor | None:
    """Optional class weighting, derived from the TRAINING fold only.

    Weighting from the full dataset would leak information about the validation
    and external folds into training. The caller is responsible for passing
    training-fold counts; this function only performs the arithmetic.
    """
    if mode == "none":
        return None
    c = np.asarray(counts, dtype=np.float64)
    c = np.where(c <= 0, np.nan, c)
    if mode == "inverse":
        w = np.nanmean(c) / c
    elif mode == "sqrt_inverse":
        w = np.sqrt(np.nanmean(c) / c)
    elif mode == "effective_number":
        beta = 0.999
        w = (1.0 - beta) / (1.0 - np.power(beta, c))
    else:
        raise ValueError(f"unknown class-weight mode {mode!r}")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(w, dtype=torch.float32)

"""
builders.py — the three segmentation models behind a single interface.

Design requirement: all three must present `forward(x) -> logits [B, C, H, W]`
at the INPUT resolution, so that the training loop, loss and evaluation code are
byte-identical across architectures. Any architecture-specific reshaping lives
here and nowhere else. A difference in the training pipeline would confound the
architecture comparison that RQ3 depends on.

Pretraining policy is read from config and applied uniformly. If a model cannot
obtain pretrained weights, the builder does NOT silently fall back — it raises,
because mixed initialisation would turn a pretraining difference into an apparent
architecture effect.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common import CHECKPOINTS_DIR

# --------------------------------------------------------------------------- #
# Pinned pretrained weights
# --------------------------------------------------------------------------- #
# scripts/fetch_pretrained_weights.py pins every initialisation to disk and
# records its SHA-256 in data/metadata/pretrained_weights.json. Training then
# never touches the network, and the exact bytes used are known.
#
# Why this matters for the design: huggingface.co is blocked from this host, so
# M3 (SegFormer-B0) would otherwise have been forced to random initialisation
# while M1/M2 received ImageNet weights. That mixture would confound the
# architecture comparison RQ3 rests on. The mirror hf-mirror.com restores parity,
# so the policy is all-pretrained.
PRETRAINED_DIR = CHECKPOINTS_DIR / "pretrained"
TORCH_HOME_DIR = CHECKPOINTS_DIR / "torch_home"
SEGFORMER_LOCAL_DIR = PRETRAINED_DIR / "nvidia_mit-b0"


def configure_offline_weights() -> dict[str, str]:
    """Point torch/HF at the pinned local weights and forbid network access.

    Returns the environment actually applied, for the run record.
    """
    applied: dict[str, str] = {}
    if TORCH_HOME_DIR.exists():
        os.environ["TORCH_HOME"] = str(TORCH_HOME_DIR)
        applied["TORCH_HOME"] = str(TORCH_HOME_DIR)
    # Stop transformers/timm from reaching out; the pinned directory is used
    # explicitly, so an accidental network fetch should be a loud failure rather
    # than a silent substitution of different weights.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    applied["HF_HUB_OFFLINE"] = os.environ["HF_HUB_OFFLINE"]
    applied["TRANSFORMERS_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"]
    return applied


def pretrained_policy() -> dict[str, Any]:
    """Report which initialisations are actually available on disk."""
    policy: dict[str, Any] = {"dir": str(PRETRAINED_DIR), "available": {}}
    for mid, fname in (("M1", "resnet34-333f7ec4.pth"),
                       ("M2", "resnet50-19c8e357.pth")):
        policy["available"][mid] = (PRETRAINED_DIR / fname).exists()
    policy["available"]["M3"] = (
        (SEGFORMER_LOCAL_DIR / "config.json").exists()
        and (SEGFORMER_LOCAL_DIR / "pytorch_model.bin").exists()
    )
    policy["uniform"] = all(policy["available"].values())
    return policy


# --------------------------------------------------------------------------- #
# SegFormer wrapper
# --------------------------------------------------------------------------- #
class SegFormerSegmentation(nn.Module):
    """Wraps HuggingFace SegFormer so it returns input-resolution logits.

    `SegformerForSemanticSegmentation` emits logits at 1/4 resolution. Returning
    them un-upsampled would silently change the effective evaluation resolution
    relative to the smp models, which is exactly the kind of difference that must
    not exist between architectures in this study.
    """

    def __init__(self, hf_name: str, num_classes: int, pretrained: bool,
                 ignore_mismatched_sizes: bool = True) -> None:
        super().__init__()
        self.hf_name = hf_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        if pretrained:
            from transformers import SegformerForSemanticSegmentation
            # Prefer the pinned local copy so the run is hermetic and offline.
            # `ignore_mismatched_sizes` is required because the checkpoint is the
            # ImageNet *classification* model (1000 labels): the encoder is
            # adopted and the segmentation decode head is newly initialised.
            # That is exactly the same structure as M1/M2, where an ImageNet
            # encoder is paired with a randomly initialised decoder -- which is
            # what keeps the initialisation policy consistent across the three.
            source = str(SEGFORMER_LOCAL_DIR) if SEGFORMER_LOCAL_DIR.exists() else hf_name
            self.weight_source = source
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                source, num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes)
        else:
            from transformers import (SegformerConfig,
                                      SegformerForSemanticSegmentation)
            self.weight_source = "random_init"
            cfg = SegformerConfig(num_labels=num_classes)
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    model_id: str
    name: str
    arch: str
    family: str
    encoder: str | None = None
    hf_model: str | None = None
    notes: list[str] = field(default_factory=list)


SPECS: dict[str, ModelSpec] = {
    "M1": ModelSpec("M1", "unet_resnet34", "Unet", "cnn_encoder_decoder",
                    encoder="resnet34",
                    notes=["skip connections; strong local inductive bias"]),
    "M2": ModelSpec("M2", "deeplabv3plus", "DeepLabV3Plus", "dilated_cnn",
                    encoder="resnet50",
                    notes=["atrous spatial pyramid pooling; wide receptive field "
                           "without resolution loss"]),
    "M3": ModelSpec("M3", "segformer_b0", "SegFormer", "transformer",
                    hf_model="nvidia/mit-b0",
                    notes=["hierarchical self-attention; weakest spatial "
                           "inductive bias of the three"]),
}


def build_model(
    model_id: str,
    num_classes: int,
    pretrained: bool = True,
    in_channels: int = 3,
    verbose: bool = False,
) -> tuple[nn.Module, ModelSpec]:
    """Construct one of the three models. Raises rather than degrading silently."""
    if model_id not in SPECS:
        raise KeyError(f"unknown model_id {model_id!r}; known: {sorted(SPECS)}")
    spec = SPECS[model_id]

    if verbose:
        print(f"building {model_id} ({spec.name}), pretrained={pretrained}")

    if pretrained:
        configure_offline_weights()

    if spec.arch in ("Unet", "DeepLabV3Plus"):
        import segmentation_models_pytorch as smp
        cls = getattr(smp, spec.arch)
        try:
            model = cls(
                encoder_name=spec.encoder,
                encoder_weights="imagenet" if pretrained else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        except Exception as exc:                              # noqa: BLE001
            raise RuntimeError(
                f"{spec.name}: could not build with "
                f"encoder_weights={'imagenet' if pretrained else None}. "
                f"Pinned weights should be at {PRETRAINED_DIR} with a "
                f"TORCH_HOME mirror at {TORCH_HOME_DIR}; run "
                f"scripts/fetch_pretrained_weights.py if they are missing. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
    elif spec.arch == "SegFormer":
        try:
            model = SegFormerSegmentation(spec.hf_model, num_classes, pretrained)
        except Exception as exc:                              # noqa: BLE001
            raise RuntimeError(
                f"{spec.name}: could not build (pretrained={pretrained}). "
                f"Expected pinned weights at {SEGFORMER_LOCAL_DIR}. "
                f"huggingface.co is BLOCKED from this host; the weights are "
                f"obtained through hf-mirror.com by "
                f"scripts/fetch_pretrained_weights.py. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        raise ValueError(f"unsupported arch {spec.arch!r}")

    return model, spec


def model_metadata(model: nn.Module, spec: ModelSpec) -> dict[str, Any]:
    """Per-run model record.

    Parameter count is recorded because it is an acknowledged CONFOUNDER for the
    architecture comparison (protocol §10): the three families differ in
    capacity, and capacity is not separable from family in this design. Recording
    it lets the limitation be quantified rather than merely asserted.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_id": spec.model_id,
        "name": spec.name,
        "arch": spec.arch,
        "family": spec.family,
        "encoder": spec.encoder,
        "hf_model": spec.hf_model,
        "params_total_m": round(total / 1e6, 3),
        "params_trainable_m": round(trainable / 1e6, 3),
        "notes": "; ".join(spec.notes),
    }


def count_parameters_by_module(model: nn.Module, top: int = 8) -> list[tuple[str, float]]:
    """Largest submodules by parameter count — for documenting capacity allocation."""
    rows = [(n, sum(p.numel() for p in m.parameters()) / 1e6)
            for n, m in model.named_children()]
    rows.sort(key=lambda t: -t[1])
    return rows[:top]


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the compute device, honouring an explicit override.

    Device-agnostic by design (DECISIONS D-005): the GPU question for this host is
    unresolved, so no code may assume CUDA is present.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

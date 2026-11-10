#!/usr/bin/env python
"""
probe_pretrained.py — determine which pretrained initialisations are actually
obtainable in this environment.

Why this matters: research_protocol.md §10 requires ImageNet initialisation to be
held IDENTICAL across the three models, because a difference in pretraining would
confound the architecture comparison (RQ3). If one model cannot obtain pretrained
weights, that is a design constraint that must be discovered now and disclosed —
not silently papered over by giving one model random initialisation while the
others get ImageNet weights.

Writes data/metadata/pretrained_probe.json.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import LOGS_DIR, METADATA_DIR, ensure_dirs, utcnow  # noqa: E402

RESULTS: dict = {"probed_at_utc": utcnow(), "models": {}}


def probe(name: str, fn) -> None:
    t0 = time.time()
    rec: dict = {"ok": False, "seconds": None, "error": None, "params_m": None}
    try:
        model = fn()
        n = sum(p.numel() for p in model.parameters())
        rec.update(ok=True, params_m=round(n / 1e6, 3))
        print(f"[OK]   {name:28s} params={rec['params_m']}M")
    except Exception as exc:                                   # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        rec["traceback_tail"] = traceback.format_exc()[-600:]
        print(f"[FAIL] {name:28s} {rec['error'][:200]}")
    rec["seconds"] = round(time.time() - t0, 1)
    RESULTS["models"][name] = rec


def main() -> int:
    ensure_dirs()
    print("=" * 72)
    print("PRETRAINED WEIGHT AVAILABILITY PROBE")
    print("=" * 72)

    import warnings
    warnings.filterwarnings("ignore")

    import segmentation_models_pytorch as smp

    probe("M1_unet_resnet34_imagenet",
          lambda: smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                           in_channels=3, classes=8))
    probe("M1_unet_resnet34_none",
          lambda: smp.Unet(encoder_name="resnet34", encoder_weights=None,
                           in_channels=3, classes=8))
    probe("M2_deeplabv3plus_resnet50_imagenet",
          lambda: smp.DeepLabV3Plus(encoder_name="resnet50",
                                    encoder_weights="imagenet",
                                    in_channels=3, classes=8))
    probe("M2_deeplabv3plus_resnet50_none",
          lambda: smp.DeepLabV3Plus(encoder_name="resnet50",
                                    encoder_weights=None,
                                    in_channels=3, classes=8))

    def segformer():
        from transformers import SegformerForSemanticSegmentation
        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=8, ignore_mismatched_sizes=True)

    probe("M3_segformer_b0_pretrained", segformer)

    def segformer_scratch():
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        cfg = SegformerConfig(num_labels=8)
        return SegformerForSemanticSegmentation(cfg)

    probe("M3_segformer_b0_scratch", segformer_scratch)

    RESULTS["summary"] = {
        "m1_m2_pretrained_available":
            RESULTS["models"]["M1_unet_resnet34_imagenet"]["ok"]
            and RESULTS["models"]["M2_deeplabv3plus_resnet50_imagenet"]["ok"],
        "m3_pretrained_available": RESULTS["models"]["M3_segformer_b0_pretrained"]["ok"],
        "m3_scratch_available": RESULTS["models"]["M3_segformer_b0_scratch"]["ok"],
    }
    RESULTS["summary"]["pretraining_uniform"] = (
        RESULTS["summary"]["m1_m2_pretrained_available"]
        and RESULTS["summary"]["m3_pretrained_available"])

    out = METADATA_DIR / "pretrained_probe.json"
    out.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    s = RESULTS["summary"]
    print(f"M1/M2 ImageNet weights available : {s['m1_m2_pretrained_available']}")
    print(f"M3 SegFormer pretrained available: {s['m3_pretrained_available']}")
    print(f"M3 from scratch available        : {s['m3_scratch_available']}")
    print(f"pretraining UNIFORM across models: {s['pretraining_uniform']}")
    print("=" * 72)
    if not s["pretraining_uniform"]:
        print()
        print("ACTION REQUIRED: pretraining is not uniform across the three models.")
        print("The protocol requires identical initialisation for all models. Either")
        print("obtain the missing weights, or change the design to a consistent")
        print("policy (all-scratch or all-pretrained) and record the change in")
        print("DECISIONS.md. Do NOT run the architecture comparison with mixed")
        print("initialisation and present it as an architecture effect.")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

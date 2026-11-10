#!/usr/bin/env python
"""
fetch_pretrained_weights.py — pin all pretrained initialisations locally.

WHY THIS EXISTS
---------------
The three models must start from a *consistent* initialisation policy. If M1 and
M2 received ImageNet weights while M3 did not, the architecture comparison that
RQ3 rests on would be confounded by pretraining, and no amount of statistical
care downstream could repair it (research_protocol.md section 10).

The probe (scripts/probe_pretrained.py) found:
    M1 U-Net/ResNet34   ImageNet  AVAILABLE   (download.pytorch.org)
    M2 DeepLabV3+/R50   ImageNet  AVAILABLE   (download.pytorch.org)
    M3 SegFormer-B0     ImageNet  BLOCKED     (huggingface.co unreachable)
  -> pretraining_uniform = false

huggingface.co is blocked from this host, but the mirror hf-mirror.com is
reachable and serves nvidia/mit-b0. Downloading through it restores parity, so
the design keeps an all-pretrained policy instead of degrading to all-scratch.

Pinning the files to disk also makes the experiment reproducible offline: after
this script runs once, training never touches the network, and the exact bytes
used are recorded with their SHA-256.

Integrity checks performed (not assumed):
  * torchvision weight files embed the first 8 hex digits of their own SHA-256
    in the filename. That prefix is verified against the computed digest, so a
    corrupted or substituted download is caught. This is the authoritative
    check; byte counts are NOT hard-coded, because asserting an unverified size
    would cause a false failure loop.
  * The M3 config is verified to be genuinely SegFormer-B0 by architecture
    (model_type == 'segformer', hidden_sizes == [32, 64, 160, 256],
    depths == [2, 2, 2, 2]) rather than trusted because the URL said so.

Output: checkpoints/pretrained/<file>
        data/metadata/pretrained_weights.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    CHECKPOINTS_DIR, LOGS_DIR, METADATA_DIR, ensure_dirs, get_logger,
    sha256_file, utcnow, write_json,
)

PRETRAINED_DIR = CHECKPOINTS_DIR / "pretrained"
HF_MIRROR = "https://hf-mirror.com"

# The torch hub cache is where scripts/probe_pretrained.py already placed the
# ResNet encoders. Reusing it avoids re-transferring ~190 MB over a slow link.
TORCH_HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

# --------------------------------------------------------------------------- #
# What we need
# --------------------------------------------------------------------------- #
# torchvision filenames carry the first 8 hex chars of the file's SHA-256.
TORCHVISION_FILES = {
    "M1_resnet34": {
        "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
        "filename": "resnet34-333f7ec4.pth",
        "sha256_prefix": "333f7ec4",
        "role": "U-Net encoder (ResNet-34), ImageNet-1k",
    },
    "M2_resnet50": {
        "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "filename": "resnet50-19c8e357.pth",
        "sha256_prefix": "19c8e357",
        "role": "DeepLabV3+ encoder (ResNet-50), ImageNet-1k",
    },
}

HF_REPO = "nvidia/mit-b0"
# Content-Length values observed from the mirror; used only to enable resume and
# to detect truncation. The recorded SHA-256 is the durable evidence.
HF_FILES = {
    "config.json": {"expected_bytes": 70_043},
    "pytorch_model.bin": {"expected_bytes": 14_380_029},
}


# --------------------------------------------------------------------------- #
# Resumable download
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, logger, retries: int = 8,
             expected_bytes: int | None = None) -> bool:
    """Download with resume. Returns True when the transfer reached EOF.

    Some servers ignore Range and restart from zero; that is handled by
    detecting a response that ignores the offset and truncating the partial
    file rather than appending, which would corrupt it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_bytes and have == expected_bytes:
            logger.info("  already complete: %s", dest.name)
            return True
        if expected_bytes and have > expected_bytes:
            logger.warning("  %s larger than expected (%d > %d); restarting",
                           dest.name, have, expected_bytes)
            dest.unlink()
            have = 0

        req = urllib.request.Request(url, headers={"User-Agent": "oem-research/1.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", 200)
                if have and status == 200:
                    # Server ignored Range and is resending from byte 0, so the
                    # existing partial must be dropped, not appended to.
                    logger.warning("  server ignored Range at offset %d; "
                                   "restarting from zero", have)
                    dest.unlink(missing_ok=True)
                    have = 0
                mode = "ab" if have else "wb"
                t0 = last = time.time()
                with dest.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        now = time.time()
                        if now - last >= 15:
                            done = dest.stat().st_size
                            rate = done / max(1e-6, now - t0) / 1e6
                            pct = (f"{100.0 * done / expected_bytes:5.1f}%"
                                   if expected_bytes else "  ?  ")
                            logger.info("    %s  %s bytes  (%.2f MB/s)",
                                        pct, f"{done:,}", rate)
                            last = now
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("  attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(min(30, 3 * attempt))
            continue

        size = dest.stat().st_size
        if expected_bytes and size != expected_bytes:
            logger.warning("  size mismatch after attempt %d: %d != %d",
                           attempt, size, expected_bytes)
            time.sleep(min(30, 3 * attempt))
            continue
        logger.info("  complete: %s (%s bytes)", dest.name, f"{size:,}")
        return True

    logger.error("  FAILED after %d attempts: %s", retries, url)
    return False


def ensure_torchvision_file(spec: dict, logger, verify_only: bool) -> dict:
    """Obtain a torchvision weight file and verify its embedded SHA-256 prefix.

    Order of preference: an already-pinned copy, then the torch hub cache
    (already populated by the probe), then a fresh download. Every candidate is
    hash-verified before being accepted.
    """
    dest = PRETRAINED_DIR / spec["filename"]
    prefix = spec["sha256_prefix"]

    def digest_ok(p: Path) -> tuple[bool, str]:
        if not p.exists():
            return False, ""
        d = sha256_file(p)
        return d.startswith(prefix), d

    # 1. already pinned and valid
    ok, digest = digest_ok(dest)
    if ok:
        logger.info("  already pinned and verified: %s", dest.name)
        return {"path": dest, "sha256": digest, "source": "already_pinned"}

    # 2. reuse the torch hub cache
    cached = TORCH_HUB_CACHE / spec["filename"]
    ok, digest = digest_ok(cached)
    if ok:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        logger.info("  reused torch hub cache -> %s (sha256=%s)", dest.name,
                    digest[:16])
        return {"path": dest, "sha256": digest, "source": "torch_hub_cache"}

    if verify_only:
        return {"path": None, "sha256": None,
                "source": "missing",
                "error": "not pinned and no valid hub-cache copy"}

    # 3. download, re-verifying the hash each attempt
    for attempt in range(1, 4):
        dest.unlink(missing_ok=True)
        if not download(spec["url"], dest, logger):
            return {"path": None, "sha256": None, "source": "download_failed",
                    "error": "transfer did not complete"}
        ok, digest = digest_ok(dest)
        if ok:
            logger.info("  downloaded and verified: %s", dest.name)
            return {"path": dest, "sha256": digest, "source": "download"}
        logger.error("  attempt %d: SHA-256 prefix mismatch (%s does not start "
                     "with %s); retrying", attempt, digest[:16], prefix)
    return {"path": None, "sha256": None, "source": "hash_mismatch",
            "error": f"SHA-256 prefix {prefix} never matched"}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="check already-pinned files without downloading")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("pretrained_fetch", LOGS_DIR / "fetch_pretrained.log")
    logger.info("=" * 72)
    logger.info("PRETRAINED WEIGHT PINNING  %s", utcnow())
    logger.info("=" * 72)

    record: dict = {
        "fetched_at_utc": utcnow(),
        "policy": "all-pretrained (ImageNet-1k) for M1, M2 and M3",
        "why": ("huggingface.co is blocked from this host; hf-mirror.com is "
                "reachable and serves nvidia/mit-b0, so M3 can share the "
                "ImageNet initialisation of M1/M2 instead of forcing an "
                "all-scratch downgrade"),
        "files": {},
        "checks": {},
        "ok": False,
        "problems": [],
    }
    problems: list[str] = []

    # ---- M1 / M2: torchvision ResNet encoders ---------------------------- #
    for key, spec in TORCHVISION_FILES.items():
        logger.info("[%s] %s", key, spec["role"])
        got = ensure_torchvision_file(spec, logger, args.verify_only)
        if got["path"] is None:
            problems.append(f"{key}: {got.get('error', 'unavailable')}")
            continue
        p: Path = got["path"]
        record["files"][key] = {
            "role": spec["role"], "url": spec["url"],
            "local_path": str(p.relative_to(CHECKPOINTS_DIR.parent)),
            "bytes": p.stat().st_size, "sha256": got["sha256"],
            "filename_embedded_prefix_verified": True,
            "obtained_via": got["source"],
        }
        logger.info("  sha256=%s  bytes=%s  via=%s", got["sha256"][:16],
                    f"{p.stat().st_size:,}", got["source"])

    # ---- mirror the ResNet encoders into a hermetic TORCH_HOME ------------ #
    # torchvision (and therefore segmentation_models_pytorch) resolves encoder
    # weights from <TORCH_HOME>/hub/checkpoints. Populating that layout here
    # means training loads exactly the verified bytes with no network access.
    # Hard-linked where the filesystem allows it, so this costs no extra space.
    th_ckpt = CHECKPOINTS_DIR / "torch_home" / "hub" / "checkpoints"
    th_ckpt.mkdir(parents=True, exist_ok=True)
    for key, spec in TORCHVISION_FILES.items():
        src = PRETRAINED_DIR / spec["filename"]
        if not src.exists():
            continue
        dst = th_ckpt / spec["filename"]
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.unlink(missing_ok=True)
        try:
            import os as _os
            _os.link(src, dst)
            how = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            how = "copy"
        logger.info("  TORCH_HOME <- %s (%s)", spec["filename"], how)
    record["torch_home"] = str(th_ckpt.relative_to(CHECKPOINTS_DIR.parent))

    # ---- M3: SegFormer-B0 encoder via the mirror -------------------------- #
    m3_dir = PRETRAINED_DIR / "nvidia_mit-b0"
    logger.info("[M3] SegFormer-B0 encoder (MiT-B0), ImageNet-1k  via %s", HF_MIRROR)
    for fname, spec in HF_FILES.items():
        dest = m3_dir / fname
        url = f"{HF_MIRROR}/{HF_REPO}/resolve/main/{fname}"
        if args.verify_only:
            if not dest.exists():
                problems.append(f"M3:{fname}: not present at {dest}")
                continue
        elif not download(url, dest, logger, expected_bytes=spec["expected_bytes"]):
            problems.append(f"M3:{fname}: download failed")
            continue
        digest = sha256_file(dest)
        record["files"][f"M3_mit_b0/{fname}"] = {
            "role": "SegFormer-B0 encoder (MiT-B0), ImageNet-1k",
            "url": url, "mirror": HF_MIRROR,
            "local_path": str(dest.relative_to(CHECKPOINTS_DIR.parent)),
            "bytes": dest.stat().st_size, "sha256": digest,
        }
        logger.info("  %s sha256=%s", fname, digest[:16])

    # ---- Verify the M3 config really is SegFormer-B0 ---------------------- #
    cfg_path = m3_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            checks = {
                "model_type": cfg.get("model_type"),
                "hidden_sizes": cfg.get("hidden_sizes"),
                "depths": cfg.get("depths"),
                "architectures": cfg.get("architectures"),
            }
            is_b0 = (checks["model_type"] == "segformer"
                     and checks["hidden_sizes"] == [32, 64, 160, 256]
                     and checks["depths"] == [2, 2, 2, 2])
            record["checks"]["m3_config_is_segformer_b0"] = bool(is_b0)
            record["checks"]["m3_config_detail"] = checks
            if not is_b0:
                problems.append(f"M3 config is not SegFormer-B0 as expected: {checks}")
            else:
                logger.info("  config verified as SegFormer-B0: %s", checks)
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"M3 config unreadable: {exc}")

    # ---- Load-test the pinned M3 weights offline -------------------------- #
    if (m3_dir / "config.json").exists() and (m3_dir / "pytorch_model.bin").exists():
        try:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            from transformers import SegformerForSemanticSegmentation
            net = SegformerForSemanticSegmentation.from_pretrained(
                str(m3_dir), num_labels=8, ignore_mismatched_sizes=True)
            n_params = sum(p.numel() for p in net.parameters())
            record["checks"]["m3_offline_load_ok"] = True
            record["checks"]["m3_params_m"] = round(n_params / 1e6, 3)
            logger.info("  offline load OK: %.3fM params (num_labels=8)",
                        n_params / 1e6)
        except Exception as exc:                                  # noqa: BLE001
            record["checks"]["m3_offline_load_ok"] = False
            problems.append(f"M3 offline load failed: {type(exc).__name__}: {exc}")
            logger.error("  offline load FAILED: %s", exc)

    # ---- Verdict ---------------------------------------------------------- #
    record["problems"] = problems
    record["ok"] = (len(problems) == 0)
    write_json(METADATA_DIR / "pretrained_weights.json", record)

    logger.info("-" * 72)
    if record["ok"]:
        logger.info("RESULT: PASS - all three initialisations pinned and verified")
        logger.info("Policy: all-pretrained (ImageNet-1k) across M1, M2, M3.")
    else:
        logger.error("RESULT: FAIL - %d problem(s)", len(problems))
        for p in problems:
            logger.error("  - %s", p)
        logger.error("Do NOT run the architecture comparison with mixed "
                     "initialisation; fix or switch to a consistent policy.")
    logger.info("=" * 72)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

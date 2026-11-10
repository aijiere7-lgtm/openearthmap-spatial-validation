#!/usr/bin/env python
"""
fetch_openearthmap.py — resumable retrieval + integrity verification of OpenEarthMap.

Why a dedicated script rather than curl in a shell pipeline:
  A `curl -C -` loop piped into `tail`/`tee` failed on this host with
  `curl: (23) client returned ERROR on write of 15116 bytes` while the same
  curl invocation succeeded when run directly without the pipeline. Rather than
  debug shell pipe semantics under Git Bash, the transfer is done in Python,
  where resume logic, progress accounting and integrity checks are explicit.

Why a mirror: the canonical Zenodo record (zenodo.org/records/7223446) is not
reachable from this host (proxy 502 on CONNECT). ModelScope publishes the same
upstream dataset together with a SHA-256, which is what this script verifies.
See data/metadata/dataset_source_verification.md section 6 for the rationale
and the disclosed limitation.

Usage:
    python scripts/fetch_openearthmap.py [--verify-only] [--no-extract]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.inventory import discover_layout  # noqa: E402


def _has_imagery(root: Path) -> bool:
    """True when image/label directories with content exist under `root`.

    Used instead of testing for the extraction directory's mere existence: after
    the outer tar is unpacked that directory exists but contains only a zip, so
    an existence check would wrongly report success.

    Do NOT test `layout.image_dir` truthiness. For a multi-region corpus --
    which is what OpenEarthMap is (`<region>/images` + `<region>/labels`) --
    `Layout` deliberately leaves `image_dir`/`label_dir` as None and carries the
    per-region directories in `region_dirs` instead, because collapsing them to
    one directory would silently truncate the corpus to a single region. An
    `image_dir` test therefore returns False for a fully extracted corpus, the
    stage-2 skip never fires, and every pipeline run re-extracts the 9.1 GB
    inner zip -- rewriting all 7,338 corpus files and, if interrupted, leaving
    truncated imagery behind. Count the files instead; that is independent of
    the layout shape.
    """
    if not root.exists():
        return False
    try:
        lay = discover_layout(root)
    except Exception:                                          # noqa: BLE001
        return False
    return lay.n_image_files > 0 and lay.n_label_files > 0

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EXPECTED_SHA256 = "947bdb1259767afadfc8bcc8df998bfdd78b517374076653cc2f50206588f3cf"
EXPECTED_BYTES = 9_095_072_903
ARCHIVE_NAME = "OpenEarthMap.tar.gz"
API_URL = (
    "https://www.modelscope.cn/api/v1/datasets/OmniData/OpenEarthMap/repo"
    f"?Revision=master&FilePath=raw/{ARCHIVE_NAME}"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
META_DIR = PROJECT_ROOT / "data" / "metadata"
LOG_DIR = PROJECT_ROOT / "logs"

CHUNK = 1 << 22          # 4 MiB
MAX_ATTEMPTS = 200
PROGRESS_EVERY_S = 20.0


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def sha256_of(path: Path, logfile: Path | None = None, label: str = "") -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    t0 = time.time()
    last = t0
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            now = time.time()
            if now - last >= PROGRESS_EVERY_S:
                pct = 100.0 * done / total if total else 0.0
                mbps = done / max(now - t0, 1e-9) / 1e6
                log(f"  hashing {label}: {pct:5.1f}%  ({mbps:.0f} MB/s)", logfile)
                last = now
    return h.hexdigest()


def download(url: str, dest: Path, logfile: Path) -> int:
    """Resumable download. Returns final byte count."""
    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "openearthmap-fetch/1.0"})

    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= EXPECTED_BYTES:
            log(f"download complete: {have} bytes", logfile)
            return have

        attempt += 1
        if attempt > MAX_ATTEMPTS:
            raise RuntimeError(
                f"exceeded {MAX_ATTEMPTS} attempts at {have}/{EXPECTED_BYTES} bytes"
            )

        headers = {"Range": f"bytes={have}-"} if have else {}
        log(f"attempt {attempt}: requesting from byte {have}", logfile)

        try:
            # A fresh request each attempt: the LFS CDN link is time-signed, so
            # re-resolving the API URL is what keeps long resumes working.
            with session.get(
                url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True
            ) as resp:
                if resp.status_code not in (200, 206):
                    log(f"  unexpected status {resp.status_code}; retrying", logfile)
                    time.sleep(5)
                    continue

                if have and resp.status_code != 206:
                    # Server ignored the range: restart from zero rather than
                    # appending a full body onto a partial file.
                    log("  server ignored Range header; restarting from 0", logfile)
                    have = 0
                    part.unlink(missing_ok=True)

                mode = "ab" if have else "wb"
                written = have
                t0 = time.time()
                last = t0
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last >= PROGRESS_EVERY_S:
                            pct = 100.0 * written / EXPECTED_BYTES
                            mbps = (written - have) / max(now - t0, 1e-9) / 1e6
                            log(
                                f"  {pct:5.1f}%  {written:,}/{EXPECTED_BYTES:,} bytes"
                                f"  ({mbps:.1f} MB/s)",
                                logfile,
                            )
                            last = now
                log(f"  chunk ended at {written:,} bytes", logfile)

        except (requests.RequestException, OSError) as exc:
            log(f"  transfer error: {type(exc).__name__}: {exc}", logfile)
            time.sleep(5)

        if part.exists() and part.stat().st_size >= EXPECTED_BYTES:
            log(f"download complete: {part.stat().st_size} bytes", logfile)
            return part.stat().st_size


def already_complete(dest: Path, logfile: Path) -> bool:
    """True when `dest` is present, the right size, and hash-verified.

    WHY THIS EXISTS. Without it the script re-acquires the archive on every run:
    `download()` is called unconditionally unless `--verify-only` is passed. The
    transfer is *resumable* but was not *idempotent*, and the difference matters
    for a 9 GB file -- a re-run after the archive had already landed restarted a
    fresh transfer and left a 465 MB `.part` behind.

    The hash is always computed rather than trusted from a marker file, because
    the cost is asymmetric: hashing 9 GB takes about 9 seconds, while acting on
    an archive that is silently corrupt wastes hours and can invalidate results.
    """
    if not dest.exists():
        return False
    n = dest.stat().st_size
    if n != EXPECTED_BYTES:
        log(f"existing {dest.name} is {n:,} bytes, expected "
            f"{EXPECTED_BYTES:,}: it will be re-acquired", logfile)
        return False
    log(f"existing {dest.name} matches the expected size; verifying SHA-256 "
        f"before skipping the transfer ...", logfile)
    sha = sha256_of(dest, logfile, dest.name)
    if sha != EXPECTED_SHA256:
        log(f"existing {dest.name} hashes to {sha}, not the expected value: it "
            f"will be re-acquired", logfile)
        return False
    log("existing archive is complete and verified; skipping the transfer",
        logfile)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="skip download; verify and extract an existing archive")
    ap.add_argument("--no-extract", action="store_true",
                    help="verify only, do not extract")
    ap.add_argument("--force-download", action="store_true",
                    help="re-acquire even if a verified archive is already present")
    ap.add_argument("--deep-verify", action="store_true",
                    help="read every member of the nested archive before "
                         "extracting (slow: it decompresses the whole archive)")
    args = ap.parse_args()

    for d in (RAW_DIR, META_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / "fetch_openearthmap.log"
    dest = RAW_DIR / ARCHIVE_NAME
    part = dest.with_suffix(dest.suffix + ".part")

    log("=== OpenEarthMap acquisition ===", logfile)
    log(f"destination : {dest}", logfile)
    log(f"expected    : {EXPECTED_BYTES:,} bytes", logfile)
    log(f"expected sha: {EXPECTED_SHA256}", logfile)

    # ---- 1. transfer ------------------------------------------------------ #
    skip_transfer = args.verify_only or (
        not args.force_download and already_complete(dest, logfile))
    if skip_transfer:
        # A partial download of an archive we already hold complete has no value,
        # and leaving it would make the next run resume from a stale offset
        # against a file that is already finished.
        if part.exists():
            log(f"removing stale partial download {part.name} "
                f"({part.stat().st_size:,} bytes): the complete archive is "
                f"present and verified", logfile)
            part.unlink()
    else:
        try:
            size = download(API_URL, dest, logfile)
        except RuntimeError as exc:
            log(f"FAIL: {exc}", logfile)
            return 1
        if part.exists() and not dest.exists():
            part.replace(dest)
        elif part.exists() and dest.exists():
            pass
    target = dest if dest.exists() else part
    if not target.exists():
        log("FAIL: no archive found to verify", logfile)
        return 1

    # ---- 2. size ---------------------------------------------------------- #
    actual_bytes = target.stat().st_size
    log(f"size measured: {actual_bytes:,} bytes", logfile)
    if actual_bytes != EXPECTED_BYTES:
        log(f"FAIL: size mismatch (expected {EXPECTED_BYTES:,})", logfile)
        return 1
    log("size OK", logfile)

    # ---- 3. sha256 -------------------------------------------------------- #
    log("computing SHA-256 ...", logfile)
    actual_sha = sha256_of(target, logfile, target.name)
    log(f"measured sha256: {actual_sha}", logfile)
    log(f"expected sha256: {EXPECTED_SHA256}", logfile)
    if actual_sha != EXPECTED_SHA256:
        log("FAIL: SHA-256 mismatch. Archive NOT verified. Not proceeding.", logfile)
        return 2
    log("SHA-256 OK", logfile)

    # ---- 4. archive readability ------------------------------------------- #
    log("inspecting archive members ...", logfile)
    n_members = 0
    sample: list[str] = []
    with tarfile.open(target, "r:gz") as tf:
        for m in tf:
            n_members += 1
            if len(sample) < 25:
                sample.append(m.name)
    log(f"archive members: {n_members:,}", logfile)
    for s in sample:
        log(f"  {s}", logfile)

    # ---- 5. provenance ---------------------------------------------------- #
    prov = {
        "dataset": "OpenEarthMap",
        "upstream_project": "https://open-earth-map.org/",
        "upstream_paper": "arXiv:2210.10732",
        "upstream_doi_published": "10.1109/WACV56688.2023.00619",
        "canonical_record": "https://zenodo.org/records/7223446",
        "canonical_status": "unreachable_from_this_host",
        "canonical_expected_md5": "64155d1dc9d3b68536063f79878e1a67",
        "retrieved_from": "https://www.modelscope.cn/datasets/OmniData/OpenEarthMap",
        "retrieved_url": API_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "archive_file": target.name,
        "archive_bytes": actual_bytes,
        "sha256_measured": actual_sha,
        "sha256_expected": EXPECTED_SHA256,
        "sha256_match": True,
        "archive_members": n_members,
        "licence_declared_by_mirror": "CC BY-NC-SA 4.0",
        "caveat": (
            "Canonical Zenodo artefact could not be fetched from this host. "
            "Hash verification establishes transport integrity of the mirror "
            "artefact; it does not independently establish byte-equivalence with "
            "the Zenodo .zip, which uses a different hash algorithm and container "
            "format. File-level content comparison against a Zenodo download "
            "remains outstanding."
        ),
    }
    (META_DIR / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"provenance written: {META_DIR / 'provenance.json'}", logfile)

    # ---- 6. extract (TWO STAGES) ------------------------------------------ #
    # The distributed archive is DOUBLY NESTED. Listing it directly showed
    # exactly two members:
    #     OpenEarthMap/
    #     OpenEarthMap/OpenEarthMap.zip
    # so a single tar extraction yields a ~9 GB zip and NO imagery. That failure
    # would surface much later as a confusing layout-discovery error, so both
    # stages are handled here and the resulting layout is reported.
    if not args.no_extract:
        extract_root = RAW_DIR / "OpenEarthMap"

        # -- stage 1: the outer tar.gz --
        if extract_root.exists():
            log(f"stage 1: already present, skipping ({extract_root})", logfile)
        else:
            log(f"stage 1: extracting {target.name} to {RAW_DIR} ...", logfile)
            with tarfile.open(target, "r:gz") as tf:
                # `filter="data"` rejects absolute paths and traversal outside
                # the destination. Python 3.14 makes it the default; setting it
                # explicitly keeps the behaviour identical on 3.12+ and avoids
                # the deprecation warning. Older interpreters do not accept the
                # keyword, hence the fallback.
                try:
                    tf.extractall(RAW_DIR, filter="data")
                except TypeError:
                    tf.extractall(RAW_DIR)
            log("stage 1 complete", logfile)

        # -- stage 2: any archive nested inside --
        if _has_imagery(extract_root):
            log("stage 2: imagery already extracted, skipping", logfile)
        else:
            inner = sorted(
                (p for p in extract_root.rglob("*.zip") if p.is_file()),
                key=lambda p: -p.stat().st_size)
            if not inner:
                log(f"stage 2: no nested archive found under {extract_root}. If "
                    f"imagery is also absent, the archive layout differs from "
                    f"what was expected -- inspect the tree before proceeding.",
                    logfile)
            for arc in inner:
                log(f"stage 2: extracting {arc.name} "
                    f"({arc.stat().st_size:,} bytes) ...", logfile)
                with zipfile.ZipFile(arc) as zf:
                    # `testzip()` decompresses every member, which doubles the
                    # cost of this stage. It is off by default because the outer
                    # archive's SHA-256 has already been verified against the
                    # published digest, and that digest covers the nested zip's
                    # bytes: if the container matches, the zip is byte-identical
                    # to the published one. `extractall()` additionally checks
                    # each member's CRC as it writes, so a corrupt member raises
                    # rather than passing silently. Use --deep-verify to run the
                    # full pre-pass anyway.
                    if args.deep_verify:
                        log("  deep verify: reading every member (this is slow)",
                            logfile)
                        bad = zf.testzip()
                        if bad is not None:
                            log(f"  FAIL: corrupt member in {arc.name}: {bad}",
                                logfile)
                            return 3
                        log("  deep verify: all members readable", logfile)
                    zf.extractall(extract_root)
                log(f"  extracted {arc.name}", logfile)

        # -- report what we actually got, rather than assuming --
        lay = discover_layout(extract_root)
        log(f"layout: image_dir={lay.image_dir} label_dir={lay.label_dir} "
            f"images={lay.n_image_files} labels={lay.n_label_files} "
            f"top_level={lay.top_level}", logfile)
        if not _has_imagery(extract_root):
            log("WARNING: no image/label directories found after extraction. Do "
                "NOT proceed to the inventory; inspect the extracted tree first. "
                "Both archive layers may need attention.", logfile)
            imagery_ok: bool | None = False
            n_images = n_labels = 0
        else:
            imagery_ok = True
            n_images, n_labels = lay.n_image_files, lay.n_label_files
            log(f"extraction verified: {n_images} images, "
                f"{n_labels} labels", logfile)
    else:
        # --no-extract: extraction was not requested, so imagery presence is
        # simply not verified here. PARTIAL, not PASS -- this step did not run
        # its check.
        extract_root = RAW_DIR / "OpenEarthMap"
        imagery_ok = None
        n_images = n_labels = 0

    log("=== DONE ===", logfile)

    # ---- gate output ------------------------------------------------------- #
    # Protocol section 16 requires every QC gate to emit PASS / PARTIAL / FAIL
    # *as code*; run_pipeline.py parses exactly these lines. Without them this
    # step was always reported as PARTIAL ("printed no [STATUS] line"), which
    # meant the pipeline's aggregate status could never reach PASS even when the
    # archive and both extraction layers were fully verified.
    print()
    print("[PHASE] Data acquisition and extraction")
    print(f"[STATUS] {'PASS' if imagery_ok else ('FAIL' if imagery_ok is False else 'PARTIAL')}")
    print(f"[COMPLETED] archive SHA-256 verified; "
          f"{n_images} images / {n_labels} labels present")
    print(f"[FILES] {extract_root}")
    if imagery_ok is False:
        print("[ISSUES] no image/label directories found after extraction; the "
              "inventory must not be built from this tree")
    elif imagery_ok is None:
        print("[ISSUES] --no-extract was given; imagery presence was not verified")

    return 1 if imagery_ok is False else 0


if __name__ == "__main__":
    sys.exit(main())

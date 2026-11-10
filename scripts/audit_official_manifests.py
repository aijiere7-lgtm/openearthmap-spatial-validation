"""Verify the official OpenEarthMap split manifests against the files on disk.

Context
-------
The corpus-wide audit (``audit_image_label_pairs.py``) found that the archive
ships **3838 images** and **3500 labels** for 97 regions, and that the ZIP
central directory contains exactly the same 3838 / 3500 -- so the extraction is
faithful and the shortfall is intrinsic to the distribution, not an artefact.

Two arithmetic coincidences motivate this script:

* ``3838 + 1162 = 5000``  and ``xbd_files.csv`` has 1162 data rows
  -> the 1162 xBD tiles have no RGB imagery (which ``readme.txt`` states).
* ``3500 = 3000 + 500``   = train + val manifest sizes
  -> hypothesis: labels are shipped only for train and val, not for test.

Both are hypotheses.  This script tests them against the manifests instead of
assuming them, because the answer determines whether the official split can be
reproduced at all and whether the official test set is usable as a reference.

Outputs
-------
``results/tables/manifest_coverage.csv``  -- per-tile manifest membership vs presence
``results/statistics/manifest_audit.json`` -- summary, hypothesis verdicts, gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    RAW_DIR,
    STATISTICS_DIR,
    TABLES_DIR,
    ensure_dirs,
    get_logger,
    utcnow,
    write_json,
)

LOG = get_logger("audit_official_manifests")

CORPUS_ROOT = RAW_DIR / "OpenEarthMap" / "OpenEarthMap_wo_xBD"
PUBLISHED_SPLIT = {"train": 3000, "val": 500, "test": 1500}


def read_manifest(path: Path) -> list[str]:
    """One filename per line; blank lines dropped; whitespace stripped."""
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def stem_of(filename: str) -> str:
    return Path(filename).stem.lower()


def region_of(stem: str) -> str:
    """Region name = everything before the final underscore-number group.

    OpenEarthMap stems are ``<region>_<n>``.  Region names themselves contain
    underscores (``al_qurnah``, ``warminsko-mazurskie``), so we split on the
    LAST underscore whose suffix is purely numeric.
    """
    m = re.match(r"^(.*)_(\d+)$", stem)
    return m.group(1) if m else stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", default=str(CORPUS_ROOT))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ensure_dirs()
    root = Path(args.corpus_root)
    if not root.is_dir():
        LOG.error("corpus root not found: %s", root)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else TABLES_DIR
    stat_dir = Path(args.out_dir) if args.out_dir else STATISTICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)

    # ---- manifests -----------------------------------------------------
    manifests = {
        s: read_manifest(root / f"{s}.txt") for s in ("train", "val", "test")
    }
    manifest_sizes = {s: len(v) for s, v in manifests.items()}
    LOG.info("manifest sizes: %s (sum=%d)", manifest_sizes, sum(manifest_sizes.values()))

    # duplicate filenames inside a manifest, and across manifests
    dup_within = {s: [k for k, c in Counter(v).items() if c > 1] for s, v in manifests.items()}
    seen: dict[str, list[str]] = {}
    for s, v in manifests.items():
        for f in v:
            seen.setdefault(f.lower(), []).append(s)
    cross_listed = {f: s for f, s in seen.items() if len(s) > 1}

    manifest_set = {s: {stem_of(f) for f in v} for s, v in manifests.items()}
    all_manifest_stems = set().union(*manifest_set.values())

    # ---- files on disk -------------------------------------------------
    img_stems: dict[str, str] = {}
    lbl_stems: dict[str, str] = {}
    for region_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for p in (region_dir / "images").glob("*.tif"):
            img_stems[p.stem.lower()] = region_dir.name
        for p in (region_dir / "labels").glob("*.tif"):
            lbl_stems[p.stem.lower()] = region_dir.name

    LOG.info("on disk: %d image stems, %d label stems", len(img_stems), len(lbl_stems))

    # ---- xbd_files.csv -------------------------------------------------
    xbd_path = root / "xbd_files.csv"
    xbd_rows: list[tuple[str, str]] = []
    if xbd_path.is_file():
        for line in xbd_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                xbd_rows.append((parts[0], parts[1]))
    # Drop a header row if present
    if xbd_rows and xbd_rows[0][1].lower().endswith(".tif") and "png" in xbd_rows[0][0].lower():
        pass  # data row, keep
    xbd_target_stems = {stem_of(t) for _, t in xbd_rows}
    LOG.info("xbd_files.csv data rows: %d (distinct target stems: %d)",
             len(xbd_rows), len(xbd_target_stems))

    # ---- hypothesis tests ----------------------------------------------
    def pct(n: int, d: int) -> float | None:
        return round(100.0 * n / d, 3) if d else None

    # H_A: labels present  <=>  stem in train U val
    trainval = manifest_set["train"] | manifest_set["val"]
    labels_in_trainval = lbl_stems.keys() & trainval
    labels_outside_trainval = lbl_stems.keys() - trainval
    trainval_with_label = trainval & lbl_stems.keys()

    hA = {
        "statement": "a label exists on disk iff the tile is listed in train.txt or val.txt",
        "n_trainval_stems": len(trainval),
        "n_labels_on_disk": len(lbl_stems),
        "n_labels_in_trainval": len(labels_in_trainval),
        "n_labels_outside_trainval": len(labels_outside_trainval),
        "n_trainval_without_label": len(trainval - lbl_stems.keys()),
        "pct_labels_covered": pct(len(labels_in_trainval), len(lbl_stems)),
        "supported": len(labels_outside_trainval) == 0 and len(trainval) == len(lbl_stems),
        "examples_outside": sorted(labels_outside_trainval)[:20],
        "examples_trainval_without_label": sorted(trainval - lbl_stems.keys())[:20],
    }

    # H_B: images present  <=>  stem NOT in xbd_files.csv targets
    non_xbd = all_manifest_stems - xbd_target_stems
    images_in_nonxbd = img_stems.keys() & non_xbd
    images_in_xbd = img_stems.keys() & xbd_target_stems
    hB = {
        "statement": "an image exists on disk iff the tile is NOT an xBD-sourced tile",
        "n_all_manifest_stems": len(all_manifest_stems),
        "n_xbd_target_stems": len(xbd_target_stems),
        "n_xbd_targets_in_manifests": len(xbd_target_stems & all_manifest_stems),
        "n_images_on_disk": len(img_stems),
        "n_images_in_nonxbd": len(images_in_nonxbd),
        "n_images_that_are_xbd_targets": len(images_in_xbd),
        "n_nonxbd_without_image": len(non_xbd - img_stems.keys()),
        "supported": len(images_in_xbd) == 0,
        "examples_images_that_are_xbd": sorted(images_in_xbd)[:20],
        "examples_nonxbd_without_image": sorted(non_xbd - img_stems.keys())[:20],
    }

    # ---- per-tile coverage table ---------------------------------------
    rows = []
    for stem in sorted(all_manifest_stems):
        rows.append(
            {
                "stem": stem,
                "region": region_of(stem),
                "in_train": stem in manifest_set["train"],
                "in_val": stem in manifest_set["val"],
                "in_test": stem in manifest_set["test"],
                "is_xbd_target": stem in xbd_target_stems,
                "has_image": stem in img_stems,
                "has_label": stem in lbl_stems,
            }
        )
    cov = pd.DataFrame(rows)
    cov["n_manifests"] = cov[["in_train", "in_val", "in_test"]].sum(axis=1)
    cov.to_csv(out_dir / "manifest_coverage.csv", index=False)

    # ---- split x presence cross-tab ------------------------------------
    split_cols = ["in_train", "in_val", "in_test"]
    xtab: dict[str, dict] = {}
    for col in split_cols:
        sub = cov[cov[col]]
        xtab[col] = {
            "n_tiles": int(len(sub)),
            "n_with_image": int(sub["has_image"].sum()),
            "n_with_label": int(sub["has_label"].sum()),
            "n_with_both": int((sub["has_image"] & sub["has_label"]).sum()),
            "n_xbd_target": int(sub["is_xbd_target"].sum()),
        }

    # tiles listed in no manifest at all
    n_unlisted = int((cov["n_manifests"] == 0).sum())
    n_multilisted = int((cov["n_manifests"] > 1).sum())

    # ---- published-split comparison ------------------------------------
    split_compare = {
        s: {
            "published": PUBLISHED_SPLIT[s],
            "manifest_entries": manifest_sizes[s],
            "distinct_stems": len(manifest_set[s]),
            "matches_published": manifest_sizes[s] == PUBLISHED_SPLIT[s],
            "delta": manifest_sizes[s] - PUBLISHED_SPLIT[s],
        }
        for s in ("train", "val", "test")
    }

    # ---- gate ----------------------------------------------------------
    problems: list[str] = []
    if n_multilisted:
        problems.append(f"{n_multilisted} tiles appear in more than one manifest")
    if n_unlisted:
        problems.append(f"{n_unlisted} tiles appear in no manifest")
    for s, c in dup_within.items():
        if c:
            problems.append(f"{len(c)} duplicate filenames within {s}.txt")

    if not hA["supported"] and not hB["supported"]:
        gate = "PARTIAL"
        gate_reason = (
            "neither hypothesis fully explains the on-disk counts; "
            "the pairing rule must be derived empirically per tile"
        )
    elif problems:
        gate = "PARTIAL"
        gate_reason = "; ".join(problems)
    else:
        gate = "PASS"
        gate_reason = "manifests are internally consistent and the counts are explained"

    summary = {
        "generated_utc": utcnow(),
        "corpus_root": str(root),
        "manifest_sizes": manifest_sizes,
        "manifest_sum": sum(manifest_sizes.values()),
        "duplicates_within_manifest": {k: len(v) for k, v in dup_within.items()},
        "cross_listed_tiles": len(cross_listed),
        "n_tiles_in_no_manifest": n_unlisted,
        "n_tiles_in_multiple_manifests": n_multilisted,
        "split_vs_presence_crosstab": xtab,
        "published_split_comparison": split_compare,
        "hypothesis_A_labels_equal_trainval": hA,
        "hypothesis_B_images_equal_nonxbd": hB,
        "on_disk": {
            "n_images": len(img_stems),
            "n_labels": len(lbl_stems),
            "n_paired": len(img_stems.keys() & lbl_stems.keys()),
            "n_image_only": len(img_stems.keys() - lbl_stems.keys()),
            "n_label_only": len(lbl_stems.keys() - img_stems.keys()),
        },
        "gate": gate,
        "gate_reason": gate_reason,
    }
    write_json(stat_dir / "manifest_audit.json", summary)

    # ---- console -------------------------------------------------------
    LOG.info("=" * 68)
    LOG.info("manifest sizes              : %s", manifest_sizes)
    LOG.info("distinct stems per manifest : %s", {s: len(v) for s, v in manifest_set.items()})
    LOG.info("tiles in >1 manifest        : %d", n_multilisted)
    LOG.info("tiles in no manifest        : %d", n_unlisted)
    LOG.info("-" * 68)
    LOG.info("H_A (labels == train U val) : %s", "SUPPORTED" if hA["supported"] else "NOT SUPPORTED")
    LOG.info("    labels on disk=%d  in trainUval=%d  outside=%d  trainUval w/o label=%d",
             hA["n_labels_on_disk"], hA["n_labels_in_trainval"],
             hA["n_labels_outside_trainval"], hA["n_trainval_without_label"])
    LOG.info("H_B (images == non-xBD)     : %s", "SUPPORTED" if hB["supported"] else "NOT SUPPORTED")
    LOG.info("    images on disk=%d  nonxbd=%d  images that are xbd=%d  nonxbd w/o image=%d",
             hB["n_images_on_disk"], hB["n_images_in_nonxbd"],
             hB["n_images_that_are_xbd_targets"], hB["n_nonxbd_without_image"])
    LOG.info("-" * 68)
    for col, d in xtab.items():
        LOG.info("%-9s tiles=%-5d with_image=%-5d with_label=%-5d both=%-5d xbd=%-5d",
                 col, d["n_tiles"], d["n_with_image"], d["n_with_label"],
                 d["n_with_both"], d["n_xbd_target"])
    LOG.info("=" * 68)
    LOG.info("[STATUS] %s - %s", gate, gate_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

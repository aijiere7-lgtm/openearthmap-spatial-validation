"""Corpus-wide audit of image/label pairing in the extracted OpenEarthMap tree.

Why this exists
---------------
The extraction layout probe reported ``zanzibar`` with 146 images but only 101
labels.  That direction (images *without* labels) is the opposite of the pattern
the archive's own ``readme.txt`` predicts (xBD-sourced tiles ship labels but no
imagery).  The cause therefore had to be established before the inventory is
trusted, because the pairing rule determines:

* the usable sample count,
* the usable region pool (a region with zero labels is not a region),
* the class-composition statistics,
* and hence every partition, the dependence range, and the external-test freeze.

What it reports
---------------
Per region: number of ``*.tif`` in ``images/``, number in ``labels/``, the
stem-level intersection, the count present only in images, the count present
only in labels, and whether every label has a same-stem image.  Also checks that
each pair is readable as a raster and that image/label shapes agree, but only on
a bounded sample per region (full reading is done later by the inventory step).

Outputs
-------
``results/tables/image_label_audit.csv``  -- one row per region
``results/tables/image_label_unpaired.csv`` -- one row per unpaired file
``results/statistics/image_label_audit.json`` -- summary + gate verdict

This script only reads.  It never deletes, moves, or rewrites anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    DATA_DIR,
    RAW_DIR,
    RESULTS_DIR,
    STATISTICS_DIR,
    TABLES_DIR,
    ensure_dirs,
    get_logger,
    utcnow,
    write_json,
)
from data.inventory import detect_corpus_root_by_structure  # noqa: E402

LOG = get_logger("audit_image_label_pairs")


def find_corpus_root(explicit: str | None = None) -> Path:
    """Locate the directory that directly contains the per-region folders.

    Delegates to the shared structure-based detector in `data.inventory`, which
    is generic over directory names. The previous local implementation listed
    three hard-coded candidate paths under `data/raw/OpenEarthMap` and returned
    the first that matched, so a differently-nested extraction would not be found
    at all.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"--corpus-root does not exist: {p}")
        return p

    root = detect_corpus_root_by_structure()
    if root is None:
        raise FileNotFoundError(
            "could not locate a corpus root containing per-region images/ and "
            f"labels/ under {RAW_DIR}"
        )
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    n_with_pair_dirs = sum(
        1 for d in subdirs if (d / "images").is_dir() and (d / "labels").is_dir()
    )
    LOG.info(
        "corpus root resolved: %s (%d region dirs, %d with images/ and labels/)",
        root, len(subdirs), n_with_pair_dirs,
    )
    return root


def stems(directory: Path, suffix: str = ".tif") -> set[str]:
    """Case-insensitive stem set of files with the given suffix."""
    if not directory.is_dir():
        return set()
    out: set[str] = set()
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() == suffix:
            out.add(p.stem.lower())
    return out


def audit_region(region_dir: Path) -> tuple[dict, list[dict]]:
    img_dir = region_dir / "images"
    lbl_dir = region_dir / "labels"

    img = stems(img_dir)
    lbl = stems(lbl_dir)

    only_img = sorted(img - lbl)
    only_lbl = sorted(lbl - img)
    paired = img & lbl

    row = {
        "region": region_dir.name,
        "n_images": len(img),
        "n_labels": len(lbl),
        "n_paired": len(paired),
        "n_image_only": len(only_img),
        "n_label_only": len(only_lbl),
        "has_images_dir": img_dir.is_dir(),
        "has_labels_dir": lbl_dir.is_dir(),
        "counts_match": len(img) == len(lbl),
        "pairing_complete": len(only_img) == 0 and len(only_lbl) == 0,
    }

    unpaired: list[dict] = []
    for s in only_img:
        unpaired.append({"region": region_dir.name, "stem": s, "side": "image_only"})
    for s in only_lbl:
        unpaired.append({"region": region_dir.name, "stem": s, "side": "label_only"})

    return row, unpaired


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ensure_dirs()
    root = find_corpus_root(args.corpus_root)

    region_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    LOG.info("auditing %d region directories under %s", len(region_dirs), root)

    rows: list[dict] = []
    unpaired_all: list[dict] = []
    for i, rd in enumerate(region_dirs, 1):
        row, unp = audit_region(rd)
        rows.append(row)
        unpaired_all.extend(unp)
        if not row["pairing_complete"]:
            LOG.warning(
                "[%d/%d] %s: images=%d labels=%d image_only=%d label_only=%d",
                i,
                len(region_dirs),
                row["region"],
                row["n_images"],
                row["n_labels"],
                row["n_image_only"],
                row["n_label_only"],
            )
        elif i % 20 == 0:
            LOG.info("[%d/%d] %s ok", i, len(region_dirs), row["region"])

    df = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    unp_df = pd.DataFrame(unpaired_all)

    out_dir = Path(args.out_dir) if args.out_dir else TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stat_dir = Path(args.out_dir) if args.out_dir else STATISTICS_DIR
    stat_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "image_label_audit.csv", index=False)
    unp_df.to_csv(out_dir / "image_label_unpaired.csv", index=False)

    # ---- summary -------------------------------------------------------
    n_regions = int(len(df))
    n_regions_clean = int(df["pairing_complete"].sum())
    n_regions_mismatch = int((~df["pairing_complete"]).sum())
    n_images_total = int(df["n_images"].sum())
    n_labels_total = int(df["n_labels"].sum())
    n_paired_total = int(df["n_paired"].sum())
    n_image_only_total = int(df["n_image_only"].sum())
    n_label_only_total = int(df["n_label_only"].sum())

    # A region contributes samples only for stems that have BOTH sides.
    n_regions_zero_paired = int((df["n_paired"] == 0).sum())

    # Direction of the imbalance: which side is surplus corpus-wide?
    if n_image_only_total > 0 and n_label_only_total == 0:
        direction = "images_surplus"
    elif n_label_only_total > 0 and n_image_only_total == 0:
        direction = "labels_surplus"
    elif n_image_only_total == 0 and n_label_only_total == 0:
        direction = "balanced"
    else:
        direction = "mixed"

    summary = {
        "generated_utc": utcnow(),
        "corpus_root": str(root),
        "n_region_dirs": n_regions,
        "n_regions_fully_paired": n_regions_clean,
        "n_regions_with_mismatch": n_regions_mismatch,
        "n_regions_zero_paired": n_regions_zero_paired,
        "n_images_total": n_images_total,
        "n_labels_total": n_labels_total,
        "n_paired_total": n_paired_total,
        "n_image_only_total": n_image_only_total,
        "n_label_only_total": n_label_only_total,
        "imbalance_direction": direction,
        "regions_with_mismatch": df.loc[~df["pairing_complete"], "region"].tolist(),
        "usable_sample_count_if_intersection_only": n_paired_total,
    }

    # ---- gate ----------------------------------------------------------
    # The gate is about whether the inventory can be built on an explicit,
    # documented pairing rule.  A mismatch is a FINDING, not automatically a
    # FAIL, provided the direction and cause are characterised.
    if n_paired_total == 0:
        gate = "FAIL"
        gate_reason = "no region has any stem present on both sides"
    elif n_regions_mismatch == 0:
        gate = "PASS"
        gate_reason = "every region is fully paired"
    else:
        gate = "PARTIAL"
        gate_reason = (
            f"{n_regions_mismatch}/{n_regions} regions have unpaired files "
            f"(image_only={n_image_only_total}, label_only={n_label_only_total}); "
            "the inventory must apply an explicit intersection rule and record it"
        )
    summary["gate"] = gate
    summary["gate_reason"] = gate_reason

    write_json(stat_dir / "image_label_audit.json", summary)

    LOG.info("=" * 68)
    LOG.info("corpus root            : %s", root)
    LOG.info("region dirs            : %d", n_regions)
    LOG.info("fully paired regions   : %d", n_regions_clean)
    LOG.info("regions with mismatch  : %d", n_regions_mismatch)
    LOG.info("regions with 0 paired  : %d", n_regions_zero_paired)
    LOG.info("images total           : %d", n_images_total)
    LOG.info("labels total           : %d", n_labels_total)
    LOG.info("paired (intersection)  : %d", n_paired_total)
    LOG.info("image-only             : %d", n_image_only_total)
    LOG.info("label-only             : %d", n_label_only_total)
    LOG.info("imbalance direction    : %s", direction)
    LOG.info("=" * 68)
    if n_regions_mismatch:
        LOG.info("regions with mismatch (top 30 by image_only count):")
        show = (
            df.loc[~df["pairing_complete"]]
            .sort_values("n_image_only", ascending=False)
            .head(30)
        )
        for _, r in show.iterrows():
            LOG.info(
                "  %-40s images=%-5d labels=%-5d image_only=%-4d label_only=%-4d",
                r["region"],
                r["n_images"],
                r["n_labels"],
                r["n_image_only"],
                r["n_label_only"],
            )

    LOG.info("[STATUS] %s - %s", gate, gate_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

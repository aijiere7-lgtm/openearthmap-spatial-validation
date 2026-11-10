"""Per-region availability table for the OpenEarthMap distribution on disk.

Why this exists
---------------
``audit_image_label_pairs.py`` and ``audit_official_manifests.py`` established
three facts about the release that this project downloaded:

1. The distribution ships **3838 images** and **3500 labels** for 97 regions.
2. Every one of the 3500 labels belongs to a tile listed in ``train.txt`` or
   ``val.txt``; **no test tile has a label on disk**.  This is official design:
   the Zenodo record and the upstream README both state that test-set
   performance is evaluated on a CodaLab leaderboard, i.e. the test labels are
   withheld from the public release.
3. Every image on disk belongs to a tile that is **not** xBD-sourced; the 1162
   xBD tiles ship labels but no RGB imagery, as ``readme.txt`` states.

Consequence: the set of tiles usable *for this study* is the intersection
``image AND label``, which is a strict subset of the 5000 published tiles.  The
region pool, the class-composition statistics, the dependence range and the
external-test freeze all have to be computed on that intersection, not on the
nominal 97-region / 5000-image corpus.

This script produces that intersection table and the resulting region pool.

Outputs
-------
``results/tables/region_availability.csv``  -- one row per region
``results/statistics/region_availability.json`` -- pool definition + gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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

LOG = get_logger("region_availability")

CORPUS_ROOT = RAW_DIR / "OpenEarthMap" / "OpenEarthMap_wo_xBD"

# A region must contribute at least this many usable tiles to take part in
# partitioning.  Rationale: with fewer tiles a region cannot host both a
# validation fold and a buffer, and leave-one-region-out becomes degenerate.
MIN_USABLE_TILES_PER_REGION = 4


def region_of(stem: str) -> str:
    m = re.match(r"^(.*)_(\d+)$", stem)
    return m.group(1) if m else stem


def read_manifest(path: Path) -> list[str]:
    return [
        s.strip()
        for s in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if s.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", default=str(CORPUS_ROOT))
    ap.add_argument("--min-usable", type=int, default=MIN_USABLE_TILES_PER_REGION)
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

    # ---- manifests -> tile table ----------------------------------------
    split_of: dict[str, str] = {}
    for s in ("train", "val", "test"):
        for f in read_manifest(root / f"{s}.txt"):
            split_of[Path(f).stem.lower()] = s

    # ---- xbd targets ----------------------------------------------------
    xbd_targets: set[str] = set()
    for line in (root / "xbd_files.csv").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            xbd_targets.add(Path(parts[1]).stem.lower())

    # ---- on-disk presence ------------------------------------------------
    img_stems: set[str] = set()
    lbl_stems: set[str] = set()
    for rd in sorted(d for d in root.iterdir() if d.is_dir()):
        img_stems |= {p.stem.lower() for p in (rd / "images").glob("*.tif")}
        lbl_stems |= {p.stem.lower() for p in (rd / "labels").glob("*.tif")}

    # ---- per-tile frame --------------------------------------------------
    rows = []
    for stem, split in split_of.items():
        rows.append(
            {
                "stem": stem,
                "region": region_of(stem),
                "split": split,
                "is_xbd": stem in xbd_targets,
                "has_image": stem in img_stems,
                "has_label": stem in lbl_stems,
            }
        )
    tiles = pd.DataFrame(rows)
    tiles["usable"] = tiles["has_image"] & tiles["has_label"]
    tiles.to_csv(out_dir / "tile_availability.csv", index=False)

    # ---- per-region aggregation -----------------------------------------
    g = tiles.groupby("region")
    reg = pd.DataFrame(
        {
            "n_tiles_manifest": g.size(),
            "n_train": g["split"].apply(lambda s: int((s == "train").sum())),
            "n_val": g["split"].apply(lambda s: int((s == "val").sum())),
            "n_test": g["split"].apply(lambda s: int((s == "test").sum())),
            "n_xbd": g["is_xbd"].sum().astype(int),
            "n_with_image": g["has_image"].sum().astype(int),
            "n_with_label": g["has_label"].sum().astype(int),
            "n_usable": g["usable"].sum().astype(int),
        }
    ).reset_index()

    # usable split composition (only train/val can be usable, by construction)
    usable = tiles[tiles["usable"]]
    us = usable.groupby("region")["split"].apply(
        lambda s: pd.Series(
            {"n_usable_train": int((s == "train").sum()), "n_usable_val": int((s == "val").sum())}
        )
    )
    us = us.unstack(fill_value=0).reset_index()
    reg = reg.merge(us, on="region", how="left")
    for c in ("n_usable_train", "n_usable_val"):
        if c not in reg.columns:
            reg[c] = 0
        reg[c] = reg[c].fillna(0).astype(int)

    reg["pct_of_manifest_usable"] = (
        100.0 * reg["n_usable"] / reg["n_tiles_manifest"]
    ).round(2)
    reg["eligible"] = reg["n_usable"] >= args.min_usable
    reg = reg.sort_values(["n_usable", "region"], ascending=[False, True]).reset_index(drop=True)

    # ---- pool summary ----------------------------------------------------
    n_regions_total = int(len(reg))
    eligible = reg[reg["eligible"]]
    n_eligible = int(len(eligible))
    n_ineligible = n_regions_total - n_eligible
    n_usable_tiles = int(reg["n_usable"].sum())
    n_usable_in_eligible = int(eligible["n_usable"].sum())

    # regions that are entirely xBD (no imagery at all)
    entirely_xbd = reg[(reg["n_with_image"] == 0)]["region"].tolist()
    # regions with imagery but no usable pair
    no_usable = reg[(reg["n_usable"] == 0) & (reg["n_with_image"] > 0)]["region"].tolist()
    # regions dropped purely by the min-usable threshold
    below_threshold = reg[(reg["n_usable"] > 0) & (~reg["eligible"])]["region"].tolist()

    n_test_tiles_with_label = int(
        ((tiles["split"] == "test") & tiles["has_label"]).sum()
    )

    pool = {
        "generated_utc": utcnow(),
        "corpus_root": str(root),
        "min_usable_tiles_per_region": args.min_usable,
        "n_regions_total": n_regions_total,
        "n_regions_eligible": n_eligible,
        "n_regions_ineligible": n_ineligible,
        "n_regions_entirely_xbd_no_imagery": len(entirely_xbd),
        "n_regions_with_imagery_but_zero_usable": len(no_usable),
        "n_regions_below_min_usable_threshold": len(below_threshold),
        "n_usable_tiles_total": n_usable_tiles,
        "n_usable_tiles_in_eligible_regions": n_usable_in_eligible,
        "n_usable_tiles_lost_to_threshold": n_usable_tiles - n_usable_in_eligible,
        "n_test_tiles_with_label_on_disk": n_test_tiles_with_label,
        "eligible_regions": eligible["region"].tolist(),
        "ineligible_regions": reg[~reg["eligible"]]["region"].tolist(),
        "entirely_xbd_regions": entirely_xbd,
        "regions_with_imagery_but_no_usable_pair": no_usable,
        "regions_below_threshold": below_threshold,
        "usable_tile_count_distribution": {
            "min": int(eligible["n_usable"].min()) if n_eligible else 0,
            "median": float(eligible["n_usable"].median()) if n_eligible else 0.0,
            "max": int(eligible["n_usable"].max()) if n_eligible else 0,
            "sum": n_usable_in_eligible,
        },
    }

    # ---- gate ------------------------------------------------------------
    if n_eligible < 2:
        gate = "FAIL"
        gate_reason = "fewer than two eligible regions: no geographically disjoint design is possible"
    elif n_test_tiles_with_label != 0:
        gate = "PARTIAL"
        gate_reason = (
            f"unexpected: {n_test_tiles_with_label} test tiles have labels on disk; "
            "the official-test-unusable premise would need re-checking"
        )
    elif n_eligible < 10:
        gate = "PARTIAL"
        gate_reason = f"only {n_eligible} eligible regions; leave-one-region-out will be fragile"
    else:
        gate = "PASS"
        gate_reason = (
            f"{n_eligible} eligible regions carrying {n_usable_in_eligible} usable tiles; "
            "test labels confirmed absent (official CodaLab-held design)"
        )
    pool["gate"] = gate
    pool["gate_reason"] = gate_reason

    write_json(stat_dir / "region_availability.json", pool)
    reg.to_csv(out_dir / "region_availability.csv", index=False)

    # ---- console ---------------------------------------------------------
    LOG.info("=" * 72)
    LOG.info("regions total                 : %d", n_regions_total)
    LOG.info("eligible regions (>=%d usable) : %d", args.min_usable, n_eligible)
    LOG.info("  entirely xBD, no imagery    : %d", len(entirely_xbd))
    LOG.info("  imagery but zero usable pair: %d", len(no_usable))
    LOG.info("  below min-usable threshold  : %d", len(below_threshold))
    LOG.info("usable tiles (image AND label): %d", n_usable_tiles)
    LOG.info("  inside eligible regions     : %d", n_usable_in_eligible)
    LOG.info("  lost to the threshold       : %d", n_usable_tiles - n_usable_in_eligible)
    LOG.info("test tiles carrying a label   : %d  (expected 0)", n_test_tiles_with_label)
    LOG.info("usable tiles per region       : min=%d median=%.0f max=%d",
             pool["usable_tile_count_distribution"]["min"],
             pool["usable_tile_count_distribution"]["median"],
             pool["usable_tile_count_distribution"]["max"])
    LOG.info("-" * 72)
    LOG.info("eligible regions and their usable tile counts:")
    for _, r in eligible.iterrows():
        LOG.info("  %-32s usable=%-4d (train=%-4d val=%-3d) xbd=%-4d pct_of_manifest=%.1f%%",
                 r["region"], r["n_usable"], r["n_usable_train"], r["n_usable_val"],
                 r["n_xbd"], r["pct_of_manifest_usable"])
    if len(reg[~reg["eligible"]]):
        LOG.info("-" * 72)
        LOG.info("ineligible regions:")
        for _, r in reg[~reg["eligible"]].iterrows():
            LOG.info("  %-32s usable=%-4d with_image=%-4d xbd=%-4d",
                     r["region"], r["n_usable"], r["n_with_image"], r["n_xbd"])
    LOG.info("=" * 72)
    LOG.info("[STATUS] %s - %s", gate, gate_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

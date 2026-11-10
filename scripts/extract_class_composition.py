#!/usr/bin/env python
"""
extract_class_composition.py — per-sample and per-region class pixel counts.

WHY THIS EXISTS
---------------
Two separate needs are met by one pass over the labels, so the labels are read
once rather than twice:

1. SPATIAL DEPENDENCE FEATURES. The distance-similarity curve (protocol section 7)
   needs a feature vector per sample. The land-cover class composition is the
   natural choice: it is the quantity the model is actually trained to predict,
   and its spatial autocorrelation is the phenomenon under study. Using image
   pixel statistics instead would describe the imagery, not the target.

2. A PENDING PREREQUISITE. `region_class_pixels.csv` (region x class pixel
   counts) is required to characterise the external test regions and to
   stratify region selection. It must be measured, never synthesised.

It also yields `dominant_class` per sample, which `random_folds` needs for
stratification.

QC VALUE: the global class shares are compared against the values published in
the dataset paper (Table 2). Substantial divergence would mean the retrieved
archive is not the dataset it claims to be, and the data QC gate must fail.
That comparison is the reason this is a QC artefact and not merely a cache.

Output: data/metadata/sample_class_composition.csv
        data/metadata/region_class_pixels.csv
        data/metadata/region_summary.csv
        results/tables/class_composition_global.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    CLASS_NAMES, LABEL_VALUE_TO_CLASS_INDEX, LABEL_VALUE_UNKNOWN, LOGS_DIR,
    METADATA_DIR, NUM_CLASSES, PUBLISHED_PIXEL_SHARE_PCT, RESULTS_DIR,
    ensure_dirs, get_logger, utcnow, write_json,
)
from data.inventory import (  # noqa: E402
    _read_label, resolve_inventory_path, resolve_inventory_root,
)

INVENTORY = METADATA_DIR / "dataset_inventory.csv"
OUT_SAMPLE = METADATA_DIR / "sample_class_composition.csv"
OUT_REGION = METADATA_DIR / "region_class_pixels.csv"
OUT_REGION_SUMMARY = METADATA_DIR / "region_summary.csv"
OUT_GLOBAL = RESULTS_DIR / "tables" / "class_composition_global.csv"


def autodetect_root(inventory_csv: Path, logger=None) -> tuple[Path | None, dict]:
    """Resolve the directory the inventory's relative paths are based on.

    Delegates to the shared evidence-based resolver in `data.inventory`. The
    previous local implementation scored candidates by `n_label_files` from
    `discover_layout`, which recurses -- so the outer `data/raw/OpenEarthMap` and
    the inner `data/raw/OpenEarthMap/OpenEarthMap_wo_xBD` both reported 3,500
    label files, the tie was kept by the first enumerated candidate, and the
    OUTER directory won. Every one of the 2,687 relative `label_path` values then
    failed to resolve. Path evidence cannot make that mistake: the outer
    directory resolves none of them.

    Returns (root, info) so the caller can persist the resolution method rather
    than presenting a guess as a fact.
    """
    root, info = resolve_inventory_root(inventory_csv=inventory_csv,
                                        logger=logger)
    if logger is not None:
        logger.info("corpus root method: %s", info.get("method"))
        for note in info.get("notes", []):
            logger.warning("root resolution: %s", note)
    return root, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(INVENTORY))
    ap.add_argument("--root", default=None,
                    help="root the inventory's relative paths are based on")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N samples (smoke testing)")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("class_composition",
                        LOGS_DIR / "extract_class_composition.log")
    logger.info("=" * 72)
    logger.info("CLASS COMPOSITION EXTRACTION  %s", utcnow())
    logger.info("=" * 72)

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        logger.error("inventory not found: %s -- run scripts/build_inventory.py first",
                     inv_path)
        return 2

    inv = pd.read_csv(inv_path, keep_default_na=False)
    logger.info("inventory rows: %d", len(inv))

    if args.root:
        root = Path(args.root)
        root_info: dict = {"method": "explicit", "root": str(root)}
    else:
        root, root_info = autodetect_root(inv_path, logger)
    if root is None:
        logger.error("could not locate the extracted imagery under data/; "
                     "pass --root explicitly")
        return 2
    logger.info("resolving relative paths against: %s", root)
    if not root.is_dir():
        logger.error("resolved root is not a directory: %s", root)
        return 2

    if "label_path" not in inv.columns:
        logger.error("inventory has no label_path column")
        return 2

    # Only rows with a readable label can contribute pixel counts.
    if "label_readable" in inv.columns:
        usable = inv[inv["label_readable"].astype(str).str.lower()
                     .isin({"true", "1", "yes"})].copy()
    else:
        usable = inv[inv["label_path"].astype(str).str.len() > 0].copy()
    if args.limit:
        usable = usable.head(args.limit)
    logger.info("samples with readable labels: %d", len(usable))

    px_cols = [f"px_class_{c}" for c in range(NUM_CLASSES)]
    sh_cols = [f"share_class_{c}" for c in range(NUM_CLASSES)]

    sample_rows: list[dict] = []
    region_px: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(NUM_CLASSES, dtype=np.int64))
    region_imgs: dict[str, int] = defaultdict(int)
    region_country: dict[str, str] = {}
    global_px = np.zeros(NUM_CLASSES, dtype=np.int64)
    n_missing = 0
    n_unreadable = 0
    # The residual "unknown" value (0) is a DOCUMENTED part of the encoding and
    # is expected in every tile -- see DECISIONS D-022. It is counted separately
    # from genuinely unexpected values, because lumping them together turns a
    # normal observation into a QC alarm and would train the reader to ignore
    # the warning that matters.
    n_unknown_px = 0
    unexpected_values: dict[int, int] = defaultdict(int)

    for i, row in enumerate(usable.itertuples(index=False), 1):
        rec = row._asdict() if hasattr(row, "_asdict") else dict(row)
        sid = str(rec.get("sample_id", ""))
        region = str(rec.get("region") or "UNKNOWN")
        country = str(rec.get("country") or "")

        lp = str(rec.get("label_path") or "")
        if not lp:
            n_missing += 1
            continue
        lab_path = resolve_inventory_path(root, lp)
        if not lab_path.exists():
            n_missing += 1
            if n_missing <= 3:
                logger.warning("label path does not resolve: %s -> %s",
                               lp, lab_path)
            continue

        arr = _read_label(lab_path)
        if arr is None:
            n_unreadable += 1
            continue
        if arr.ndim == 3:
            arr = arr[..., 0]

        # Count every observed value, then map raster values to class indices.
        # The raster is NOT encoded as class indices: 0 = "unknown",
        # 1 = Bareland, ..., 8 = buildings. Indexing by raster value directly
        # would report "unknown" as Bareland and discard Building entirely --
        # see DECISIONS D-022.
        vals, counts = np.unique(arr, return_counts=True)
        per_class = np.zeros(NUM_CLASSES, dtype=np.int64)
        for v, c in zip(vals.tolist(), counts.tolist()):
            iv = int(v)
            ci = LABEL_VALUE_TO_CLASS_INDEX.get(iv)
            if ci is not None:
                per_class[ci] += int(c)
            elif iv == LABEL_VALUE_UNKNOWN:
                n_unknown_px += int(c)
            else:
                unexpected_values[iv] += int(c)

        n_px = int(per_class.sum())
        if n_px == 0:
            continue

        shares = per_class / n_px
        dom = int(np.argmax(per_class))
        srow = {
            "sample_id": sid, "region": region, "country": country,
            "n_pixels": n_px,
            "n_classes_present": int((per_class > 0).sum()),
            "dominant_class": dom,
            "dominant_class_name": CLASS_NAMES[dom],
            "dominant_share": float(shares[dom]),
        }
        srow.update({c: int(v) for c, v in zip(px_cols, per_class.tolist())})
        srow.update({c: float(v) for c, v in zip(sh_cols, shares.tolist())})
        sample_rows.append(srow)

        region_px[region] += per_class
        region_imgs[region] += 1
        if country and region not in region_country:
            region_country[region] = country
        global_px += per_class

        if i % 250 == 0:
            logger.info("  ... %d/%d processed", i, len(usable))

    if not sample_rows:
        # Report WHICH stage consumed every row. "nothing written" alone was the
        # original failure message, and it did not distinguish an unresolvable
        # root from an unreadable raster from an all-zero class count -- the
        # three causes have completely different remedies.
        logger.error("no samples produced pixel counts; nothing written")
        logger.error("  root used            : %s", root)
        logger.error("  rows attempted       : %d", len(usable))
        logger.error("  label path missing   : %d", n_missing)
        logger.error("  label unreadable     : %d", n_unreadable)
        logger.error("  zero counted pixels  : %d", len(usable) - n_missing
                     - n_unreadable)
        if unexpected_values:
            logger.error("  unexpected label values : %s", dict(unexpected_values))
        if n_missing == len(usable):
            logger.error("  DIAGNOSIS: no label path resolved under the root. "
                         "The root is almost certainly one directory level off "
                         "(inventory paths are relative to the directory that "
                         "holds the per-region folders). Pass --root explicitly.")
        elif n_unreadable == len(usable):
            logger.error("  DIAGNOSIS: every label failed to open. Check that "
                         "rasterio can read the files and that the extraction "
                         "completed.")
        elif unexpected_values:
            logger.error("  DIAGNOSIS: label values were not in the expected "
                         "encoding (see DECISIONS D-022); every pixel was "
                         "discarded as unmapped.")
        return 1

    samples = pd.DataFrame(sample_rows)
    samples.to_csv(OUT_SAMPLE, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", OUT_SAMPLE, len(samples))

    # ---- per-region table -------------------------------------------------- #
    reg_rows: list[dict] = []
    for region, counts in sorted(region_px.items()):
        tot = int(counts.sum())
        if tot == 0:
            continue
        shares = counts / tot
        dom = int(np.argmax(counts))
        rrow = {
            "region": region,
            "country": region_country.get(region, ""),
            "n_images": int(region_imgs[region]),
            "n_pixels": tot,
            "dominant_class": dom,
            "dominant_class_name": CLASS_NAMES[dom],
            "dominant_share": float(shares[dom]),
            "n_classes_present": int((counts > 0).sum()),
        }
        rrow.update({c: int(v) for c, v in zip(px_cols, counts.tolist())})
        rrow.update({c: float(v) for c, v in zip(sh_cols, shares.tolist())})
        reg_rows.append(rrow)

    regions = pd.DataFrame(reg_rows)
    regions.to_csv(OUT_REGION, index=False, encoding="utf-8")
    logger.info("wrote %s (%d regions)", OUT_REGION, len(regions))

    regions[["region", "country", "n_images", "n_pixels", "dominant_class",
             "dominant_class_name", "dominant_share",
             "n_classes_present"]].to_csv(
        OUT_REGION_SUMMARY, index=False, encoding="utf-8")
    logger.info("wrote %s", OUT_REGION_SUMMARY)

    # ---- global shares vs published (independent QC cross-check) ---------- #
    tot = int(global_px.sum())
    measured = global_px / tot * 100.0
    g = pd.DataFrame({
        "class_code": list(range(NUM_CLASSES)),
        "class_name": CLASS_NAMES,
        "measured_pixels": global_px.tolist(),
        "measured_share_pct": measured.round(4).tolist(),
        "published_share_pct": [PUBLISHED_PIXEL_SHARE_PCT[c]
                                for c in range(NUM_CLASSES)],
    })
    g["abs_diff_pp"] = (g["measured_share_pct"]
                        - g["published_share_pct"]).abs().round(4)
    g.to_csv(OUT_GLOBAL, index=False, encoding="utf-8")
    logger.info("wrote %s", OUT_GLOBAL)

    max_dev = float(g["abs_diff_pp"].max())
    worst = g.loc[g["abs_diff_pp"].idxmax(), "class_name"]
    logger.info("total pixels counted: %s", f"{tot:,}")
    logger.info("largest deviation from published share: %.2f pp (%s)",
                max_dev, worst)

    summary = {
        "extracted_at_utc": utcnow(),
        "root": str(root),
        "root_resolution": root_info,
        "n_samples": len(samples),
        "n_regions": len(regions),
        "total_pixels": tot,
        "n_label_path_missing": n_missing,
        "n_label_unreadable": n_unreadable,
        "unknown_label_pixels": n_unknown_px,
        "unknown_pixel_share_pct": round(
            n_unknown_px / (tot + n_unknown_px) * 100.0, 4) if tot else None,
        "unexpected_label_pixels": dict(unexpected_values),
        "global_share_max_abs_diff_pp": max_dev,
        "global_share_worst_class": worst,
        "published_comparison_csv": str(OUT_GLOBAL),
    }
    write_json(METADATA_DIR / "class_composition_summary.json", summary)

    print()
    print("[PHASE] Class composition extraction")
    print("[STATUS] PASS" if max_dev < 3.0 else "[STATUS] PARTIAL")
    print(f"[COMPLETED] {len(samples)} samples, {len(regions)} regions, "
          f"{tot:,} pixels")
    print(f"[QC] max deviation from published class shares: {max_dev:.2f} pp "
          f"({worst})")
    print(f"[QC] residual 'unknown' pixels excluded from class shares: "
          f"{n_unknown_px:,} ({summary['unknown_pixel_share_pct']}%) -- expected, "
          f"documented in DECISIONS D-022")
    if unexpected_values:
        print(f"[QC] WARNING unexpected label values (not in the audited "
              f"encoding): {dict(unexpected_values)}")
    if n_missing or n_unreadable:
        print(f"[QC] missing={n_missing} unreadable={n_unreadable}")
    print(f"[FILES] {OUT_SAMPLE}")
    print(f"[FILES] {OUT_REGION}")
    print(f"[NEXT] region metadata + gazetteer, then spatial dependence")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

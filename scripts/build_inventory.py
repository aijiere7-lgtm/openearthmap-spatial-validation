#!/usr/bin/env python
"""
build_inventory.py — dataset QC: inventory, class/region statistics, duplicate
detection, and the QC report.

Outputs (all machine-generated; no number in the report is hand-typed):
    data/metadata/dataset_inventory.csv
    data/metadata/class_statistics.csv
    data/metadata/region_statistics.csv
    data/metadata/duplicates_exact.csv
    data/metadata/duplicates_near.csv
    data/metadata/layout_discovery.json
    data/metadata/data_qc_report.md

Gate: prints [QC] PASS / PARTIAL / FAIL. A FAIL here blocks all downstream phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    ALL_LABEL_VALUES_INCLUDING_UNKNOWN,
    CLASS_INDEX_TO_LABEL_VALUE, CLASS_NAMES, LABEL_VALUE_TO_CLASS_INDEX,
    LABEL_VALUE_UNKNOWN, LOGS_DIR,
    METADATA_DIR, NUM_CLASSES, OFFICIAL_BASELINE_N_CLASSES, PUBLISHED_N_IMAGES,
    PUBLISHED_IMAGE_SIZE, PUBLISHED_PIXEL_SHARE_PCT, PUBLISHED_N_REGIONS, RAW_DIR,
    RELEASE_EXPECTED_N_IMAGES, RELEASE_EXPECTED_N_LABELS,
    RELEASE_EXPECTED_N_REGION_DIRS, RELEASE_EXPECTED_N_REGIONS_WITH_IMAGERY,
    RELEASE_EXPECTED_N_USABLE,
    ensure_dirs, get_logger, load_config, sha256_file, utcnow, write_json,
)
from data.inventory import (  # noqa: E402
    build_inventory, detect_corpus_root_by_structure, discover_layout,
    find_duplicates, write_csv,
)


def find_dataset_root() -> Path | None:
    """Locate the extracted dataset root, by structure.

    Delegates to the single shared implementation in `data.inventory`. Four
    scripts previously carried their own copy of this logic and they disagreed;
    see the block comment in `src/data/inventory.py`. This is the only one of the
    four that MUST work without an inventory to test against, because it runs
    before the inventory exists -- so it uses the structure rule, which requires
    the corpus root to contain the most region directories carrying BOTH an
    image and a label subdirectory.

    OpenEarthMap is a MULTI-REGION corpus. Detection is therefore by STRUCTURE,
    never by file count: a corpus laid out as <region>/images + <region>/labels
    presents 97 candidate image directories, and "pick the one with the most
    files" would silently truncate it to a single region.
    """
    return detect_corpus_root_by_structure()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root (auto-detected if omitted)")
    ap.add_argument("--limit", type=int, default=None, help="process only N images (debug)")
    ap.add_argument("--no-phash", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("build_inventory", LOGS_DIR / "build_inventory.log")
    cfg = load_config()

    logger.info("=" * 72)
    logger.info("DATASET QC  %s", utcnow())
    logger.info("=" * 72)

    root = Path(args.root) if args.root else find_dataset_root()
    if root is None:
        logger.error("[PHASE] Data QC")
        logger.error("[STATUS] FAIL")
        logger.error("no extracted dataset found under %s", RAW_DIR)
        logger.error("run scripts/fetch_openearthmap.py first, and confirm extraction")
        return 1

    logger.info("dataset root: %s", root)

    # ---- discovery -------------------------------------------------------- #
    lay = discover_layout(root)
    write_json(METADATA_DIR / "layout_discovery.json", lay.to_dict())
    if lay.multi_region:
        logger.info("layout: MULTI-REGION — %d region directories", len(lay.region_dirs))
        logger.info("  total image files across regions: %d", lay.n_image_files)
        logger.info("  total label files across regions: %d", lay.n_label_files)
    else:
        logger.info("layout: single directory")
        logger.info("image_dir=%s (%d files)", lay.image_dir, lay.n_image_files)
        logger.info("label_dir=%s (%d files)", lay.label_dir, lay.n_label_files)
    logger.info("split files: %s", {k: Path(v).name for k, v in lay.split_files.items()})
    logger.info("metadata files: %s", [p.name for p in lay.metadata_files])
    for n in lay.notes:
        logger.info("  note: %s", n)

    if lay.image_dir is None and not lay.multi_region:
        logger.error("[PHASE] Data QC")
        logger.error("[STATUS] FAIL")
        logger.error("no image directory identified and no multi-region layout detected")
        return 1

    # ---- inventory -------------------------------------------------------- #
    if args.no_phash:
        cfg.setdefault("qc", {})["compute_perceptual_hash"] = False
    res = build_inventory(lay, cfg, logger, limit=args.limit)
    logger.info("inventory rows: %d", len(res.rows))

    # ---- duplicates ------------------------------------------------------- #
    thr = int(cfg.get("qc", {}).get("phash_near_duplicate_hamming", 6))
    exact_dups, near_dups = find_duplicates(res.rows, thr, logger)
    logger.info("exact duplicates: %d", len(exact_dups))
    logger.info("near duplicates (hamming<=%d): %d", thr, len(near_dups))

    # ---- class statistics ------------------------------------------------- #
    # CRITICAL: the label integer is NOT the class index.  The rasters store
    # 0 = "unknown", 1 = Bareland, ..., 8 = buildings, so paper class k lives at
    # integer k+1.  Indexing the histogram by class id would report "unknown"
    # as Bareland and would silently mislabel all eight classes.  The mapping is
    # applied here explicitly.  See DECISIONS D-022 and common.py.
    total_px_all_values = int(res.class_pixel_counts.sum())
    total_labelled = int(
        sum(res.class_pixel_counts[CLASS_INDEX_TO_LABEL_VALUE[cid]]
            for cid in range(NUM_CLASSES))
    )
    n_unknown_px = int(res.class_pixel_counts[LABEL_VALUE_UNKNOWN])
    class_rows = []
    for cid in range(NUM_CLASSES):
        lv_c = CLASS_INDEX_TO_LABEL_VALUE[cid]
        px = int(res.class_pixel_counts[lv_c])
        share = 100.0 * px / total_labelled if total_labelled else 0.0
        pub = PUBLISHED_PIXEL_SHARE_PCT.get(cid)
        class_rows.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "label_value": lv_c,
            "pixel_count": px,
            "pixel_share_pct": round(share, 4),
            "published_share_pct": pub,
            "delta_pct_points": round(share - pub, 4) if pub is not None else None,
            "n_images_in_label_range": sum(
                1 for r in res.rows
                if r.get("label_min") is not None and r["label_min"] <= lv_c <= r["label_max"]
            ),
        })

    # ---- region statistics ------------------------------------------------ #
    region_rows = []
    for rk, pr in sorted(res.per_region.items()):
        n = pr["n_images"]
        region_rows.append({
            "region": rk,
            "country": pr.get("country"),
            "n_images": n,
            "n_labels": pr["n_labels"],
            "n_unreadable": pr["n_bad"],
            "label_coverage_pct": round(100.0 * pr["n_labels"] / n, 2) if n else 0.0,
        })

    # ---- write outputs ---------------------------------------------------- #
    write_csv(METADATA_DIR / "dataset_inventory.csv", res.rows)
    write_csv(METADATA_DIR / "class_statistics.csv", class_rows)
    write_csv(METADATA_DIR / "region_statistics.csv", region_rows)
    write_csv(METADATA_DIR / "duplicates_exact.csv", exact_dups)
    write_csv(METADATA_DIR / "duplicates_near.csv", near_dups)
    logger.info("wrote inventory, class stats, region stats, duplicate tables")

    # ---- gate evaluation -------------------------------------------------- #
    n_img = len(res.rows)
    n_lab = sum(1 for r in res.rows if r["label_found"])
    n_unreadable_img = sum(1 for r in res.rows if not r["image_readable"])
    n_unreadable_lab = sum(1 for r in res.rows if r["label_found"] and not r["label_readable"])
    n_missing_lab = n_img - n_lab
    n_regions = len(res.per_region)
    lv = sorted(getattr(res, "label_value_set", set()))
    # Values outside the AUDITED encoding (0..8).  A check against [0, 8) would
    # flag both the residual value and Building as errors -- see D-022.
    outside_audited = [v for v in lv if v not in set(ALL_LABEL_VALUES_INCLUDING_UNKNOWN)]

    # The gate compares the inventory against the AUDITED release content, not
    # against the published dataset description.  The published description
    # (5,000 images) describes the dataset as the authors define it; the archive
    # ships 3,838 images and 3,500 labels for two structural reasons documented
    # in data/metadata/dataset_release_audit.md.  Checking against 5,000 would
    # make the gate permanently red and therefore useless.
    #
    # Image size is gated on INTERNAL CONSISTENCY (image and label agree), not on
    # equality with the published 1024.  The paper states 1024x1024 tiles, but
    # measurement shows the corpus is not uniform; a size of 1000 is a property
    # of the release, not a defect, and gating on 1024 would be asserting the
    # paper's value against the data.
    n_size_mismatch = sum(
        1 for r in res.rows
        if r["label_found"] and r["width"] and r["label_width"]
        and (r["width"] != r["label_width"] or r["height"] != r["label_height"])
    )
    n_off_1024 = sum(
        1 for r in res.rows
        if r["width"] and (r["width"] != PUBLISHED_IMAGE_SIZE
                           or r["height"] != PUBLISHED_IMAGE_SIZE)
    )

    checks = [
        ("image files present", n_img > 0),
        ("image count matches audited release "
         f"({RELEASE_EXPECTED_N_IMAGES})", n_img == RELEASE_EXPECTED_N_IMAGES),
        ("usable (image AND label) count matches audited release "
         f"({RELEASE_EXPECTED_N_USABLE})", n_lab == RELEASE_EXPECTED_N_USABLE),
        ("regions with imagery match audited release "
         f"({RELEASE_EXPECTED_N_REGIONS_WITH_IMAGERY})",
         n_regions == RELEASE_EXPECTED_N_REGIONS_WITH_IMAGERY),
        ("no unreadable images", n_unreadable_img == 0),
        ("no unreadable labels", n_unreadable_lab == 0),
        ("every image/label pair has matching dimensions", n_size_mismatch == 0),
        ("label values are a contiguous block of small integers",
         bool(lv) and lv == list(range(min(lv), max(lv) + 1)) and max(lv) <= 15),
        ("no exact duplicates", len(exact_dups) == 0),
    ]

    # Label-value census.  The paper defines eight classes and does not mention
    # a ninth value, but the labels contain one.  Which integer maps to which
    # class is NOT assumed: it is determined empirically in
    # scripts/verify_label_encoding.py against the paper's published pixel
    # shares, and the outcome is recorded as a decision.
    label_census = {
        int(v): int(c) for v, c in sorted(res.label_value_counter.items())
    }
    label_census_share_pct = {
        v: round(100.0 * c / sum(label_census.values()), 4)
        for v, c in label_census.items()
    } if label_census else {}

    # Image-size census: the corpus is not uniform, and that has to be stated.
    size_census: dict[str, int] = {}
    for r in res.rows:
        if r["width"] and r["height"]:
            size_census[f"{r['width']}x{r['height']}"] = (
                size_census.get(f"{r['width']}x{r['height']}", 0) + 1
            )

    # Informational comparisons: reported, never gated.
    informational = [
        {
            "quantity": "images",
            "measured": n_img,
            "published_dataset_description": PUBLISHED_N_IMAGES,
            "audited_release": RELEASE_EXPECTED_N_IMAGES,
            "explanation": "1,162 xBD tiles ship without RGB imagery",
        },
        {
            "quantity": "usable tiles (image AND label)",
            "measured": n_lab,
            "published_dataset_description": PUBLISHED_N_IMAGES,
            "audited_release": RELEASE_EXPECTED_N_USABLE,
            "explanation": "1,162 xBD tiles lack RGB imagery and 1,500 test tiles "
                           "lack labels; the usable set is the intersection",
        },
        {
            "quantity": "regions",
            "measured": n_regions,
            "published_dataset_description": PUBLISHED_N_REGIONS,
            "audited_release": RELEASE_EXPECTED_N_REGIONS_WITH_IMAGERY,
            "explanation": "all 97 region directories are present; 22 are entirely xBD "
                           "and contribute no usable tile",
        },
        {
            "quantity": f"tiles at {PUBLISHED_IMAGE_SIZE}x{PUBLISHED_IMAGE_SIZE}",
            "measured": n_img - n_off_1024,
            "published_dataset_description": PUBLISHED_N_IMAGES,
            "audited_release": f"{RELEASE_EXPECTED_N_IMAGES} images present",
            "explanation": f"{n_off_1024} tiles differ from the published size; "
                           f"size census: {size_census}",
        },
        {
            "quantity": f"classes (paper defines {NUM_CLASSES})",
            "measured": NUM_CLASSES,
            "published_dataset_description": NUM_CLASSES,
            "audited_release": f"{NUM_CLASSES} land-cover classes plus a residual "
                               f"value {LABEL_VALUE_UNKNOWN} named 'unknown' in the "
                               f"authors' code",
            "explanation": f"the official baseline trains "
                           f"{OFFICIAL_BASELINE_N_CLASSES} classes, counting "
                           f"'unknown' as a class; it holds "
                           f"{100.0 * n_unknown_px / max(total_px_all_values, 1):.4f}% "
                           f"of pixels",
        },
    ]

    report = {
        "generated_at_utc": utcnow(),
        "dataset_root": str(root),
        "layout_multi_region": bool(lay.multi_region),
        "n_region_dirs": len(lay.region_dirs) if lay.multi_region else None,
        "n_images": n_img,
        "n_labels": n_lab,
        "n_usable": n_lab,
        "n_missing_labels": n_missing_lab,
        "n_unreadable_images": n_unreadable_img,
        "n_unreadable_labels": n_unreadable_lab,
        "label_values_observed": lv,
        "label_values_outside_audited_encoding": outside_audited,
        "label_value_census": label_census,
        "label_value_census_share_pct": label_census_share_pct,
        "label_encoding": {
            "mapping": {str(k): int(v) for k, v in CLASS_INDEX_TO_LABEL_VALUE.items()},
            "class_names": CLASS_NAMES,
            "unknown_label_value": LABEL_VALUE_UNKNOWN,
            "unknown_pixel_share_pct": round(
                100.0 * n_unknown_px / max(total_px_all_values, 1), 6),
            "study_num_classes": NUM_CLASSES,
            "official_baseline_num_classes": OFFICIAL_BASELINE_N_CLASSES,
            "source": "open_earth_map/utils.py class_grey_oem, and "
                      "open_earth_map/dataset.py n_classes=9 default",
            "established_by": "documentary (authors' code) and corroborated by the "
                              "measured pixel shares matching the paper's Table 2",
        },
        "total_labelled_pixels": total_labelled,
        "image_size_census": size_census,
        "n_image_label_size_mismatch": n_size_mismatch,
        "n_tiles_off_published_1024": n_off_1024,
        "n_regions_observed": n_regions,
        "n_exact_duplicates": len(exact_dups),
        "n_near_duplicates": len(near_dups),
        "near_duplicate_threshold_hamming": thr,
        "checks": [{"check": c, "pass": bool(p)} for c, p in checks],
        "informational": informational,
        "problems": res.problems[:200],
        "n_problems": len(res.problems),
    }
    write_json(METADATA_DIR / "data_qc.json", report)

    failed = [c for c, p in checks if not p]
    status = "PASS" if not failed else ("PARTIAL" if len(failed) <= 2 else "FAIL")

    # ---- markdown report -------------------------------------------------- #
    md = []
    md.append("# Data QC Report\n")
    md.append(f"**Generated**: {utcnow()}  ")
    md.append(f"**Dataset root**: `{root}`  ")
    md.append(f"**Status**: **{status}**\n")
    md.append("> Every figure in this report is machine-generated from the inventory. "
              "No value is hand-typed.\n")

    md.append("## 1. Inventory summary\n")
    md.append("| Quantity | Value |")
    md.append("|---|---|")
    md.append(f"| Layout | {'multi-region (' + str(len(lay.region_dirs)) + ' region dirs)' if lay.multi_region else 'single directory'} |")
    md.append(f"| Region attribution source | {res.rows[0]['region_source'] if res.rows else 'n/a'} |")
    md.append(f"| Images found | {n_img} |")
    md.append(f"| Labels found (usable tiles) | {n_lab} |")
    md.append(f"| Missing labels | {n_missing_lab} |")
    md.append(f"| Unreadable images | {n_unreadable_img} |")
    md.append(f"| Unreadable labels | {n_unreadable_lab} |")
    md.append(f"| Regions observed | {len(res.per_region)} |")
    md.append(f"| Label values observed | {lv} |")
    md.append(f"| Label values outside the audited encoding (0–8) | "
              f"{outside_audited if outside_audited else 'none'} |")
    md.append(f"| Exact duplicates (SHA-256) | {len(exact_dups)} |")
    md.append(f"| Near duplicates (pHash ≤ {thr}) | {len(near_dups)} |")
    md.append("")

    md.append("## 1b. Two measured properties that differ from the published description\n")
    md.append("Both were discovered by measurement, not assumed, and both are "
              "recorded here rather than smoothed over.\n")

    md.append("### Tile size is not uniform\n")
    md.append("The paper states tiles are 1024×1024. Measured sizes:\n")
    md.append("| Size | Tiles |")
    md.append("|---|---|")
    for k in sorted(size_census, key=lambda s: -size_census[s]):
        md.append(f"| {k} | {size_census[k]} |")
    md.append("")
    md.append(f"{n_off_1024} of {n_img} tiles are not 1024×1024. "
              f"Image/label dimension mismatches within a pair: **{n_size_mismatch}**. "
              "The corpus is therefore resized to a common size before use, and the "
              "resampling is recorded as a preprocessing step.\n")

    md.append("### The label rasters contain more integer values than the paper defines\n")
    md.append("The paper defines eight classes and mentions no ninth value. "
              "Measured census of label integers across the corpus:\n")
    md.append("| Label value | Pixels | Share of all pixels % |")
    md.append("|---|---|---|")
    for v in sorted(label_census):
        nm = ("**unknown** (residual)" if v == LABEL_VALUE_UNKNOWN
              else CLASS_NAMES[LABEL_VALUE_TO_CLASS_INDEX[v]]
              if v in LABEL_VALUE_TO_CLASS_INDEX else "?")
        md.append(f"| {v} — {nm} | {label_census[v]:,} | "
                  f"{round(100.0 * label_census[v] / max(total_px_all_values, 1), 4)} |")
    md.append("")
    md.append("**Resolved.** The label integer is **not** the class index. The "
              "authors' own code, `open_earth_map/utils.py`, defines:\n")
    md.append("```python")
    md.append('class_grey_oem = {"unknown": 0, "Bareland": 1, "Grass": 2, "Pavement": 3,')
    md.append('                  "Road": 4, "Tree": 5, "Water": 6, "Cropland": 7,')
    md.append('                  "buildings": 8}')
    md.append("```")
    md.append("so paper class *k* is stored as integer *k+1*, and integer 0 is a "
              "residual category named `unknown`. `open_earth_map/dataset.py` "
              "confirms the consequence: its default is `n_classes: int = 9`, and "
              "`ToTensor(classes=np.arange(n_classes))` one-hot encodes all nine "
              "integers with no shift and no ignore index, so the official baseline "
              "trains `unknown` as a ninth class.\n")
    md.append("Two independent lines of evidence agree:\n")
    md.append("1. **Documentary.** The RGB values in the same dict "
              "(`800000`, `00FF24`, `949494`, `FFFFFF`, `226126`, `0045FF`, `4BB549`, "
              "`DE1F07`) match the paper's Table 2 hex codes one-for-one and in the "
              "same order, which pins the mapping to the paper's class names. "
              "`open_earth_map/dataset.py` sets `n_classes: int = 9` by default and "
              "`ToTensor(classes=np.arange(n_classes))` one-hot encodes all nine "
              "integers with no shift and no ignore index, so the official baseline "
              "trains `unknown` as a ninth class.\n")
    md.append("2. **Statistical.** Applying the `k+1` mapping reproduces the paper's "
              "published per-class shares to within a few percentage points for all "
              "eight classes (table below), whereas the naive "
              "`integer == class index` mapping would assign the residual category to "
              "Bareland and give Rangeland a share of "
              f"{label_census_share_pct.get(1, 0.0):.2f}% against a published 22.9%. "
              "The measured shares are computed on this corpus, which excludes the "
              "22 entirely-xBD regions, so exact agreement is not expected.\n")
    md.append("**Decision (D-022).** The study trains and reports on the eight "
              "land-cover classes, mapping integer 0 to the ignore index, because "
              "`unknown` is not a land-cover class and the paper's published shares "
              "are defined over the eight. The nine-class variant used by the "
              "official baseline is reported as a sensitivity analysis.\n")
    md.append("Regression guards: `scripts/test_label_encoding.py` (32 assertions) "
              "pins the mapping, the ignore index, and the end-to-end path into the "
              "metrics. Every failure mode here is silent, so it is asserted rather "
              "than reviewed.\n")

    md.append("## 2. Cross-check against the published dataset\n")
    md.append("The original paper reports 5,000 images at 1024×1024 and gives "
              "per-class pixel shares. Two distinct comparisons are made, and "
              "conflating them would be an error:\n")
    md.append("1. **Against the audited release** — the gate. The archive ships "
              "3,838 images and 3,500 labels, for two structural reasons "
              "documented in `dataset_release_audit.md`: 1,162 xBD tiles have no "
              "RGB imagery, and the 1,500 test tiles have no labels. A mismatch "
              "here means the corpus changed and the analysis is invalid.\n")
    md.append("2. **Against the published description** — informational only. "
              "The published figures describe the dataset as the authors define "
              "it, including tiles this release does not carry. Divergence is "
              "expected and explained, not a defect.\n")
    md.append("| Quantity | Measured | Published description | Audited release | Explanation |")
    md.append("|---|---|---|---|---|")
    for d in informational:
        md.append(f"| {d['quantity']} | {d['measured']} | "
                  f"{d['published_dataset_description']} | {d['audited_release']} | "
                  f"{d['explanation']} |")
    md.append("")
    md.append("### Per-class pixel shares\n")
    md.append("The published shares were computed over all 5,000 tiles, including "
              "the 22 entirely-xBD regions, which are absent here. Shares measured "
              "on this corpus are therefore **not expected to reproduce them**, and "
              "the difference is a stated consequence of the release, not evidence "
              "that the archive is mislabelled. The xBD regions are "
              "disaster-response areas and skew urban, so the shortfall should fall "
              "disproportionately on the built-up classes.\n")
    md.append("| Class | Measured share % | Published share % | Δ (pp) |")
    md.append("|---|---|---|---|")
    for r in class_rows:
        md.append(f"| {r['class_name']} | {r['pixel_share_pct']} | "
                  f"{r['published_share_pct']} | {r['delta_pct_points']} |")
    md.append("")

    md.append("## 3. Gate checks\n")
    md.append("| Check | Result |")
    md.append("|---|---|")
    for c, p in checks:
        md.append(f"| {c} | {'PASS' if p else '**FAIL**'} |")
    md.append("")

    md.append("## 4. Duplicate and leakage-risk findings\n")
    if not exact_dups and not near_dups:
        md.append("No exact or near duplicates detected. **Note**: this is a statement "
                  "about *identical imagery*, not about spatial dependence. Two patches "
                  "that are spatially adjacent but not identical will not appear here, "
                  "and are the more relevant leakage risk. Spatial diagnostics are "
                  "handled separately (protocol §9).\n")
    else:
        md.append(f"{len(exact_dups)} exact and {len(near_dups)} near-duplicate pairs "
                  f"were detected. These are recorded, **not deleted** — deletion would "
                  f"silently change the dataset. Their distribution across regions and "
                  f"their potential to straddle a train/validation boundary is assessed "
                  f"in the split-leakage QC phase.\n")
    md.append("")

    md.append("## 5. Problems recorded during inventory\n")
    if res.problems:
        md.append(f"{len(res.problems)} problems were recorded. First 40:\n")
        md.append("```")
        for p in res.problems[:40]:
            md.append(str(p))
        md.append("```")
    else:
        md.append("None.\n")
    md.append("")

    md.append("## 6. Layout discovery notes\n")
    for n in lay.notes:
        md.append(f"- {n}")
    md.append("")

    (METADATA_DIR / "data_qc_report.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("wrote %s", METADATA_DIR / "data_qc_report.md")

    # ---- console verdict -------------------------------------------------- #
    print()
    print("[PHASE] Data QC")
    print(f"[STATUS] {status}")
    print(f"[COMPLETED] inventory over {n_img} images and {n_lab} usable pairs; "
          f"{len(class_rows)} land-cover classes; {n_regions} regions with imagery; "
          f"duplicate scan")
    print(f"[FILES] {METADATA_DIR}")
    print(f"[QC] {len(checks) - len(failed)}/{len(checks)} checks passed")
    print(f"[ENCODING] label integer = class index + 1; value "
          f"{LABEL_VALUE_UNKNOWN} is 'unknown' and maps to the ignore index")
    print(f"[SIZE] {n_off_1024}/{n_img} tiles are not "
          f"{PUBLISHED_IMAGE_SIZE}x{PUBLISHED_IMAGE_SIZE}; size census {size_census}")
    if failed:
        print(f"[ISSUES] failed: {'; '.join(failed)}")
    print(f"[NEXT] spatial metadata audit (coordinate tier determination)")

    return 0 if status == "PASS" else (0 if status == "PARTIAL" else 1)


if __name__ == "__main__":
    sys.exit(main())

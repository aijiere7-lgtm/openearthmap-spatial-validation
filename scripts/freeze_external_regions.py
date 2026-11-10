#!/usr/bin/env python
"""
freeze_external_regions.py — select and freeze the external geographic test.

This is the study's load-bearing design element (research_protocol.md §6).
It runs ONCE, before any model is trained, and its output is thereafter
read-only.

Hard rules enforced here:
  * Whole regions only.
  * Selection is deterministic and seeded.
  * Selection is PERFORMANCE-BLIND: no model, no accuracy, no preliminary
    result is read by this script. Only geography, continent and class
    composition are used.
  * The frozen record is hashed, and the hash is re-verified by every
    downstream gate.

Refuses to overwrite an existing freeze unless --force is given, and records
the override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    CLASS_NAMES, COORD_SOURCES_OK_FOR_REGION_DISPERSION, EXTERNAL_REGIONS_JSON,
    LOGS_DIR, METADATA_DIR, NUM_CLASSES, ensure_dirs, freeze_content_sha256,
    freeze_external_regions, get_logger, load_config, read_json,
    sha256_json_canonical, utcnow, write_json,
)
from splits.strategies import select_external_regions  # noqa: E402


def load_region_table() -> pd.DataFrame | None:
    """Region-level table: region, country, [continent], lon, lat."""
    for name in ("region_metadata.csv", "regions.csv"):
        p = METADATA_DIR / name
        if p.exists():
            return pd.read_csv(p, keep_default_na=False)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing freeze (recorded as an override)")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("freeze_external", LOGS_DIR / "freeze_external_regions.log")
    cfg = load_config()
    logger.info("=" * 72)
    logger.info("EXTERNAL GEOGRAPHIC TEST FREEZE  %s", utcnow())
    logger.info("=" * 72)

    if EXTERNAL_REGIONS_JSON.exists() and not args.force:
        # An existing freeze is verified, not overwritten. The previous version
        # returned exit 1 here with a bare "already exists" error, which made a
        # routine re-run of the pipeline look like a failure: the External QC
        # gate asks whether the freeze EXISTS and is UNCHANGED (protocol section
        # 16), not whether it happened to be written during this invocation.
        #
        # Nothing is written on this path, so the no-overwrite guarantee that
        # motivated the guard is untouched. What IS added is a check the guard
        # was missing: if the file no longer matches its own recorded
        # content_sha256, every partition already derived from it is
        # unattributable, and that must fail loudly rather than pass as "frozen".
        existing = read_json(EXTERNAL_REGIONS_JSON)
        stored = str(existing.get("content_sha256") or "")
        recomputed = freeze_content_sha256(existing)
        names = freeze_external_regions(existing)

        logger.info("external_regions.json already exists and is FROZEN; "
                    "verifying it rather than overwriting.")
        logger.info("  frozen at  : %s", existing.get("frozen_at_utc"))
        logger.info("  coord tier : %s", existing.get("coordinate_tier"))
        logger.info("  n_external : %s", len(names))
        logger.info("  stored sha256     : %s", stored[:16])
        logger.info("  recomputed sha256 : %s", recomputed[:16])

        print()
        print("[PHASE] External geographic test freeze")
        if stored and stored == recomputed:
            logger.info("content_sha256 verified unchanged.")
            print(f"[STATUS] PASS — existing freeze reused and hash-verified; "
                  f"{len(names)} external regions; nothing rewritten")
            print(f"[COMPLETED] {len(names)} external regions from the "
                  f"{existing.get('frozen_at_utc')} freeze")
            print(f"[FILES] {EXTERNAL_REGIONS_JSON}")
            print(f"[QC] content_sha256 = {stored}")
            print("[NEXT] generate S0-S4 partitions")
            return 0

        logger.error("the existing freeze FAILS its own content hash.")
        logger.error("  stored     : %s", stored)
        logger.error("  recomputed : %s", recomputed)
        logger.error("Refusing to proceed: every partition already derived from "
                     "this file would be unattributable. Restore the file, or "
                     "re-run with --force if the freeze was created in error.")
        print("[STATUS] FAIL — existing freeze present but its content_sha256 "
              "does not match; not overwritten")
        return 1

    # A --force overwrite must leave a trace. The previous version of this script
    # DOCUMENTED that the override is recorded but never recorded it, so a
    # superseded freeze would have vanished without evidence -- and results
    # produced under it would have looked as if they belonged to the new one.
    superseded: dict | None = None
    if EXTERNAL_REGIONS_JSON.exists() and args.force:
        old = read_json(EXTERNAL_REGIONS_JSON)
        superseded = {
            "frozen_at_utc": old.get("frozen_at_utc"),
            "content_sha256": old.get("content_sha256"),
            "n_external": old.get("n_external"),
            "external_regions": (old.get("selection", {}) or {}).get(
                "external_regions"),
        }
        logger.warning("OVERWRITING an existing freeze with --force.")
        logger.warning("  superseded frozen_at: %s", superseded["frozen_at_utc"])
        logger.warning("  superseded sha256   : %s", superseded["content_sha256"])
        logger.warning("  Any result produced under the superseded freeze is "
                       "INVALID and must be re-produced.")

    # ---- inputs ----------------------------------------------------------- #
    region_table = load_region_table()
    if region_table is None:
        logger.error("no region-level metadata table found at %s", METADATA_DIR)
        logger.error("Expected columns: region, country, [continent], lon, lat")
        logger.error("This table must be derived from the dataset's own geographic "
                     "metadata. It cannot be synthesised.")
        print()
        print("[PHASE] External geographic test freeze")
        print("[STATUS] FAIL — region metadata unavailable")
        print("[ISSUES] a region table with coordinates is required to select "
              "geographically isolated regions")
        return 1

    class_path = METADATA_DIR / "region_class_pixels.csv"
    if not class_path.exists():
        logger.error("missing %s (region x class pixel counts)", class_path)
        logger.error("Produce it from the per-region label histograms in the QC "
                     "phase; it is required for the class-coverage constraint.")
        print()
        print("[PHASE] External geographic test freeze")
        print("[STATUS] FAIL — region x class pixel table unavailable")
        return 1

    class_share = pd.read_csv(class_path)
    audit = read_json(METADATA_DIR / "spatial_metadata_audit.json") \
        if (METADATA_DIR / "spatial_metadata_audit.json").exists() else {}
    tier = audit.get("tier", "unknown")

    ext_cfg = cfg.get("external_test", {})
    seed = int(ext_cfg.get("seed", 20260917))
    target = tuple(ext_cfg.get("target_n_regions", [12, 15]))
    min_px = int(ext_cfg.get("min_class_presence_px", 1000))
    min_tiles = ext_cfg.get("min_tiles_per_region", 4)
    min_tiles = int(min_tiles) if min_tiles is not None else None
    strat = ext_cfg.get("stratify_by", "continent")

    logger.info("tier=%s seed=%d target=%s min_class_px=%d min_tiles=%s "
                "stratify_by=%s",
                tier, seed, target, min_px, min_tiles, strat)

    # ---- selection -------------------------------------------------------- #
    # Region-level dispersion is a COARSER use of coordinates than estimating the
    # dependence range r: the distances are inter-region (10^3-10^4 km) and a
    # gazetteer centroid's positional error (10^1-10^2 km) is immaterial to the
    # ordering it produces. The two standards are named separately in common.py
    # so that neither can be silently substituted for the other.
    sel = select_external_regions(
        region_table=region_table,
        class_share=class_share,
        seed=seed,
        target_n=target,
        stratify_col=strat,
        min_class_px=min_px,
        maximise_min_distance=bool(ext_cfg.get("maximise_min_distance", True)),
        acceptable_coord_sources=COORD_SOURCES_OK_FOR_REGION_DISPERSION,
        min_tiles=min_tiles,
    )
    if "error" in sel:
        logger.error("selection failed: %s", sel["error"])
        cr = sel.get("coordinate_report") or {}
        for k in ("n_regions_in_table", "n_after_class_presence_filter",
                  "n_with_coordinates", "n_rejected_for_coord_source",
                  "n_usable_for_selection"):
            if k in cr:
                logger.error("  %s = %s", k, cr[k])
        print()
        print("[PHASE] External geographic test freeze")
        print("[STATUS] FAIL — %s" % sel["error"])
        if cr.get("coord_source_counts"):
            print(f"[ISSUES] coord_source counts: {cr['coord_source_counts']}")
        print("[NEXT] resolve the coordinate provenance, then re-run. Do NOT "
              "invent coordinates to unblock this step: a fabricated coordinate "
              "produces a selection that looks valid and is not.")
        return 1

    cr = sel.get("coordinate_report") or {}
    if cr.get("n_excluded_no_coordinates"):
        logger.warning(
            "%d region(s) were excluded from the external-test selection for "
            "lacking usable coordinates: %s",
            cr["n_excluded_no_coordinates"],
            cr.get("excluded_regions"))
        logger.warning("  %s", cr.get("exclusion_note"))

    record = {
        "frozen_at_utc": utcnow(),
        "protocol_version": cfg.get("project", {}).get("protocol_version", "1.0"),
        "dataset": cfg.get("dataset", {}).get("name"),
        "coordinate_tier": tier,
        "performance_blind": True,
        "selection": sel,
        "coordinate_standard": {
            "applied": "COORD_SOURCES_OK_FOR_REGION_DISPERSION",
            "rationale": (
                "Region-level dispersion is measured at inter-region distances "
                "(10^3-10^4 km), so a gazetteer centroid's positional error "
                "(10^1-10^2 km) is immaterial to the ordering. The same "
                "coordinates are NOT acceptable for estimating the dependence "
                "range r, whose scale is 1-10 km; that stricter standard is "
                "enforced separately in scripts/spatial_dependence.py."),
        },
        "class_names": {str(i): CLASS_NAMES[i] for i in range(NUM_CLASSES)},
        "freeze_rules": [
            "no patch from an external region may appear in any training or "
            "validation set, under any strategy, at any seed",
            "the external test may not be used for hyperparameter tuning, early "
            "stopping, architecture selection, threshold optimisation, "
            "augmentation selection or loss selection",
            "the external test is evaluated once, after all training and model "
            "selection is complete",
        ],
    }
    # `superseded_freeze` must be part of the HASHED content, not appended
    # afterwards. Downstream verification (scripts/build_splits.py) pops
    # `content_sha256`, re-hashes every remaining key, and refuses to proceed on
    # a mismatch -- so a field added after hashing would make the freeze look
    # tampered with and block the entire pipeline.
    if superseded is not None:
        record["superseded_freeze"] = superseded
    record["content_sha256"] = sha256_json_canonical(record)

    write_json(EXTERNAL_REGIONS_JSON, record)
    logger.info("froze %d external regions: %s",
                sel["n_external"], sel["external_regions"])
    logger.info("content_sha256 = %s", record["content_sha256"])

    # ---- report ----------------------------------------------------------- #
    md = [
        "# External Geographic Test — Frozen Record\n",
        f"**Frozen at**: {record['frozen_at_utc']}  ",
        f"**Content SHA-256**: `{record['content_sha256']}`  ",
        f"**Coordinate tier**: {tier}  ",
        f"**Performance-blind**: yes — no model or accuracy value entered the "
        f"selection\n",
        "## Selection\n",
        f"- Algorithm: {sel['algorithm']}",
        f"- Seed: {sel['seed']}",
        f"- Input pool: {sel['input_pool_size']} regions",
        f"- Class-presence filter: every class present with ≥ {min_px} px",
        f"- Minimum tiles per region: "
        f"{sel.get('coordinate_report', {}).get('min_tiles_applied')}",
        f"- External regions: **{sel['n_external']}**",
        f"- Development regions: **{sel['n_development']}**",
        "",
        "> The two pools are counted over different populations, deliberately. "
        "The **selection pool** is restricted to regions where every class "
        "reaches the minimum pixel count, because the external test must be able "
        "to score every class. The **development pool** is every other region "
        "that can support a spatial split; it is NOT restricted by the "
        "class-presence filter, since a region lacking one class is still usable "
        "for training and validation. Reading the development count off the "
        "selection pool understates it.",
        "",
        "### External regions\n",
    ]
    for r in sel["external_regions"]:
        md.append(f"- `{r}`")
    if cr:
        md += ["", "### Candidate pool and coordinate provenance\n",
               "| Quantity | Value |", "|---|---|",
               f"| Regions in the metadata table | {cr.get('n_regions_in_table')} |",
               f"| After the class-presence filter | {cr.get('n_after_class_presence_filter')} |",
               f"| Carrying coordinates | {cr.get('n_with_coordinates')} |",
               f"| Rejected for coordinate provenance | {cr.get('n_rejected_for_coord_source')} |",
               f"| Usable for selection | {cr.get('n_usable_for_selection')} |",
               f"| Excluded for lacking coordinates | {cr.get('n_excluded_no_coordinates')} |",
               ""]
        if cr.get("coord_source_counts"):
            md.append(f"Coordinate sources present: `{cr['coord_source_counts']}`\n")
        if cr.get("n_excluded_no_coordinates"):
            md.append(f"> {cr.get('exclusion_note')}\n")
            md.append("The excluded regions are listed in "
                      "`external_regions.json` under "
                      "`selection.coordinate_report.excluded_regions`. They are "
                      "neither external nor development, and the selection was "
                      "made over the reduced pool.\n")
        md.append("Coordinate standard applied: **region-level dispersion**, "
                  "which tolerates gazetteer centroids because the distances are "
                  "inter-region. This is deliberately a weaker standard than the "
                  "one applied to the dependence range `r`, where the same "
                  "positional error is the same size as the quantity being "
                  "measured.\n")
    md += ["", "## Class composition: external vs development\n",
           "| Class | External % | Development % | Divergence (pp) |",
           "|---|---|---|---|"]
    for c in range(NUM_CLASSES):
        md.append(f"| {CLASS_NAMES[c]} | "
                  f"{sel['class_share_external_pct'].get(c, 0):.3f} | "
                  f"{sel['class_share_development_pct'].get(c, 0):.3f} | "
                  f"{sel['class_share_divergence_pp'].get(c, 0):+.3f} |")
    md += ["",
           "Class-composition divergence is **reported, not corrected**. "
           "Re-selecting regions to reduce divergence would convert the external "
           "test into a tuned artefact.", "",
           "## Freeze rules (binding)\n"]
    for r in record["freeze_rules"]:
        md.append(f"- {r}")
    md.append("")
    (METADATA_DIR / "external_regions.md").write_text("\n".join(md), encoding="utf-8")

    print()
    print("[PHASE] External geographic test freeze")
    print("[STATUS] PASS")
    print(f"[COMPLETED] {sel['n_external']} external regions; development pool "
          f"{sel['n_development']} regions (selection pool was "
          f"{sel.get('n_selection_pool', sel['input_pool_size'])}, restricted by "
          f"the class-presence filter; the development pool is NOT)")
    print(f"[FILES] {EXTERNAL_REGIONS_JSON}")
    print(f"[QC] content_sha256 = {record['content_sha256']}")
    print("[NEXT] generate S0-S4 partitions")
    return 0


if __name__ == "__main__":
    sys.exit(main())

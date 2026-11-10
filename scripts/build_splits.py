#!/usr/bin/env python
"""
build_splits.py — construct S0-S4 partitions on the development pool, verify
the external freeze, and measure spatial independence.

Gate behaviour:
  * The frozen external-region record's content hash is re-verified before
    anything is written. A hash mismatch aborts the run.
  * External regions are EXCLUDED from every partition. A final check asserts
    that no external sample appears in any training or validation set, under
    any strategy, and fails the run if one does.

Outputs:
  data/splits/<strategy>[_<param>]_fold<k>.csv
  data/splits/all_splits_index.csv
  results/tables/spatial_independence_diagnostics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    EXTERNAL_REGIONS_JSON, LOGS_DIR, METADATA_DIR, SPLITS_DIR, TABLES_DIR,
    ensure_dirs, freeze_content_sha256, freeze_development_regions,
    freeze_external_regions, get_logger, load_config, read_json,
    utcnow, write_json,
)
from splits.diagnostics import (  # noqa: E402
    fold_train_sizes, fold_val_sizes, leakage_diagnostics,
    representative_train_size,
)
from splits.strategies import (  # noqa: E402
    apply_buffer, group_folds, leave_one_region_out,
    random_folds, spatial_block_folds,
)


def load_inventory() -> pd.DataFrame:
    p = METADATA_DIR / "dataset_inventory.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run scripts/build_inventory.py first")
    df = pd.read_csv(p, keep_default_na=False)
    for c in ("lon", "lat"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_partition(name: str, assign: pd.DataFrame, meta: pd.DataFrame) -> Path:
    a = assign.merge(meta, on="sample_id", how="left", suffixes=("", "_m"))
    keep = [c for c in ("sample_id", "region", "country", "lon", "lat",
                        "split", "fold", "eval_fold", "strategy", "block_id",
                        "distance_to_training", "distance_to_val_km",
                        "region_held_out") if c in a.columns]
    out = SPLITS_DIR / f"{name}.csv"
    a[keep].to_csv(out, index=False, encoding="utf-8")
    return out


def load_dependence_range(logger) -> tuple[float | None, dict]:
    """Read the estimated dependence range r and its provenance.

    The range is written by scripts/spatial_dependence.py into
    data/metadata/spatial_dependence.json, NESTED under the "range" key, together
    with the variogram window and a per-model breakdown. An earlier version of
    this script looked for a top-level "r_km" in a file named
    spatial_dependence_range.json, which is never written -- so S2/S3 silently
    fell back to study-extent scales and were NOT derived from r, defeating the
    whole point of the block-size ladder. The reader is now pinned to the actual
    artifact and reports what it found.

    Returns (r_km or None, info). `info["status"]` is one of:
      "identified"      -- r measured, attained its sill inside the window
      "lower_bound"     -- r fitted but the window ended before the sill
      "absent"          -- no artifact, or no numeric range inside it
    """
    info: dict = {"status": "absent", "source": None, "r_km": None,
                  "upper_km": None, "per_model_km": None, "note": None}
    rp = METADATA_DIR / "spatial_dependence.json"
    if not rp.exists():
        logger.warning("%s not found -- run scripts/spatial_dependence.py first. "
                       "S2/S3 cannot be derived from r.", rp.name)
        return None, info
    doc = read_json(rp)
    rng = doc.get("range") or {}
    r_km = rng.get("r_km")
    if r_km is None:
        info["note"] = rng.get("note") or "no numeric range in the artifact"
        logger.warning("spatial_dependence.json carries no numeric r (%s)",
                       info["note"])
        return None, info

    info["source"] = str(rp)
    info["r_km"] = float(r_km)
    info["tier"] = doc.get("tier")
    info["fitted_on"] = doc.get("range_fitted_on")
    info["model"] = rng.get("model")
    info["fit_r2"] = rng.get("fit_r2")
    info["stable"] = rng.get("stable")
    info["note"] = rng.get("note")
    info["upper_km"] = (doc.get("variogram_window") or {}).get("upper_km")
    per = doc.get("range_per_model_km") or {}
    info["per_model_km"] = per
    identified = doc.get("range_identified")
    info["range_identified"] = identified
    info["status"] = "identified" if identified else "lower_bound"

    # The block-size ladder is defined on r, so the interval of plausible r is
    # carried alongside it: block sizes inherit the uncertainty in r and the
    # manuscript must state the ladder as a sensitivity range, not a fixed set.
    # ONLY models that attained their sill inside the window may contribute --
    # admitting an extrapolated fit would widen the interval for the wrong
    # reason (a model that never plateaued, not genuine disagreement).
    flags = (rng.get("per_model") or {})
    info["models_identified"] = sorted(
        m for m, v in flags.items()
        if isinstance(v, dict) and v.get("identified_within_window"))
    info["models_excluded"] = sorted(
        m for m, v in flags.items()
        if isinstance(v, dict) and v.get("identified_within_window") is False)
    vals = [per[m] for m in info["models_identified"] if per.get(m)]
    if not vals:
        vals = [v for v in per.values() if v]
    info["interval_km"] = ([float(min(vals)), float(max(vals))] if vals else None)
    logger.info("dependence range r = %.3f km (%s, model=%s, R2=%s); status=%s",
                info["r_km"], info["fitted_on"], info["model"],
                (round(info["fit_r2"], 4) if info["fit_r2"] else None),
                info["status"])
    if info["interval_km"]:
        logger.info("range interval across identified models (%s): %.3f - %.3f "
                    "km", ",".join(info["models_identified"]) or "none",
                    info["interval_km"][0], info["interval_km"][1])
    if info["models_excluded"]:
        logger.info("models excluded from the interval as extrapolations: %s",
                    ",".join(info["models_excluded"]))
    if info["status"] == "lower_bound":
        logger.warning(
            "the fitted range did NOT attain its sill inside the variogram "
            "window (upper %.2f km): r = %.3f km is a LOWER BOUND. Block sizes "
            "derived from it are therefore lower bounds on the dependence scale "
            "and must be reported as such.", info["upper_km"] or float("nan"),
            info["r_km"])
    return info["r_km"], info


def attach_coordinates(inv: pd.DataFrame, logger) -> pd.DataFrame:
    """Attach lon/lat to the inventory, preferring the tile-level side table.

    The inventory has no geographic columns, so a spatial design cannot be built
    from it directly. Two coordinate sources exist and they are NOT equivalent:

      * data/metadata/sample_coordinates.csv -- one row per tile, derived from
        each GeoTIFF's own CRS and affine transform (coordinate tier T1). This is
        the only source at the scale the dependence range is measured on, and it
        is the one S2/S3 must use.
      * data/metadata/region_metadata.csv -- one row per region. Acceptable for
        between-region dispersion but NOT for estimating r (decision D-018), and
        it collapses every tile in a region to a single point, which would make
        a block design degenerate.

    Returns the inventory with `lon`/`lat` (and `coord_source` when available).
    Records the provenance in `inv.attrs["coord_source"]`.
    """
    inv = inv.copy()
    inv.attrs["coord_source"] = "none"

    sc = METADATA_DIR / "sample_coordinates.csv"
    if sc.exists():
        coords = pd.read_csv(sc, keep_default_na=False)
        need = {"sample_id", "lon", "lat"}
        if need <= set(coords.columns):
            cols = ["sample_id", "lon", "lat"] + [
                c for c in ("coord_source", "coord_tier", "crs_type")
                if c in coords.columns]
            c = coords[cols].copy()
            c["lon"] = pd.to_numeric(c["lon"], errors="coerce")
            c["lat"] = pd.to_numeric(c["lat"], errors="coerce")
            c = c.dropna(subset=["lon", "lat"]).drop_duplicates("sample_id")
            inv = inv.merge(c, on="sample_id", how="left")
            n = int(inv["lon"].notna().sum())
            logger.info("tile-level coordinates joined: %d of %d samples "
                        "(%s)", n, len(inv), sc.name)
            if n:
                inv.attrs["coord_source"] = "sample_coordinates.csv (tile-level, T1)"
                return inv
            logger.warning("%s yielded no usable coordinates; falling back to "
                           "region-level coordinates", sc.name)
        else:
            logger.warning("%s lacks the expected columns %s", sc.name, need)
    else:
        logger.warning("%s not found -- run "
                       "scripts/extract_sample_coordinates.py to get tile-level "
                       "coordinates", sc.name)

    rm = METADATA_DIR / "region_metadata.csv"
    if rm.exists():
        r = pd.read_csv(rm, keep_default_na=False)
        if {"region", "lon", "lat"} <= set(r.columns):
            r = r[["region", "lon", "lat"]].copy()
            r["region"] = r["region"].astype(str)
            r["lon"] = pd.to_numeric(r["lon"], errors="coerce")
            r["lat"] = pd.to_numeric(r["lat"], errors="coerce")
            r = r.dropna(subset=["lon", "lat"]).drop_duplicates("region")
            inv["region"] = inv["region"].astype(str)
            inv = inv.merge(r, on="region", how="left", suffixes=("", "_reg"))
            n = int(inv["lon"].notna().sum())
            logger.warning(
                "REGION-LEVEL coordinate fallback in use: %d of %d samples. "
                "Every tile in a region shares one point, so a spatial block "
                "design degenerates and the resulting range would not be "
                "measured at the scale the phenomenon occurs on.", n, len(inv))
            if n:
                inv.attrs["coord_source"] = ("region_metadata.csv "
                                             "(region-level, NOT valid for r)")
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("build_splits", LOGS_DIR / "build_splits.log")
    cfg = load_config()
    logger.info("=" * 72)
    logger.info("SPLIT CONSTRUCTION  %s", utcnow())
    logger.info("=" * 72)

    # ---- external freeze verification (must happen first) ----------------- #
    if not EXTERNAL_REGIONS_JSON.exists():
        logger.error("external_regions.json not found. The external geographic "
                     "test MUST be frozen before splits are built "
                     "(research_protocol.md §6).")
        print("\n[STATUS] FAIL — external test not frozen")
        return 1

    ext = read_json(EXTERNAL_REGIONS_JSON)
    stored = ext.get("content_sha256")
    recomputed = freeze_content_sha256(ext)
    if stored != recomputed:
        logger.error("external_regions.json hash mismatch — the freeze has been "
                     "modified. Refusing to build splits.")
        print("\n[STATUS] FAIL — external freeze integrity check failed")
        return 1
    logger.info("external freeze verified (sha256 %s...)", stored[:16])

    external_regions = set(freeze_external_regions(ext))
    logger.info("external regions: %d", len(external_regions))

    # ---- data ------------------------------------------------------------- #
    inv = load_inventory()
    logger.info("inventory: %d samples", len(inv))

    # COORDINATES ARE NOT IN THE INVENTORY. The inventory describes file
    # geometry and integrity (paths, sizes, hashes, readability) and carries no
    # geographic columns, so reading lon/lat from it always failed -- which
    # silently dropped S2 and S3, the two designs RQ2 depends on, while the
    # stage still reported PASS. Coordinates are joined from the tile-level side
    # table produced by scripts/extract_sample_coordinates.py, with the
    # region-level table as a documented fallback.
    inv = attach_coordinates(inv, logger)
    has_xy = ("lon" in inv.columns and "lat" in inv.columns
              and inv["lon"].notna().any() and inv["lat"].notna().any())
    audit_p = METADATA_DIR / "spatial_metadata_audit.json"
    tier = read_json(audit_p).get("tier") if audit_p.exists() else "unknown"
    logger.info("coordinate tier: %s (has_xy=%s, source=%s)", tier, has_xy,
                inv.attrs.get("coord_source", "none"))

    # Provenance of the S2/S3 block-size ladder, so the manuscript can state
    # exactly what the block sizes were derived from.
    result_ranges: list[dict] = []

    inv["region"] = inv["region"].astype(str)

    # DEVELOPMENT POOL = tiles that are actually trainable AND lie in a region
    # the freeze assigned to development. Three filters, each with a reason:
    #   1. the region must be in the frozen development list -- "all regions
    #      minus the external ones" also retains regions excluded for having too
    #      few tiles (kinshasa), which cannot support a leave-one-region-out fold
    #      and are not part of the pool the freeze describes;
    #   2. the tile must have a readable label -- 1,151 of 3,838 images have none
    #      (D-021), and a tile without a label cannot be trained or scored;
    #   3. the region must not be external.
    n_in = len(inv)
    if "label_readable" in inv.columns:
        usable = inv["label_readable"].astype(str).str.lower().isin(
            ("true", "1", "yes"))
    else:
        usable = inv["label_path"].astype(str).str.len() > 0
    dev_regions = set(freeze_development_regions(ext))
    if not dev_regions:
        logger.error("the freeze record carries no development_regions list; "
                     "refusing to guess the development pool from 'all minus "
                     "external', which is a different set.")
        print("\n[STATUS] FAIL — development pool cannot be reconstructed")
        return 1

    keep = (inv["region"].isin(dev_regions) & usable
            & ~inv["region"].isin(external_regions))
    dev = inv[keep].copy()
    ext_mask = inv["region"].isin(external_regions)
    logger.info("development pool: %d samples (from %d inventory rows)", len(dev),
                n_in)
    logger.info("  excluded as external region : %d samples",
                int(ext_mask.sum()))
    logger.info("  excluded as no readable label: %d samples",
                int((~usable).sum()))
    logger.info("  excluded as non-development region: %d samples",
                int((~inv["region"].isin(dev_regions)).sum()))
    if len(dev) == 0:
        logger.error("development pool is empty")
        return 1

    # dominant class per patch, for stratification
    if "dominant_class" not in dev.columns:
        dev["dominant_class"] = 0
        logger.warning("dominant_class absent from inventory; S0 stratification "
                       "degrades to unstratified. Regenerate the inventory with "
                       "per-patch dominant class for a valid S0.")

    split_cfg = cfg.get("splits", {})
    n_folds = int(split_cfg.get("n_folds", 5))
    seed = int(split_cfg.get("seed", 42))

    meta = dev[["sample_id", "region"] + (["country"] if "country" in dev.columns else [])
               + (["lon", "lat"] if has_xy else [])].copy()

    index_rows: list[dict] = []
    diag_rows: list[dict] = []
    partitions: dict[str, pd.DataFrame] = {}

    def register(name, assign, params, notes=None):
        assign = assign.copy()
        assign["strategy"] = name
        if "block_id" not in assign.columns:
            assign["block_id"] = None
        if "distance_to_training" not in assign.columns:
            assign["distance_to_training"] = np.nan
        partitions[name] = assign
        p = write_partition(name, assign, meta)
        d = leakage_diagnostics(assign, meta, phash=None, sha=None,
                                region_col="region")
        d["strategy"] = name
        d["params"] = str(params)
        diag_rows.append(d)
        index_rows.append({
            "strategy": name, "file": p.name, "n_rows": len(assign),
            "n_train_per_fold_median": representative_train_size(assign),
            "n_train_per_fold_min": (min(fold_train_sizes(assign).values())
                                     if fold_train_sizes(assign) else 0),
            "n_train_per_fold_max": (max(fold_train_sizes(assign).values())
                                     if fold_train_sizes(assign) else 0),
            "n_val_per_fold_median": (
                int(np.median(list(fold_val_sizes(assign).values())))
                if fold_val_sizes(assign) else 0),
            "n_folds": int((assign["eval_fold"] if "eval_fold" in assign.columns
                            else assign["fold"]).nunique()),
            "representation": ("expanded" if "eval_fold" in assign.columns
                               else "compact"),
            "params": str(params), "notes": "; ".join(notes or []),
        })
        logger.info("  %-28s train/fold=%d val/fold=%d folds=%d (%s)",
                    name, representative_train_size(assign),
                    (int(np.median(list(fold_val_sizes(assign).values())))
                     if fold_val_sizes(assign) else 0),
                    int((assign["eval_fold"] if "eval_fold" in assign.columns
                         else assign["fold"]).nunique()),
                    "expanded" if "eval_fold" in assign.columns else "compact")

    # ---- S0 random -------------------------------------------------------- #
    logger.info("building S0 (random patch)...")
    s0 = random_folds(dev, n_folds=n_folds, seed=seed)
    register("S0_random_patch", s0, {"n_folds": n_folds, "seed": seed})

    # ---- S1 group / scene ------------------------------------------------- #
    logger.info("building S1 (group/scene)...")
    group_col = next((c for c in ("scene", "source_dataset", "source")
                      if c in dev.columns and dev[c].astype(str).nunique() > 1), None)
    if group_col is None:
        # Fall back to region as the grouping unit, and say so.
        group_col = "region"
        logger.warning("no scene/source column found; S1 groups by region. This "
                       "makes S1 a coarser design than intended — recorded.")
    s1 = group_folds(dev, group_col=group_col, n_folds=n_folds, seed=seed)
    register("S1_group_scene", s1,
             {"group_col": group_col, "n_folds": n_folds, "seed": seed},
             notes=[f"grouping column: {group_col}"])

    # ---- S2 / S3 spatial block ------------------------------------------- #
    # S2 and S3 are the designs RQ2 rests on, so their absence is a stage
    # failure whenever coordinates are knowable -- not a limitation to note in
    # passing. An earlier version logged a warning, built only S0/S1/S4, and
    # still printed PASS.
    spatial_designs_built = 0
    if not has_xy and tier in ("T1", "T2"):
        logger.error(
            "coordinate tier is %s but no usable coordinates reached the "
            "development pool, so S2 (spatial block) and S3 (buffered block) "
            "cannot be built. These are the designs RQ2 depends on; refusing to "
            "report PASS. Check data/metadata/sample_coordinates.csv.", tier)
        print()
        print("[PHASE] Split construction and leakage QC")
        print(f"[STATUS] FAIL — S2/S3 cannot be built at coordinate tier {tier}")
        print("[ISSUES] no coordinates joined onto the development pool")
        return 4
    if has_xy:
        r_km, r_info = load_dependence_range(logger)
        mults = split_cfg.get("block_size_mults") or [0.5, 1.0, 2.0]

        if r_km is None:
            # No measured range. The protocol allows a configured fallback only
            # if it is stated explicitly; substituting study-extent scales
            # silently would let the manuscript describe S2/S3 as
            # "derived from the dependence range" when they are not.
            fb = split_cfg.get("block_size_fallback_m")
            if not fb:
                logger.error(
                    "no dependence range available AND "
                    "splits.block_size_fallback_m is not configured. Refusing to "
                    "invent block sizes from the study extent: that would make "
                    "S2/S3 look like they were derived from r when they were "
                    "not. Either produce the range or set the fallback "
                    "explicitly.")
                print()
                print("[PHASE] Split construction")
                print("[STATUS] FAIL — S2/S3 block sizes cannot be justified")
                print(f"[ISSUES] {r_info.get('note') or 'no numeric range'}")
                return 3
            scales = [float(fb) * m for m in mults]
            labels = [f"fallback_{m}x{mults[0]}r" for m in mults]
            logger.warning("using the CONFIGURED fallback block size %.0f m "
                           "(block sizes are NOT derived from r)", float(fb))
        else:
            scales = [m * r_km for m in mults]
            labels = [f"{m:g}r" for m in mults]

        range_provenance = {
            "derived_from_r": r_km is not None,
            "r_km": r_km,
            "r_status": r_info.get("status"),
            "r_interval_km": r_info.get("interval_km"),
            "variogram_window_upper_km": r_info.get("upper_km"),
            "block_size_mults": list(mults),
        }
        result_ranges.append(range_provenance)

        for lab, bs in zip(labels, scales):
            if not np.isfinite(bs) or bs <= 0:
                continue
            logger.info("building S2 block=%s (%.2f km)...", lab, bs)
            s2 = spatial_block_folds(dev, n_folds=n_folds, seed=seed,
                                     block_size_km=bs)
            register(f"S2_spatial_block_{lab}", s2,
                     {"block_size_km": round(bs, 4), "n_folds": n_folds,
                      "seed": seed, "derived_from_r": r_km is not None,
                      "r_km": (round(r_km, 4) if r_km else None),
                      "r_status": r_info.get("status")})

            logger.info("building S3 buffered block=%s buffer=%.2f km...", lab, bs)
            base = spatial_block_folds(dev, n_folds=n_folds, seed=seed,
                                       block_size_km=bs)
            # Align coordinates to `base` row order explicitly.
            coords = (dev[["sample_id", "lon", "lat"]]
                      .set_index("sample_id")
                      .reindex(base["sample_id"])
                      .reset_index())
            # EXPANDED representation: buffering excludes a sample from the
            # training set of ONE fold only, so a single `split` column cannot
            # express it. One row per (sample, eval_fold).
            buffered = []
            for k in range(n_folds):
                b = apply_buffer(base, coords, buffer_km=bs, val_fold=k).copy()
                b["eval_fold"] = k
                b["split"] = np.where(
                    b["fold"] == k, "val",
                    np.where(b["split"] == "excluded_buffer",
                             "excluded_buffer", "train"))
                buffered.append(b)
            s3 = pd.concat(buffered, ignore_index=True)
            register(f"S3_buffered_{lab}", s3,
                     {"block_size_km": round(bs, 4), "buffer_km": round(bs, 4),
                      "n_folds": n_folds, "seed": seed,
                      "representation": "expanded (eval_fold)",
                      "derived_from_r": r_km is not None,
                      "r_km": (round(r_km, 4) if r_km else None),
                      "r_status": r_info.get("status")})
    else:
        logger.warning("no coordinates: S2 and S3 cannot be constructed as "
                       "spatial designs. Recorded as a limitation; grouping-level "
                       "sensitivity (S1/S4) substitutes.")
        result_ranges.append({"derived_from_r": False,
                              "r_status": "unavailable",
                              "reason": "no coordinates on the development pool"})

    # ---- S4 leave-one-region-out ------------------------------------------ #
    logger.info("building S4 (leave-one-region-out)...")
    s4 = leave_one_region_out(dev, region_col="region")
    register("S4_leave_one_region_out", s4,
             {"n_regions": int(s4["fold"].nunique())})

    # ---- matched sample size ---------------------------------------------- #
    if bool(split_cfg.get("matched_sample_size", {}).get("enabled", True)):
        # Sizes MUST come from the representation-aware helper. The compact form
        # marks every sample `split="val"`, so counting the literal train label
        # returns 0 -- which would make `n2 >= n1` true and silently skip every
        # repetition, disabling a protocol-required control without any error.
        n1 = representative_train_size(partitions["S0_random_patch"])
        reps = int(split_cfg.get("matched_sample_size", {}).get("n_repeats", 5))
        logger.info("matched-sample-size control: random reference train/fold = %d",
                    n1)
        if n1 <= 0:
            logger.error("random split reports 0 training samples per fold; the "
                         "matched-sample-size control cannot be built. This "
                         "indicates a partition-representation problem, not an "
                         "absence of data.")
        for name, part in list(partitions.items()):
            if not name.startswith(("S2_", "S3_")):
                continue
            n2 = representative_train_size(part)
            if n2 <= 0:
                logger.warning("  %s: per-fold train size is 0; matched control "
                               "skipped for this design", name)
                continue
            if n2 >= n1:
                logger.info("  %s: train/fold %d >= random %d; no subsampling "
                            "required", name, n2, n1)
                continue
            for rep in range(reps):
                m = random_folds(dev, n_folds=n_folds, seed=seed + 1000 + rep,
                                 n_train_target=n2)
                register(f"MATCHED_{name}_N{n2}_rep{rep}", m,
                         {"target_n_train": n2, "matched_to": name,
                          "repeat": rep, "seed": seed + 1000 + rep})
            logger.info("  matched %s: random split capped at n_train=%d over %d "
                        "repetitions", name, n2, reps)

    # ---- leakage / independence table ------------------------------------- #
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(TABLES_DIR / "spatial_independence_diagnostics.csv",
                index=False, encoding="utf-8")
    pd.DataFrame(index_rows).to_csv(SPLITS_DIR / "all_splits_index.csv",
                                    index=False, encoding="utf-8")

    # ---- EXTERNAL QC: no external sample in any partition ----------------- #
    ext_ids = set(inv.loc[ext_mask, "sample_id"])
    violations: list[str] = []
    for name, part in partitions.items():
        used = set(part.loc[part["split"].isin(["train", "val"]), "sample_id"])
        bad = used & ext_ids
        if bad:
            violations.append(f"{name}: {len(bad)} external samples used")

    # ---- disjointness QC -------------------------------------------------- #
    # The grouping key must be the EVALUATION fold, not the assignment fold. S3
    # is stored in the EXPANDED representation: one row per (sample, eval_fold),
    # so a sample carries the same `fold` value in every row while its `split`
    # changes with `eval_fold`. Grouping by `fold` therefore puts each sample in
    # both the train and the val set of its own group and reports a 100% overlap
    # that does not exist. Verified on the built design: grouping by `eval_fold`
    # gives an overlap of exactly 0.
    disjoint_fail: list[str] = []
    for name, part in partitions.items():
        expanded = ("eval_fold" in part.columns
                    and part["eval_fold"].notna().any())
        key = "eval_fold" if expanded else "fold"
        for k in sorted(part[key].unique()):
            a = part[part[key] == k]
            tr = set(a.loc[a["split"] == "train", "sample_id"])
            va = set(a.loc[a["split"] == "val", "sample_id"])
            if tr & va:
                disjoint_fail.append(
                    f"{name} {key} {k}: {len(tr & va)} overlap")

    status = "PASS"
    if violations or disjoint_fail:
        status = "FAIL"

    write_json(METADATA_DIR / "split_qc.json", {
        "generated_at_utc": utcnow(),
        "external_freeze_sha256": stored,
        "coordinate_tier": tier,
        "n_development_samples": int(len(dev)),
        "n_external_samples_excluded": int(ext_mask.sum()),
        "n_partitions": len(partitions),
        "block_size_provenance": result_ranges,
        "external_leakage_violations": violations,
        "train_val_disjointness_failures": disjoint_fail,
        "status": status,
    })

    print()
    print("[PHASE] Split construction and leakage QC")
    print(f"[STATUS] {status}")
    print(f"[COMPLETED] {len(partitions)} partitions over "
          f"{len(dev)} development samples")
    print(f"[FILES] {SPLITS_DIR}")
    print(f"[QC] external leakage violations: {len(violations)}; "
          f"disjointness failures: {len(disjoint_fail)}")
    if violations or disjoint_fail:
        for v in violations[:10]:
            print(f"[ISSUES] {v}")
        for v in disjoint_fail[:10]:
            print(f"[ISSUES] {v}")
    print("[NEXT] real-data smoke test, then the experiment grid")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

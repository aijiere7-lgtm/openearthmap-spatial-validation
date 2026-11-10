#!/usr/bin/env python
"""verify_split_validity.py — audit the partitions that the grid actually used.

WHY THIS EXISTS
---------------
`research_protocol.md` §9 says a strategy's *name* is never evidence of its
spatial independence. Every independence claim in the manuscript must come from
a measurement on the partition files, not from the fact that a partition is
called "buffered" or "leave-one-region-out".

`src/splits/diagnostics.py` already measures that, and `build_splits.py` gates
the construction. This script is a SECOND, INDEPENDENT audit, written for the
final quality-control pass, for three reasons:

1. It recomputes distances with a different implementation (a `BallTree` on
   unit-sphere coordinates) than the one the splits were built with
   (`nearest_distance_km`, a chunked haversine). Two implementations that agree
   are evidence; one implementation re-run is not.
2. It checks things the construction gate did not: how the validation fold's
   class composition compares with the external test's, whether tiles can
   physically overlap across the train/validation boundary, and whether every
   registry row's `n_train`/`n_val` matches the partition it claims to have run.
3. It re-derives the headline corpus numbers from the artefacts rather than
   quoting them, including the external test's 12 regions / 470 tiles.

It writes, and never edits, the partitions.

Run:  python scripts/verify_split_validity.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from common import METADATA_DIR, SPLITS_DIR, get_logger, read_json, write_json  # noqa: E402
from splits.diagnostics import fold_frames, _fold_partitions  # noqa: E402

LOGS_DIR = PROJ / "logs"
RESULTS = PROJ / "results"
TABLES = RESULTS / "tables"
PIPELINE = RESULTS / "pipeline"
REGISTRY = PROJ / "experiments" / "experiments_registry.csv"

EARTH_RADIUS_KM = 6371.0088

# Distances at which "is there a training tile this close?" is asked. 0.5 km is
# the 0.05 km adjacency threshold corrected for the measured tile size (a tile
# spans 0.20-0.51 km, so 0.05 km could not fire at all: D-050). The others are
# the dependence range and its half and double, which are the block sizes the
# design actually uses.
DISTANCE_THRESHOLDS_KM = [0.5, 1.0, 2.0]


def balltree_nearest_km(tree_coords: np.ndarray, query_coords: np.ndarray) -> np.ndarray:
    """Nearest-neighbour great-circle distance in km, independent implementation.

    `BallTree(metric="haversine")` takes (lat, lon) in RADIANS and returns
    distances on the unit sphere, i.e. radians of arc. This is a different code
    path from `splits.strategies.nearest_distance_km`, which chunks an explicit
    haversine; agreement between them is what makes the numbers evidence.
    """
    from sklearn.neighbors import BallTree

    if len(tree_coords) == 0 or len(query_coords) == 0:
        return np.full(len(query_coords), np.nan)
    tree = BallTree(np.radians(tree_coords), metric="haversine")
    d, _ = tree.query(np.radians(query_coords), k=1)
    return d[:, 0] * EARTH_RADIUS_KM


def main() -> int:
    log = get_logger("verify_split_validity", LOGS_DIR / "verify_split_validity.log")
    TABLES.mkdir(parents=True, exist_ok=True)
    PIPELINE.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SPLITS_DIR / "all_splits_index.csv")
    coords = pd.read_csv(METADATA_DIR / "sample_coordinates.csv")
    inventory = pd.read_csv(METADATA_DIR / "dataset_inventory.csv")
    sample_class = pd.read_csv(METADATA_DIR / "sample_class_composition.csv")
    region_class = pd.read_csv(METADATA_DIR / "region_class_pixels.csv")
    ext = read_json(METADATA_DIR / "external_regions.json")
    spatial = read_json(METADATA_DIR / "spatial_dependence.json")
    split_qc = read_json(METADATA_DIR / "split_qc.json")

    r_km = float(spatial["range"]["r_km"])
    # Two artefacts carry an interval for r and they do not agree; the splits use
    # the point estimate above, so the disagreement does not affect any partition.
    # It is recorded rather than smoothed over -- see the audit report.
    r_ci_here = spatial["range"].get("r_ci_km")
    r_ci_splits = split_qc["block_size_provenance"][0].get("r_interval_km")
    ext_regions = set(ext["selection"]["external_regions"])
    dev_regions = set(ext["selection"]["development_regions"])
    freeze_sha = ext["content_sha256"]

    log.info("=" * 72)
    log.info("SPLIT VALIDITY AUDIT")
    log.info("  dependence range r = %.4f km", r_km)
    if r_ci_here and r_ci_splits and not np.allclose(
            np.sort(np.array(r_ci_here, float)),
            np.sort(np.array(r_ci_splits, float)), rtol=1e-6):
        log.warning("  r interval disagrees between artefacts: "
                    "spatial_dependence.json %s vs split_qc.json %s",
                    r_ci_here, r_ci_splits)
    log.info("  external regions   = %d, development regions = %d",
             len(ext_regions), len(dev_regions))
    log.info("=" * 72)

    findings: list[dict] = []

    def finding(severity: str, check: str, detail: str) -> None:
        findings.append({"severity": severity, "check": check, "detail": detail})
        log.info("  [%s] %s — %s", severity, check, detail)

    finding("WARN" if (r_ci_here and r_ci_splits and not np.allclose(
                np.sort(np.array(r_ci_here, float)),
                np.sort(np.array(r_ci_splits, float)), rtol=1e-6)) else "PASS",
            "one interval for the dependence range",
            f"point estimate r = {r_km:.4f} km; interval in "
            f"spatial_dependence.json = {r_ci_here}, interval in split_qc.json = "
            f"{r_ci_splits}")

    # ------------------------------------------------------------------ #
    # 0. The corpus, re-derived from the artefacts
    # ------------------------------------------------------------------ #
    inv = inventory[inventory["label_found"] == True]           # noqa: E712
    sample_coords = coords.set_index("sample_id")
    inv_ids = set(inv["sample_id"])
    log.info("usable tiles in the inventory : %d", len(inv_ids))
    log.info("tiles with coordinates (T1)    : %d", len(sample_coords))
    missing_coords = sorted(inv_ids - set(sample_coords.index))
    if missing_coords:
        finding("FAIL", "coordinate coverage",
                f"{len(missing_coords)} of the {len(inv_ids)} usable tiles have "
                f"no coordinate row; they cannot be audited for spatial "
                f"separation (e.g. {missing_coords[:5]})")
    else:
        finding("PASS", "coordinate coverage",
                f"every one of the {len(inv_ids)} usable tiles has a T1 "
                f"georeferenced centroid; the coordinate table also carries "
                f"{len(sample_coords) - len(inv_ids)} tiles that ship without a "
                f"label and are therefore outside every pool")

    # ------------------------------------------------------------------ #
    # 1. The external test: recompute its size and its regions
    # ------------------------------------------------------------------ #
    ext_ids = set(inv.loc[inv["region"].isin(ext_regions), "sample_id"])
    ext_regions_present = sorted(
        inv.loc[inv["region"].isin(ext_regions), "region"].unique())
    log.info("external test set             : %d tiles in %d regions",
             len(ext_ids), len(ext_regions_present))
    if len(ext_regions_present) != len(ext_regions):
        finding("FAIL", "external regions",
                f"the frozen freeze names {len(ext_regions)} regions but the "
                f"inventory carries tiles for {len(ext_regions_present)}")
    else:
        finding("PASS", "external regions",
                f"all {len(ext_regions)} frozen regions carry imagery; "
                f"{len(ext_ids)} tiles")

    dev_ids = set(inv.loc[inv["region"].isin(dev_regions), "sample_id"])
    other_ids = inv_ids - ext_ids - dev_ids
    if other_ids:
        others = sorted(inv.loc[inv["sample_id"].isin(other_ids), "region"].unique())
        finding("INFO", "tiles outside both pools",
                f"{len(other_ids)} usable tiles lie in regions that are in neither "
                f"the development nor the external pool: {others} "
                f"(regions with fewer than min_tiles_per_region = 4 tiles; the "
                f"development and external totals close as "
                f"{len(dev_ids)} + {len(ext_ids)} + {len(other_ids)} = "
                f"{len(dev_ids) + len(ext_ids) + len(other_ids)})")

    # ------------------------------------------------------------------ #
    # 2. Duplicate content, re-derived
    # ------------------------------------------------------------------ #
    sha = dict(zip(inv["sample_id"], inv["sha256"]))
    dup_sha = len(sha) - len(set(sha.values()))
    finding("PASS" if dup_sha == 0 else "FAIL", "exact duplicates",
            f"{dup_sha} repeated image SHA-256 across the {len(sha)} usable tiles")

    phash = {k: v for k, v in zip(inv["sample_id"], inv["phash"])
             if isinstance(v, str) and v}
    near_min, near_pairs = None, None
    if len(phash) >= 2:
        from sklearn.neighbors import NearestNeighbors

        ids = list(phash)
        bits = np.array([[int(c, 16) for c in h.zfill(64)]
                         for h in (phash[i] for i in ids)], dtype=np.uint8)
        nn = NearestNeighbors(n_neighbors=2, metric="hamming").fit(bits)
        dist, idx = nn.kneighbors(bits)
        near_min = float(dist[:, 1].min() * 64.0)
        near_pairs = int((dist[:, 1] * 64.0 <= 6).sum() // 2)
        del idx
    if near_min is None:
        finding("INFO", "near duplicates", "no perceptual hashes available")
    else:
        finding("PASS" if near_pairs == 0 else "FAIL", "near duplicates",
                f"closest perceptual-hash pair is {near_min:.0f} bits apart "
                f"(threshold 6); pairs within threshold: {near_pairs}. A pair "
                f"cannot be split across train and validation if no such pair "
                f"exists anywhere in the corpus")

    # ------------------------------------------------------------------ #
    # 3. Per-partition, per-fold validity
    # ------------------------------------------------------------------ #
    per_fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, idx_row in index.iterrows():
        strategy = str(idx_row["strategy"])
        path = SPLITS_DIR / str(idx_row["file"])
        assign = pd.read_csv(path, keep_default_na=False)
        params = eval(str(idx_row["params"]))           # written by build_splits
        buffer_km = float(params.get("buffer_km") or 0.0)

        folds = _fold_partitions(assign)
        n_train, n_val, min_d, med_d, p05_d = [], [], [], [], []
        overlap_prop, contact_prop = [], []
        region_in_train_prop = []
        val_ids_all: set[str] = set()
        id_overlap = 0
        ext_in_partition = 0
        dup_rows = 0
        val_without_coords = 0
        train_without_coords = 0

        for k, val, train in folds:
            tr_ids = list(train["sample_id"])
            va_ids = list(val["sample_id"])
            n_train.append(len(tr_ids))
            n_val.append(len(va_ids))
            val_ids_all |= set(va_ids)

            dup_rows += len(va_ids) - len(set(va_ids))
            dup_rows += len(tr_ids) - len(set(tr_ids))
            both = set(tr_ids) & set(va_ids)
            id_overlap += len(both)
            ext_in_partition += len((set(tr_ids) | set(va_ids)) & ext_ids)

            if "region" in val.columns and "region" in train.columns:
                tr_regions = set(train["region"].astype(str))
                hits = val["region"].astype(str).isin(tr_regions)
                region_in_train_prop.append(float(hits.mean()))

            va_xy = sample_coords.reindex(va_ids)
            tr_xy = sample_coords.reindex(tr_ids)
            va_ok = va_xy[["lat", "lon"]].notna().all(axis=1).to_numpy()
            tr_ok = tr_xy[["lat", "lon"]].notna().all(axis=1).to_numpy()
            val_without_coords += int((~va_ok).sum())
            train_without_coords += int((~tr_ok).sum())
            if not va_ok.any() or not tr_ok.any():
                continue

            d = balltree_nearest_km(
                tr_xy.loc[tr_ok, ["lat", "lon"]].to_numpy(float),
                va_xy.loc[va_ok, ["lat", "lon"]].to_numpy(float))
            min_d.append(float(np.nanmin(d)))
            med_d.append(float(np.nanmedian(d)))
            p05_d.append(float(np.nanpercentile(d, 5)))
            overlap_prop.append(float(np.nanmean(d < 0.5)))

            # Could a validation tile and its nearest training tile physically
            # touch? Compare centroid separation with the sum of the two
            # half-diagonals, which is an upper bound on the footprint radius.
            hd_v = va_xy.loc[va_ok, "half_diagonal_km"].to_numpy(float)
            hd_t = tr_xy.loc[tr_ok, "half_diagonal_km"].to_numpy(float)
            # nearest training index by full matrix is O(n_val*n_train); the
            # corpus is small enough for the exact answer, and the exact answer
            # is what decides whether footprints can touch.
            from sklearn.neighbors import BallTree
            tree = BallTree(np.radians(tr_xy.loc[tr_ok, ["lat", "lon"]]
                                        .to_numpy(float)), metric="haversine")
            dd, ii = tree.query(np.radians(va_xy.loc[va_ok, ["lat", "lon"]]
                                            .to_numpy(float)), k=1)
            dd = dd[:, 0] * EARTH_RADIUS_KM
            ii = ii[:, 0]
            contact = dd < (hd_v + hd_t[ii])
            contact_prop.append(float(np.nanmean(contact)))
            del tree

            per_fold_rows.append({
                "strategy": strategy, "fold": int(k),
                "n_train": len(tr_ids), "n_val": len(va_ids),
                "id_overlap_train_val": len(both),
                "external_ids_in_partition": len((set(tr_ids) | set(va_ids)) & ext_ids),
                "duplicate_rows": dup_rows,
                "min_val_to_train_km": round(float(np.nanmin(d)), 4),
                "p05_val_to_train_km": round(float(np.nanpercentile(d, 5)), 4),
                "median_val_to_train_km": round(float(np.nanmedian(d)), 4),
                "prop_val_within_0.5km_of_train": round(float(np.nanmean(d < 0.5)), 4),
                "prop_val_within_1km_of_train": round(float(np.nanmean(d < 1.0)), 4),
                "prop_val_within_2km_of_train": round(float(np.nanmean(d < 2.0)), 4),
                "prop_val_within_r_of_train": round(float(np.nanmean(d < r_km)), 4),
                "prop_val_footprint_may_contact_train": round(
                    float(np.nanmean(contact)), 4),
                "prop_val_region_also_in_train": (
                    round(region_in_train_prop[-1], 4)
                    if region_in_train_prop else None),
            })

        row = {
            "strategy": strategy,
            "representation": str(idx_row["representation"]),
            "n_folds": len(folds),
            "buffer_km": buffer_km or None,
            "n_train_median": int(np.median(n_train)) if n_train else None,
            "n_train_min": int(np.min(n_train)) if n_train else None,
            "n_train_max": int(np.max(n_train)) if n_train else None,
            "n_val_median": int(np.median(n_val)) if n_val else None,
            "id_overlap_train_val": id_overlap,
            "duplicate_rows": dup_rows,
            "external_ids_in_partition": ext_in_partition,
            "min_val_to_train_km": round(min(min_d), 4) if min_d else None,
            "p05_val_to_train_km_mean_over_folds": (
                round(float(np.mean(p05_d)), 4) if p05_d else None),
            "median_val_to_train_km_mean_over_folds": (
                round(float(np.mean(med_d)), 4) if med_d else None),
            # Both aggregations are reported because the manuscript's Table 4.3
            # column is the ACROSS-FOLD MEDIAN of the per-fold medians, while the
            # obvious default is the mean. For the buffered designs the two
            # differ by more than a factor of three (the folds exclude different
            # buffers, so per-fold medians vary widely), and a table whose caption
            # says only "aggregated over folds" cannot be checked against either
            # without knowing which one it is.
            "median_of_fold_medians_km": (
                round(float(np.median(med_d)), 4) if med_d else None),
            "prop_val_within_0.5km_of_train_mean_over_folds": (
                round(float(np.mean(overlap_prop)), 4) if overlap_prop else None),
            "prop_val_footprint_may_contact_train_mean_over_folds": (
                round(float(np.mean(contact_prop)), 4) if contact_prop else None),
            "prop_val_region_also_in_train_mean_over_folds": (
                round(float(np.mean(region_in_train_prop)), 4)
                if region_in_train_prop else None),
            "val_tiles_without_coordinates": val_without_coords,
            "train_tiles_without_coordinates": train_without_coords,
            "distinct_val_samples": len(val_ids_all),
        }
        summary_rows.append(row)
        log.info("  %-42s folds=%-3d n_train~%-5s min d=%-8s "
                 "val<0.5km=%-6s same-region=%s",
                 strategy, row["n_folds"], row["n_train_median"],
                 row["min_val_to_train_km"],
                 row["prop_val_within_0.5km_of_train_mean_over_folds"],
                 row["prop_val_region_also_in_train_mean_over_folds"])

    per_fold = pd.DataFrame(per_fold_rows)
    summary = pd.DataFrame(summary_rows)
    per_fold.to_csv(TABLES / "split_validity_per_fold.csv", index=False)
    summary.to_csv(TABLES / "split_validity_summary.csv", index=False)

    # ------------------------------------------------------------------ #
    # 4. Verdicts that the table implies
    # ------------------------------------------------------------------ #
    bad = summary[(summary["id_overlap_train_val"] > 0)
                  | (summary["external_ids_in_partition"] > 0)
                  | (summary["duplicate_rows"] > 0)]
    if bad.empty:
        finding("PASS", "partition disjointness",
                "no partition places a sample in both training and validation, "
                "none contains an external sample, and none lists a sample twice")
    else:
        finding("FAIL", "partition disjointness",
                f"{len(bad)} partition(s) violate it: "
                f"{bad['strategy'].tolist()}")

    ext_leak = summary[summary["external_ids_in_partition"] > 0]
    if ext_leak.empty:
        finding("PASS", "external test not used for training or validation",
                f"0 of the {int(summary['n_folds'].sum())} folds across "
                f"{len(summary)} partitions contain any of the {len(ext_ids)} "
                f"external tiles")

    # buffer isolation: the minimum distance must reach the declared buffer
    buffered = summary[summary["buffer_km"].notna()]
    for _, b in buffered.iterrows():
        ok = float(b["min_val_to_train_km"]) >= float(b["buffer_km"]) - 1e-6
        finding("PASS" if ok else "FAIL",
                f"buffer isolation {b['strategy']}",
                f"declared buffer {b['buffer_km']} km, measured minimum "
                f"validation-to-training distance {b['min_val_to_train_km']} km")

    # ------------------------------------------------------------------ #
    # 5. Does every registry row match the partition it claims?
    # ------------------------------------------------------------------ #
    reg = pd.read_csv(REGISTRY)
    problems: list[str] = []
    if len(reg):
        for _, rr in reg.iterrows():
            part = SPLITS_DIR / f"{rr['strategy']}.csv"
            if not part.exists():
                problems.append(f"{rr['run_id']}: no partition file")
                continue
            assign = pd.read_csv(part, keep_default_na=False)
            train, val = fold_frames(assign, int(rr["fold"]))
            if len(train) != int(rr["n_train"]) or len(val) != int(rr["n_val"]):
                problems.append(
                    f"{rr['run_id']}: registry says n_train={rr['n_train']} "
                    f"n_val={rr['n_val']}, partition gives {len(train)}/{len(val)}")
            if str(rr["external_freeze_sha256"]) != freeze_sha:
                problems.append(f"{rr['run_id']}: external freeze hash differs")
        # ---- one TRAINING configuration, judged on what training saw -------- #
        # The registry's `config_sha256` hashes the whole config FILE, so editing
        # a comment changes it while changing nothing about training. Comparing
        # that hash would flag a documentation edit as a comparability failure;
        # ignoring it entirely would miss a real change to the learning rate.
        # The check therefore compares the RESOLVED hyperparameters each run
        # recorded (written into every run artefact precisely so this is
        # answerable), and treats a differing `num_workers` as what it is: a
        # throughput knob that cannot change the data seen or the order of
        # optimisation steps (D-043, D-056).
        # Keys that are ALLOWED to differ, with the reason, because "the runs
        # differ" is not by itself a defect:
        #   * `seeds` is the random seed — an experimental factor by design, not
        #     a configuration value. Runs at seed 42 and seed 1337 are supposed
        #     to record different seeds.
        #   * `num_workers` is a throughput knob that changes neither the data
        #     seen nor the order of optimisation steps (D-043, D-056).
        ALLOWED_TO_DIFFER = {"num_workers": "throughput knob only (D-043, D-056)",
                             "seeds": "the random seed is the treatment"}
        resolved: dict[str, dict] = {}
        for path in sorted((PROJ / "experiments" / "runs").glob("*.json")):
            if path.name.endswith(".error.json"):
                continue
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            hp = blob.get("hyperparameters")
            if isinstance(hp, dict):
                resolved[blob.get("row", {}).get("run_id", path.stem)] = hp
        if len(resolved) >= 2:
            keys = sorted(set().union(*(set(v) for v in resolved.values())))
            differing = {k: {rid: v.get(k) for rid, v in resolved.items()}
                         for k in keys
                         if len({json.dumps(v.get(k), sort_keys=True)
                                 for v in resolved.values()}) > 1}
            hard = {k: v for k, v in differing.items()
                    if k not in ALLOWED_TO_DIFFER}
            if not differing:
                finding("PASS", "one training configuration",
                        f"all {len(resolved)} completed runs recorded identical "
                        f"resolved hyperparameters ({len(keys)} values)")
            elif not hard:
                finding("PASS", "one training configuration",
                        "every training value is identical across runs; the only "
                        "keys that differ are "
                        + "; ".join(f"{k} ({ALLOWED_TO_DIFFER[k]})"
                                    for k in sorted(differing)))
            else:
                finding("FAIL", "one training configuration",
                        f"resolved hyperparameters differ across runs: "
                        f"{sorted(hard)}")
        configs = set(reg["config_sha256"].astype(str))
        finding("INFO", "config-file hash",
                f"{len(configs)} distinct config_sha256 across {len(reg)} "
                f"registry rows; the file hash covers comments as well as values, "
                f"which is why the check above is made on the resolved "
                f"hyperparameters instead")
        finding("PASS" if not problems else "FAIL",
                "registry matches the partitions",
                "every row's n_train/n_val matches its partition and every row "
                "used the frozen external set" if not problems
                else "; ".join(problems))

    # ------------------------------------------------------------------ #
    # 6. Validation composition vs external composition
    # ------------------------------------------------------------------ #
    class_cols = [c for c in sample_class.columns if c.startswith("share_class_")]
    comp_rows = []
    if class_cols:
        sc = sample_class.set_index("sample_id")
        ext_share = sc.reindex(sorted(ext_ids))[class_cols].mean()
        comp_rows.append({"pool": "external_test", "n_tiles": len(ext_ids),
                          **{c: round(float(ext_share[c]), 4) for c in class_cols}})
        dev_share = sc.reindex(sorted(dev_ids))[class_cols].mean()
        comp_rows.append({"pool": "development_pool", "n_tiles": len(dev_ids),
                          **{c: round(float(dev_share[c]), 4) for c in class_cols}})
        for _, idx_row in index.iterrows():
            strategy = str(idx_row["strategy"])
            assign = pd.read_csv(SPLITS_DIR / str(idx_row["file"]),
                                 keep_default_na=False)
            _, val = fold_frames(assign, 0)
            v = sc.reindex(list(val["sample_id"]))[class_cols].mean()
            comp_rows.append({"pool": f"{strategy} (fold 0 validation)",
                              "n_tiles": len(val),
                              **{c: round(float(v[c]), 4) for c in class_cols}})
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(TABLES / "split_class_composition_val_vs_ext.csv", index=False)
        worst = (comp[~comp["pool"].isin(["external_test", "development_pool"])]
                 .drop(columns=["pool", "n_tiles"])
                 .sub(ext_share, axis=1).abs().max(axis=1))
        finding("INFO", "class composition, validation fold 0 vs external test",
                "the class shares are fractions of pixels, so the largest "
                "single-class share difference is "
                f"{100 * float(worst.max()):.2f} percentage points "
                f"({comp.loc[worst.idxmax(), 'pool']}); per-pool shares are in "
                f"results/tables/split_class_composition_val_vs_ext.csv")

    # ------------------------------------------------------------------ #
    # 7. Write the report
    # ------------------------------------------------------------------ #
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_partitions_audited": int(len(summary)),
        "n_folds_audited": int(summary["n_folds"].sum()),
        "external_freeze_sha256": freeze_sha,
        "external_tiles": len(ext_ids),
        "external_regions": len(ext_regions_present),
        "dependence_range_r_km": r_km,
        "findings": findings,
        "status": ("FAIL" if any(f["severity"] == "FAIL" for f in findings)
                   else "PASS"),
    }
    write_json(PIPELINE / "split_validity.json", payload)

    lines = [
        "# Split validity audit — independent re-measurement",
        "",
        f"Generated {payload['generated_at_utc']} by "
        f"`scripts/verify_split_validity.py`.",
        "",
        "This is a **second, independent** audit of the partitions the experiment",
        "grid used. Distances are recomputed with a `BallTree` on unit-sphere",
        "coordinates, not with the chunked haversine the splits were built with,",
        "and every number below is derived from the partition files and the",
        "inventory rather than quoted from an earlier report.",
        "",
        f"**Status: {payload['status']}** — {len(summary)} partitions, "
        f"{payload['n_folds_audited']} folds, "
        f"{len(ext_ids)} external tiles in {len(ext_regions_present)} regions, "
        f"dependence range r = {r_km:.3f} km.",
        "",
        "## Findings",
        "",
        "| severity | check | detail |",
        "|---|---|---|",
    ]
    lines += [f"| {f['severity']} | {f['check']} | {f['detail']} |"
              for f in findings]
    lines += [
        "",
        "## Per-partition summary",
        "",
        "`n_train` is the median over folds. `min d` is the smallest",
        "validation-to-training tile-centroid distance in any fold.",
        "`val <0.5 km` is the mean over folds of the share of validation tiles",
        "that have a training tile within 0.5 km. `same region` is the mean over",
        "folds of the share of validation tiles whose region also appears in the",
        "training set of the same fold.",
        "",
        "| strategy | folds | n_train | min d (km) | val <0.5 km | val footprint may touch | same region |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['n_folds']} | {r['n_train_median']} | "
            f"{r['min_val_to_train_km']} | "
            f"{r['prop_val_within_0.5km_of_train_mean_over_folds']} | "
            f"{r['prop_val_footprint_may_contact_train_mean_over_folds']} | "
            f"{r['prop_val_region_also_in_train_mean_over_folds']} |")
    lines += [
        "",
        "## Files",
        "",
        "- `results/tables/split_validity_per_fold.csv` — one row per fold.",
        "- `results/tables/split_validity_summary.csv` — one row per partition.",
        "- `results/tables/split_class_composition_val_vs_ext.csv` — class shares",
        "  of the external test, the development pool and each partition's fold-0",
        "  validation set.",
        "- `results/pipeline/split_validity.json` — this report, machine-readable.",
        "",
    ]
    (PIPELINE / "split_validity_report.md").write_text("\n".join(lines),
                                                       encoding="utf-8")
    log.info("status: %s", payload["status"])
    log.info("wrote %s", PIPELINE / "split_validity_report.md")
    print(f"[PHASE] Split validity audit")
    print(f"[STATUS] {'PASS' if payload['status'] == 'PASS' else 'FAIL'}")
    print(f"[FILES] {PIPELINE / 'split_validity_report.md'}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

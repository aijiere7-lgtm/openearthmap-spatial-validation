#!/usr/bin/env python
"""
test_split_logic.py — validates the partition and diagnostic logic on synthetic
data with KNOWN spatial structure.

Why this exists: the dataset is large and slow to obtain, and a defect in the
split or diagnostic code would not surface until after a training run. Synthetic
data with a known answer lets the logic be verified first, without the dataset.

The assertions encode what each design is supposed to guarantee, and one
deliberately encodes what it does NOT guarantee:

  * S0 (random)          -> adjacent samples land in different folds, so the
                            minimum train-validation distance is TINY.
  * S2 (spatial block)   -> the MEDIAN distance is large, but the MINIMUM can
                            still be small at block boundaries. Spatial block CV
                            does not guarantee isolation; this is precisely why
                            the protocol requires independence to be measured
                            rather than inferred from the strategy's name.
  * S3 (buffered)        -> the minimum distance is >= the buffer width, because
                            samples inside the buffer are removed from training.
  * S4 (leave-region-out)-> no region is shared between training and validation.

Run:  python tests/test_split_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splits.diagnostics import leakage_diagnostics  # noqa: E402
from splits.strategies import (  # noqa: E402
    apply_buffer, group_folds, leave_one_region_out, random_folds,
    select_external_regions, spatial_block_folds,
)

from common import NUM_CLASSES  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(f"{label} {detail}".strip())


def make_synthetic(n_regions: int = 4, per_region: int = 60, seed: int = 0) -> pd.DataFrame:
    """Four tight spatial clusters, each spanning ~1 km, placed far apart."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        # Cluster centres spread over ~40 km
        clon = 10.0 + r * 0.4
        clat = 50.0 + (r % 2) * 0.4
        for i in range(per_region):
            rows.append({
                "sample_id": f"R{r}_P{i:04d}",
                "region": f"Region{r}",
                "country": f"Country{r % 2}",
                # ~1 km cluster: 0.01 deg lon ~ 0.7 km at 50N
                "lon": clon + rng.uniform(0, 0.01),
                "lat": clat + rng.uniform(0, 0.01),
                "dominant_class": int(rng.integers(0, 8)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 72)
    print("SPLIT LOGIC VALIDATION (synthetic data, known spatial structure)")
    print("=" * 72)

    df = make_synthetic()
    meta = df[["sample_id", "region", "country", "lon", "lat"]].copy()
    n_folds = 5

    print(f"\nsynthetic data: {len(df)} samples, "
          f"{df['region'].nunique()} regions, {n_folds} folds")

    # ------------------------------------------------------------------ S0 --
    print("\n--- S0 random patch ---")
    s0 = random_folds(df, n_folds=n_folds, seed=42)
    d0 = leakage_diagnostics(s0, meta, region_col="region")
    print(f"  n_train={d0['n_train_median']} n_val={d0['n_val_median']}")
    print(f"  min_dist={d0['min_train_val_dist_km']:.4f} km  "
          f"median_dist={d0['median_train_val_dist_km']:.4f} km")
    check(d0["n_train_median"] > 0, "S0 has a non-empty training set")
    check(d0["min_train_val_dist_km"] is not None,
          "S0 produces distance diagnostics (train/val correctly derived)")
    check(d0["min_train_val_dist_km"] < 0.5,
          "S0 minimum distance is tiny (adjacent samples split across folds)",
          f"{d0['min_train_val_dist_km']:.4f} km")
    check(d0["same_region_contamination_prop"] == 1.0,
          "S0 shares regions between train and val (expected for random)",
          f"{d0['same_region_contamination_prop']}")

    # ------------------------------------------------------------------ S4 --
    print("\n--- S4 leave-one-region-out ---")
    s4 = leave_one_region_out(df, region_col="region")
    d4 = leakage_diagnostics(s4, meta, region_col="region")
    print(f"  folds={d4['n_folds']}  same_region_contamination="
          f"{d4['same_region_contamination_prop']}")
    check(d4["same_region_contamination_prop"] == 0.0,
          "S4 has ZERO region contamination",
          f"{d4['same_region_contamination_prop']}")
    check(d4["min_train_val_dist_km"] is not None and
          d4["min_train_val_dist_km"] > 1.0,
          "S4 minimum distance is large (whole regions held out)",
          f"{d4['min_train_val_dist_km']:.2f} km")

    # ------------------------------------------------------------------ S2 --
    print("\n--- S2 spatial block (block = 5 km) ---")
    s2 = spatial_block_folds(df, n_folds=n_folds, seed=42, block_size_km=5.0)
    d2 = leakage_diagnostics(s2, meta, region_col="region")
    print(f"  min_dist={d2['min_train_val_dist_km']:.4f} km  "
          f"median_dist={d2['median_train_val_dist_km']:.4f} km")
    check(d2["median_train_val_dist_km"] > d0["median_train_val_dist_km"],
          "S2 median distance exceeds S0 median distance",
          f"{d2['median_train_val_dist_km']:.3f} > "
          f"{d0['median_train_val_dist_km']:.3f} km")
    check(d2["min_train_val_dist_km"] is not None,
          "S2 produces diagnostics")

    # ------------------------------------------------------------------ S3 --
    print("\n--- S3 buffered block (block = 5 km, buffer = 2 km) ---")
    base = spatial_block_folds(df, n_folds=n_folds, seed=42, block_size_km=5.0)
    coords = (df[["sample_id", "lon", "lat"]].set_index("sample_id")
              .reindex(base["sample_id"]).reset_index())
    buf_km = 2.0
    parts = []
    for k in range(n_folds):
        b = apply_buffer(base, coords, buffer_km=buf_km, val_fold=k).copy()
        b["eval_fold"] = k
        b["split"] = np.where(b["fold"] == k, "val",
                              np.where(b["split"] == "excluded_buffer",
                                       "excluded_buffer", "train"))
        parts.append(b)
    s3 = pd.concat(parts, ignore_index=True)

    n_excl = int((s3["split"] == "excluded_buffer").sum())
    print(f"  rows={len(s3)} (expanded), buffer exclusions={n_excl}")
    check(len(s3) == n_folds * len(df),
          "S3 uses the expanded per-eval_fold representation",
          f"{len(s3)} == {n_folds} x {len(df)}")

    d3 = leakage_diagnostics(s3, meta, region_col="region")
    print(f"  min_dist={d3['min_train_val_dist_km']:.4f} km  "
          f"median_dist={d3['median_train_val_dist_km']:.4f} km")
    check(n_excl > 0, "S3 buffer actually excludes samples",
          f"{n_excl} exclusions")
    check(d3["min_train_val_dist_km"] is not None and
          d3["min_train_val_dist_km"] >= buf_km * 0.95,
          "S3 minimum train-val distance >= buffer width",
          f"{d3['min_train_val_dist_km']:.3f} km >= {buf_km} km")
    check(d3["n_train_median"] < d2["n_train_median"],
          "S3 training set is smaller than S2 (buffer cost)",
          f"{d3['n_train_median']} < {d2['n_train_median']}")

    # ---------------------------------------------------- disjointness QC --
    print("\n--- train/validation disjointness ---")
    for name, part in (("S0", s0), ("S2", s2), ("S3", s3), ("S4", s4)):
        bad = 0
        if "eval_fold" in part.columns:
            for k in part["eval_fold"].unique():
                sub = part[part["eval_fold"] == k]
                tr = set(sub.loc[sub["split"] == "train", "sample_id"])
                va = set(sub.loc[sub["split"] == "val", "sample_id"])
                bad += len(tr & va)
        else:
            for k in part["fold"].unique():
                tr = set(part.loc[part["fold"] != k, "sample_id"])
                va = set(part.loc[part["fold"] == k, "sample_id"])
                bad += len(tr & va)
        check(bad == 0, f"{name}: no sample is both train and val in the same fold",
              f"{bad} overlaps")

    # ------------------------------------------------- matched sample size --
    print("\n--- matched sample size control ---")
    n2 = int((s2.assign(split="val")["fold"] != 0).sum())
    m = random_folds(df, n_folds=n_folds, seed=99, n_train_target=n2)
    n_used = int((m["split"] == "val").sum()) + int(
        (m["split"] == "excluded_matched_sample_size").sum())
    check(n_used == len(df), "matched split accounts for every sample exactly once",
          f"{n_used} == {len(df)}")
    check((m["split"] == "excluded_matched_sample_size").sum() >= 0,
          "matched split marks exclusions rather than dropping rows")

    # ---------------------------------------------------------------------- #
    # Coordinate provenance in the external-test selection.
    #
    # The defect this guards against: an earlier implementation called
    # `dropna(subset=["lon", "lat"])` and, when nothing survived, blamed the
    # class-presence filter. That named the wrong cause, and it would have
    # silently accepted a PARTIAL exclusion -- some regions dropped from the
    # candidate pool with nothing said -- which changes what the selection means.
    # ---------------------------------------------------------------------- #
    print("\n[7] External-test selection: coordinate provenance is explicit")

    def _region_tables(sources: list[str]):
        n = len(sources)
        # Realistic shape: lon/lat are empty whenever the source is "none",
        # which is what build_region_metadata.py writes.
        lon = [float(i * 7) if s != "none" else float("nan") for i, s in enumerate(sources)]
        lat = [float(i * 3) if s != "none" else float("nan") for i, s in enumerate(sources)]
        rt = pd.DataFrame({
            "region": [f"R{i:02d}" for i in range(n)],
            "country": [f"C{i % 4}" for i in range(n)],
            "continent": [["Africa", "Asia", "Europe", "N. America"][i % 4]
                          for i in range(n)],
            "lon": lon, "lat": lat, "coord_source": sources,
        })
        cs = pd.DataFrame([
            {"region": f"R{i:02d}", "class_id": c, "pixel_count": 5000}
            for i in range(n) for c in range(NUM_CLASSES)])
        return rt, cs

    from common import COORD_SOURCES_OK_FOR_REGION_DISPERSION
    OK = COORD_SOURCES_OK_FOR_REGION_DISPERSION

    rt, cs = _region_tables(["geotiff_centroid"] * 12)
    r = select_external_regions(rt, cs, seed=1, target_n=(4, 6),
                                acceptable_coord_sources=OK)
    cr = r["coordinate_report"]
    check("error" not in r, "full coordinates: selection succeeds")
    check(cr["counts_balance"], "full coordinates: the count chain balances")
    check(cr["n_usable_for_selection"] == 12 and r["n_external"] == 4,
          "full coordinates: 12 usable, 4 chosen as external",
          f"usable={cr['n_usable_for_selection']} external={r.get('n_external')}")

    rt, cs = _region_tables(["none"] * 6 + ["geotiff_centroid"] * 6)
    r = select_external_regions(rt, cs, seed=1, target_n=(4, 6),
                                acceptable_coord_sources=OK)
    cr = r["coordinate_report"]
    check("error" not in r, "partial coordinates: selection still succeeds")
    check(cr["counts_balance"], "partial coordinates: the count chain balances")
    check(cr["n_excluded_no_coordinates"] == 6,
          "partial coordinates: all 6 coordinate-less regions are counted",
          f"counted={cr['n_excluded_no_coordinates']}")
    check(len(cr.get("excluded_regions", [])) == cr["n_excluded_no_coordinates"],
          "partial coordinates: the exclusion LIST agrees with the count",
          f"list={len(cr.get('excluded_regions', []))} "
          f"count={cr['n_excluded_no_coordinates']}")
    check(bool(cr.get("exclusion_note")),
          "partial coordinates: the exclusion is stated, not silent")

    rt, cs = _region_tables(["none"] * 12)
    r = select_external_regions(rt, cs, seed=1, target_n=(4, 6),
                                acceptable_coord_sources=OK)
    msg = r.get("error", "")
    check("COORDINATE" in msg.upper(),
          "no coordinates: the error names the COORDINATE problem")
    check("class-presence" not in msg or "not a class-coverage" in msg,
          "no coordinates: the error does not blame the class filter")

    rt, cs = _region_tables(["geotiff_centroid"] * 12)
    cs.loc[cs["class_id"] == 7, "pixel_count"] = 1
    r = select_external_regions(rt, cs, seed=1, target_n=(4, 6),
                                acceptable_coord_sources=OK)
    check("class-presence" in r.get("error", ""),
          "class filter exhausted: THAT error names the class filter",
          r.get("error", "")[:70])

    # The two coordinate standards must stay distinct: gazetteer centroids are
    # good enough for inter-region dispersion but not for estimating r.
    from common import COORD_SOURCES_OK_FOR_RANGE
    check("gazetteer_city_centroid" in OK,
          "region-level dispersion accepts gazetteer centroids")
    check("gazetteer_city_centroid" not in COORD_SOURCES_OK_FOR_RANGE,
          "range estimation REJECTS gazetteer centroids")
    check(COORD_SOURCES_OK_FOR_RANGE <= OK,
          "the range standard is strictly stricter than the dispersion standard")

    # ---------------------------------------------------------------- done --
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} assertion(s) failed")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print("RESULT: PASS — all assertions hold")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

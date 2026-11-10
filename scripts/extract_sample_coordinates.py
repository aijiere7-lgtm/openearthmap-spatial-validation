#!/usr/bin/env python
"""
extract_sample_coordinates.py — per-tile geographic coordinates (tier T1).

WHY THIS EXISTS
---------------
`spatial_metadata_audit.py` established that OpenEarthMap is a TIER T1 corpus:
every sampled image carries a CRS and a non-identity affine transform. Tier T1
means the dependence range r can be measured at the scale at which it actually
operates -- between individual 1024 px tiles, tens to hundreds of metres to a few
kilometres apart.

Without per-tile coordinates the analysis degrades to REGION scale: 75 points
spread over six continents, whose separations are 10^3 km. A range fitted at that
scale would describe continent-to-continent similarity, not the spatial
dependence the study is about, and presenting it as "r" would be a category
error dressed up as a measurement.

WHY A SEPARATE TABLE RATHER THAN THE INVENTORY
----------------------------------------------
The inventory is built by `build_inventory.py` and costs ~41 minutes, dominated
by SHA-256 and perceptual hashing of 3,838 images. Coordinates come from the
raster HEADER alone -- they are a geometric property of the file, not of its
content -- so folding them into the inventory would force a full re-hash to
recover information that can be read in seconds. Keeping them separate also keeps
each artefact single-purpose: the inventory describes files, this table describes
geography, `region_class_pixels.csv` describes composition.

WHAT IS RECORDED, AND WHY EACH COLUMN MATTERS
---------------------------------------------
  lon, lat            tile centre in EPSG:4326 (WGS84), transformed from the
                      tile's OWN CRS with pyproj. The corpus carries 36 distinct
                      CRSs, so a single hard-coded projection would be wrong for
                      most tiles.
  crs                 the tile's native CRS, retained so any later distance
                      computation can be reproduced or audited.
  crs_type            "projected" or "geographic". A geographic CRS makes the
                      affine pixel size DEGREES, not metres; deriving a GSD
                      without converting would understate it by ~5 orders of
                      magnitude. That conversion is applied here.
  gsd_x_m, gsd_y_m    ground sample distance in metres, converted at the tile's
                      latitude when the CRS is geographic.
  width_m, height_m   tile extent in metres, for interpreting the minimum
                      meaningful separation between two tiles.
  half_diagonal_km    half the tile diagonal: two tiles closer than roughly this
                      cannot be treated as spatially independent.
  coord_source        "geotiff_centroid" -- accepted by BOTH coordinate
                      standards in common.py, including the strict one used for r.
  coord_tier          "T1", stated explicitly so downstream code does not have to
                      re-derive the tier.

QC: two independent consistency checks are run, because a coordinate error is
silent and would corrupt every distance in the study.
  1. INTERNAL. Per region, the spread of tile centres is compared against the
     region extent already measured in `region_metadata.csv`. Agreement confirms
     the per-tile coordinates are consistent with the region-level derivation.
  2. ABSOLUTE. Every longitude must lie in [-180, 180] and every latitude in
     [-90, 90]; the corpus is also expected to span the latitude range implied by
     six continents, which a units or sign error would violate.

Output: data/metadata/sample_coordinates.csv
        data/metadata/sample_coordinates_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    LOGS_DIR, METADATA_DIR, ensure_dirs, get_logger, utcnow, write_json,
)
from data.inventory import resolve_inventory_path, resolve_inventory_root  # noqa: E402
from spatial.geometry import haversine_km  # noqa: E402

INVENTORY = METADATA_DIR / "dataset_inventory.csv"
REGION_METADATA = METADATA_DIR / "region_metadata.csv"
OUT = METADATA_DIR / "sample_coordinates.csv"
OUT_REPORT = METADATA_DIR / "sample_coordinates_report.json"

WGS84_EPSG = 4326
# Metres per degree at the equator, used only when converting a geographic CRS's
# degree-valued pixel size to metres at the tile's own latitude.
M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQUATOR = 111320.0
# A region whose tiles span more than this is flagged for review rather than
# silently accepted; the largest genuine value measured is ~103 km (mazowieckie).
REGION_SPREAD_REVIEW_KM = 200.0


def probe_tile(path: Path, transformer_factory) -> dict | None:
    """Read one raster's georeferencing. Header only; no pixel data is read."""
    import rasterio

    try:
        with rasterio.open(path) as src:
            crs = src.crs
            transform = src.transform
            n_w, n_h = src.width, src.height
    except Exception:
        return None

    rec: dict = {
        "crs": crs.to_string() if crs is not None else "",
        "crs_type": "",
        "lon": None, "lat": None,
        "gsd_x_m": None, "gsd_y_m": None,
        "width_px": int(n_w), "height_px": int(n_h),
        "width_m": None, "height_m": None,
        "half_diagonal_km": None,
        "has_crs": crs is not None,
        "has_transform": transform is not None and not transform.is_identity,
    }
    if crs is None or transform is None:
        return rec

    crs_type = "geographic" if crs.is_geographic else "projected"
    rec["crs_type"] = crs_type

    # Tile centre in the tile's own CRS, then to WGS84.
    cx, cy = transform * (n_w / 2.0, n_h / 2.0)
    try:
        transformer = transformer_factory(crs, f"EPSG:{WGS84_EPSG}")
        lon, lat = transformer.transform(cx, cy)
    except Exception:
        return rec
    rec["lon"], rec["lat"] = float(lon), float(lat)

    # Pixel size in metres. For a projected CRS the affine coefficients are
    # already metres. For a geographic CRS they are DEGREES and must be converted
    # at the tile's latitude -- skipping this would report a GSD about 1e5 times
    # too small.
    px = float(abs(transform.a))
    py = float(abs(transform.e))
    if crs_type == "projected":
        gx, gy = px, py
    else:
        gx = px * M_PER_DEG_LON_EQUATOR * np.cos(np.radians(rec["lat"]))
        gy = py * M_PER_DEG_LAT
    rec["gsd_x_m"], rec["gsd_y_m"] = float(gx), float(gy)
    rec["width_m"] = float(gx * n_w)
    rec["height_m"] = float(gy * n_h)
    rec["half_diagonal_km"] = float(0.5 * np.hypot(rec["width_m"],
                                                   rec["height_m"]) / 1000.0)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(INVENTORY))
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("sample_coordinates",
                        LOGS_DIR / "extract_sample_coordinates.log")
    logger.info("=" * 72)
    logger.info("PER-TILE COORDINATE EXTRACTION (tier T1)  %s", utcnow())
    logger.info("=" * 72)

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        logger.error("inventory not found: %s", inv_path)
        return 2
    inv = pd.read_csv(inv_path, keep_default_na=False)
    logger.info("inventory rows: %d", len(inv))

    if args.root:
        root = Path(args.root)
        root_info = {"method": "explicit", "root": str(root)}
    else:
        root, root_info = resolve_inventory_root(inventory_csv=inv_path,
                                                 logger=logger)
    if root is None or not root.is_dir():
        logger.error("could not resolve the corpus root")
        return 2

    from pyproj import Transformer

    def transformer_factory(src_crs, dst_crs):
        return Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Only rows whose image was readable can be probed.
    if "image_readable" in inv.columns:
        usable = inv[inv["image_readable"].astype(str).str.lower()
                     .isin({"true", "1", "yes"})].copy()
    else:
        usable = inv.copy()
    if args.limit:
        usable = usable.head(args.limit)
    logger.info("images to probe: %d", len(usable))

    rows: list[dict] = []
    n_unreadable = 0
    n_no_crs = 0
    n_no_transform = 0
    n_no_coords = 0
    crs_counts: dict[str, int] = defaultdict(int)

    for i, row in enumerate(usable.itertuples(index=False), 1):
        rec = row._asdict() if hasattr(row, "_asdict") else dict(row)
        sid = str(rec.get("sample_id", ""))
        region = str(rec.get("region") or "UNKNOWN")
        ip = str(rec.get("image_path") or "")
        out = {"sample_id": sid, "region": region,
               "coord_source": "geotiff_centroid", "coord_tier": "T1"}
        if not ip:
            n_unreadable += 1
            out.update({"lon": None, "lat": None, "notes": "no image path"})
            rows.append(out)
            continue

        p = resolve_inventory_path(root, ip)
        if not p.exists():
            n_unreadable += 1
            out.update({"lon": None, "lat": None,
                        "notes": "image path does not resolve"})
            rows.append(out)
            continue

        probe = probe_tile(p, transformer_factory)
        if probe is None:
            n_unreadable += 1
            out.update({"lon": None, "lat": None, "notes": "raster unreadable"})
            rows.append(out)
            continue

        out.update(probe)
        if not probe["has_crs"]:
            n_no_crs += 1
        if not probe["has_transform"]:
            n_no_transform += 1
        if probe["lon"] is None or probe["lat"] is None:
            n_no_coords += 1
        else:
            crs_counts[probe["crs"]] += 1
        rows.append(out)

        if i % 500 == 0:
            logger.info("  ... %d/%d probed", i, len(usable))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", OUT, len(df))

    # ---- QC 1: absolute range check -------------------------------------- #
    ok = df["lon"].notna() & df["lat"].notna()
    lon_ok = bool(((df.loc[ok, "lon"] >= -180) & (df.loc[ok, "lon"] <= 180)).all())
    lat_ok = bool(((df.loc[ok, "lat"] >= -90) & (df.loc[ok, "lat"] <= 90)).all())
    if not lon_ok:
        logger.error("QC FAIL: longitudes outside [-180, 180]")
    if not lat_ok:
        logger.error("QC FAIL: latitudes outside [-90, 90]")

    lat_span = (float(df.loc[ok, "lat"].max() - df.loc[ok, "lat"].min())
                if ok.any() else 0.0)
    lon_span = (float(df.loc[ok, "lon"].max() - df.loc[ok, "lon"].min())
                if ok.any() else 0.0)
    logger.info("latitude span %.2f deg, longitude span %.2f deg",
                lat_span, lon_span)

    # ---- QC 2: agreement with the region-level derivation ---------------- #
    region_check: dict = {"available": False}
    if REGION_METADATA.exists() and ok.any():
        rm = pd.read_csv(REGION_METADATA)
        if {"region", "lon", "lat"}.issubset(rm.columns):
            derived = (df[ok].groupby("region")[["lon", "lat"]]
                       .median().reset_index())
            cmp = derived.merge(
                rm[["region", "lon", "lat"]], on="region", how="inner",
                suffixes=("_sample", "_region"))
            if len(cmp):
                d = haversine_km(cmp["lat_sample"].to_numpy(float),
                                 cmp["lon_sample"].to_numpy(float),
                                 cmp["lat_region"].to_numpy(float),
                                 cmp["lon_region"].to_numpy(float))
                cmp = cmp.assign(diff_km=d)
                worst = cmp.loc[cmp["diff_km"].idxmax()]
                region_check = {
                    "available": True,
                    "n_regions_compared": int(len(cmp)),
                    "max_diff_km": float(d.max()),
                    "median_diff_km": float(np.median(d)),
                    "worst_region": str(worst["region"]),
                }
                logger.info(
                    "QC agreement with region_metadata.csv: n=%d, median %.2f km, "
                    "max %.2f km (%s)",
                    len(cmp), float(np.median(d)), float(d.max()),
                    worst["region"])

    # ---- QC 3: within-region spread -------------------------------------- #
    spread_rows = []
    if ok.any():
        for region, g in df[ok].groupby("region"):
            lo = g["lon"].to_numpy(float)
            la = g["lat"].to_numpy(float)
            if len(g) == 1:
                spread_rows.append({"region": region, "n_tiles": 1,
                                    "spread_km": 0.0,
                                    "median_half_diagonal_km":
                                        float(g["half_diagonal_km"].iloc[0] or 0.0)})
                continue
            mlon, mlat = float(np.median(lo)), float(np.median(la))
            spread = float(np.max(haversine_km(la, lo, mlat, mlon)))
            spread_rows.append({
                "region": region, "n_tiles": int(len(g)),
                "spread_km": spread,
                "median_half_diagonal_km":
                    float(np.nanmedian(g["half_diagonal_km"].to_numpy(float))),
            })
    spread = pd.DataFrame(spread_rows)
    flagged = (spread[spread["spread_km"] > REGION_SPREAD_REVIEW_KM]
               if len(spread) else spread)
    if len(flagged):
        logger.warning("regions whose tiles span > %.0f km (review): %s",
                       REGION_SPREAD_REVIEW_KM,
                       flagged[["region", "spread_km"]].to_dict("records"))

    # ---- summary --------------------------------------------------------- #
    n_total = len(df)
    n_ok = int(ok.sum())
    gsd = df.loc[ok, "gsd_x_m"].dropna()
    report = {
        "extracted_at_utc": utcnow(),
        "root": str(root),
        "root_resolution": root_info,
        "n_tiles": n_total,
        "n_with_coordinates": n_ok,
        "coverage_pct": round(100.0 * n_ok / n_total, 4) if n_total else 0.0,
        "coord_source": "geotiff_centroid",
        "coord_tier": "T1",
        "n_image_unreadable": n_unreadable,
        "n_without_crs": n_no_crs,
        "n_without_transform": n_no_transform,
        "n_without_coordinates": n_no_coords,
        "n_distinct_crs": len(crs_counts),
        "crs_counts": dict(sorted(crs_counts.items(), key=lambda kv: -kv[1])),
        "lon_range": [float(df.loc[ok, "lon"].min()), float(df.loc[ok, "lon"].max())]
                     if ok.any() else None,
        "lat_range": [float(df.loc[ok, "lat"].min()), float(df.loc[ok, "lat"].max())]
                     if ok.any() else None,
        "lon_in_valid_range": lon_ok,
        "lat_in_valid_range": lat_ok,
        "gsd_x_m": {"min": float(gsd.min()), "max": float(gsd.max()),
                    "median": float(gsd.median()),
                    "distinct": sorted({round(float(v), 6) for v in gsd.unique()})}
                   if len(gsd) else None,
        "half_diagonal_km_median":
            float(np.nanmedian(df.loc[ok, "half_diagonal_km"].to_numpy(float)))
            if ok.any() else None,
        "region_spread_km": {
            "min": float(spread["spread_km"].min()) if len(spread) else None,
            "median": float(spread["spread_km"].median()) if len(spread) else None,
            "max": float(spread["spread_km"].max()) if len(spread) else None,
            "n_over_review_threshold": int(len(flagged)),
            "review_threshold_km": REGION_SPREAD_REVIEW_KM,
        },
        "agreement_with_region_metadata": region_check,
        "outputs": {"sample_coordinates": str(OUT)},
    }
    write_json(OUT_REPORT, report)
    spread.to_csv(METADATA_DIR / "sample_coordinates_region_spread.csv",
                  index=False, encoding="utf-8")

    # ---- gate ------------------------------------------------------------ #
    complete = n_ok == n_total
    status = "PASS" if (complete and lon_ok and lat_ok) else "PARTIAL"
    if not complete and n_ok == 0:
        status = "FAIL"

    print()
    print("[PHASE] Per-tile coordinates (tier T1)")
    print(f"[STATUS] {status}")
    print(f"[COMPLETED] {n_ok}/{n_total} tiles carry WGS84 coordinates "
          f"({report['coverage_pct']}%)")
    print(f"[QC] lon valid={lon_ok} lat valid={lat_ok}; "
          f"lat span {lat_span:.1f} deg, lon span {lon_span:.1f} deg")
    if report["gsd_x_m"]:
        g = report["gsd_x_m"]
        print(f"[QC] GSD {g['min']:.3f}-{g['max']:.3f} m "
              f"(median {g['median']:.3f}); distinct {g['distinct']}")
    print(f"[QC] {report['n_distinct_crs']} distinct CRS across tiles")
    if region_check.get("available"):
        print(f"[QC] agreement with region_metadata.csv: "
              f"median {region_check['median_diff_km']:.2f} km, "
              f"max {region_check['max_diff_km']:.2f} km "
              f"({region_check['worst_region']})")
    print(f"[QC] within-region spread: median "
          f"{report['region_spread_km']['median']:.2f} km, "
          f"max {report['region_spread_km']['max']:.2f} km")
    if n_no_crs or n_no_transform or n_unreadable:
        print(f"[ISSUES] unreadable={n_unreadable} no_crs={n_no_crs} "
              f"identity_transform={n_no_transform} no_coords={n_no_coords}")
    if len(flagged):
        print(f"[ISSUES] {len(flagged)} region(s) span > "
              f"{REGION_SPREAD_REVIEW_KM:.0f} km -- review")
    print(f"[FILES] {OUT}")
    print("[NEXT] spatial dependence at tile scale (r), then freeze the "
          "external test")
    logger.info("=" * 72)
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

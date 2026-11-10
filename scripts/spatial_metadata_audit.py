#!/usr/bin/env python
"""
spatial_metadata_audit.py — determine the achievable coordinate tier.

The study needs coordinates to compute continuous train-validation distances.
Whether the retrieved archive actually provides them is DETERMINED HERE, not
assumed (research_protocol.md §5, DECISIONS D-009).

Tiers:
  T1  per-image / per-patch coordinates  -> true distances, range r estimable
  T2  region-level coordinates only      -> region-scale distances, coarse blocks
  T3  no usable coordinates              -> grouping only; NO numeric r reported

Under T3 the study is barred from claiming a distance-similarity relationship
or asserting that any block size is optimal. This script enforces that by
writing the tier and its evidence to disk for every downstream stage to read.

Output: data/metadata/spatial_metadata_audit.md
        data/metadata/spatial_metadata_audit.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    DATA_DIR, LOGS_DIR, METADATA_DIR, ensure_dirs, get_logger, utcnow,
    write_json,
)

# Column names that would indicate coordinates in a metadata table.
LON_HINTS = {"lon", "long", "longitude", "x", "xcoord", "easting", "utm_x"}
LAT_HINTS = {"lat", "latitude", "y", "ycoord", "northing", "utm_y"}
GEO_PATTERNS = [
    # "POINT(12.34 56.78)" / "12.34,56.78" / "12.34 56.78"
    re.compile(r"(-?\d{1,3}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)"),
]


def plausibility(lon: np.ndarray, lat: np.ndarray) -> tuple[bool, list[str]]:
    """Reject coordinate columns that cannot be geographic degrees."""
    notes: list[str] = []
    if len(lon) == 0:
        return False, ["empty"]
    if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
        notes.append("contains non-finite values")
    if lon.min() < -180 or lon.max() > 180:
        notes.append(f"longitude out of [-180,180]: [{lon.min():.3f},{lon.max():.3f}]")
    if lat.min() < -90 or lat.max() > 90:
        notes.append(f"latitude out of [-90,90]: [{lat.min():.3f},{lat.max():.3f}]")
    if len(lon) > 1 and np.ptp(lon) == 0 and np.ptp(lat) == 0:
        notes.append("all coordinates identical")
    return (len(notes) == 0), notes


def probe_geotiff_georeferencing(logger, sample_n: int = 12) -> dict:
    """Inspect the actual rasters for embedded georeferencing.

    The CSV-only audit above is insufficient. The official loader reads the
    imagery with `rasterio`, so the tiles may be true GeoTIFFs carrying a CRS and
    an affine transform. If they are, per-image coordinates ARE available and the
    tier is T1 -- a materially stronger design than the table audit alone would
    reveal. This must be determined from the files, not inferred from the file
    extension.

    Two conditions are required before declaring georeferencing usable:
      1. a CRS present with plausible bounds, and
      2. the bounds must DIFFER between tiles. A CRS alone is not enough: if
         every tile reports the same extent (a placeholder or a local pixel
         grid), no inter-image distance can be computed from it.
    """
    out: dict = {
        "rasterio_available": False,
        "n_tif_found": 0,
        "n_sampled": 0,
        "n_with_crs": 0,
        "n_with_nonidentity_transform": 0,
        "n_with_plausible_bounds": 0,
        "n_distinct_bounds": 0,
        "n_distinct_crs": 0,
        "crs_list": [],
        "n_geographic_crs_tiles": 0,
        "gsd_metres": None,
        "usable_for_distances": False,
        "examples": [],
        "notes": [],
    }
    try:
        import rasterio
        out["rasterio_available"] = True
    except Exception as exc:                                  # noqa: BLE001
        out["notes"].append(f"rasterio unavailable, cannot probe rasters: {exc}")
        return out

    # Collect IMAGERY only, and sample ACROSS regions.
    #
    # Two failure modes are avoided here:
    #   * `rglob` over data/ also yields the label rasters, which are not the
    #     imagery whose georeferencing is being established. Restricting to paths
    #     under an `images/` directory keeps the probe about imagery.
    #   * Taking the first N hits of a directory walk clusters the sample inside
    #     whichever regions sort first. Since the corpus uses many different CRS
    #     (EPSG:25832, 4326, 3857, several UTM zones, a custom TM), a clustered
    #     sample would report a CRS distribution that is an artefact of the walk
    #     order. The sample is drawn evenly across the sorted image list instead.
    tifs: list[Path] = []
    for p in DATA_DIR.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff"):
            continue
        if p.parent.name.lower() not in {"images", "image", "img", "rgb"}:
            continue
        tifs.append(p)
    tifs.sort()
    out["n_tif_found"] = len(tifs)
    if not tifs:
        out["notes"].append(
            "no imagery found under data/ yet (archive not extracted?) - "
            "raster georeferencing UNVERIFIED")
        return out

    step = max(1, len(tifs) // sample_n)
    sample = tifs[::step][:sample_n]
    out["n_sampled"] = len(sample)

    bounds_seen: set[tuple] = set()
    crs_seen: set[str] = set()
    gsd_seen: list[float] = []
    n_geographic = 0
    for p in sample:
        rec = {"path": str(p.relative_to(DATA_DIR.parent)), "crs": None,
               "transform": None, "bounds": None, "georeferenced": False,
               "gsd_m": None, "error": None}
        try:
            with rasterio.open(p) as src:
                crs = src.crs
                tr = src.transform
                b = src.bounds
                rec["crs"] = crs.to_string() if crs is not None else None
                rec["transform"] = [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]
                rec["bounds"] = [b.left, b.bottom, b.right, b.top]

                identity = (tr.a == 1 and tr.e == 1 and tr.b == 0
                            and tr.d == 0 and tr.c == 0 and tr.f == 0)
                plausible = False
                if crs is not None:
                    if crs.is_geographic:
                        plausible = (-180 <= b.left <= 180 and -90 <= b.bottom <= 90
                                     and -180 <= b.right <= 180 and -90 <= b.top <= 90)
                    else:
                        plausible = abs(b.left) < 2e7 and abs(b.bottom) < 2e7

                # Ground sampling distance in METRES. For a geographic CRS the
                # transform's pixel size is in DEGREES, so reading `transform.a`
                # as metres understates it by ~10^5. Any downstream code that
                # treats the raw pixel size as a metric GSD is wrong for those
                # tiles, so the conversion is done here and recorded.
                if crs is not None and not identity:
                    if crs.is_geographic:
                        n_geographic += 1
                        lat0 = 0.5 * (b.bottom + b.top)
                        m_per_deg_lat = 111_320.0
                        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
                        gsd_x = abs(tr.a) * m_per_deg_lon
                        gsd_y = abs(tr.e) * m_per_deg_lat
                    else:
                        gsd_x, gsd_y = abs(tr.a), abs(tr.e)
                    rec["gsd_m"] = [round(float(gsd_x), 6), round(float(gsd_y), 6)]
                    gsd_seen.append(float(0.5 * (gsd_x + gsd_y)))
                    crs_seen.add(rec["crs"])

                if crs is not None:
                    out["n_with_crs"] += 1
                if not identity:
                    out["n_with_nonidentity_transform"] += 1
                if plausible:
                    out["n_with_plausible_bounds"] += 1
                    bounds_seen.add((round(b.left, 6), round(b.bottom, 6)))
                rec["georeferenced"] = bool(crs is not None and plausible and not identity)
        except Exception as exc:                              # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out["examples"].append(rec)

    out["n_distinct_crs"] = len(crs_seen)
    out["crs_list"] = sorted(crs_seen)
    out["n_geographic_crs_tiles"] = n_geographic
    out["gsd_metres"] = {
        "min": round(min(gsd_seen), 4) if gsd_seen else None,
        "max": round(max(gsd_seen), 4) if gsd_seen else None,
        "distinct_rounded": sorted({round(g, 3) for g in gsd_seen}),
        "note": "derived from each raster's affine transform; geographic CRS "
                "pixel sizes are converted from degrees to metres",
    }
    out["n_distinct_bounds"] = len(bounds_seen)
    thr = max(1, int(0.8 * out["n_sampled"]))
    out["usable_for_distances"] = bool(
        out["n_with_crs"] >= thr
        and out["n_with_plausible_bounds"] >= thr
        and out["n_distinct_bounds"] > 1
    )

    if out["n_with_crs"] == 0:
        out["notes"].append(
            "no CRS on any sampled tile: the distributed imagery is not "
            "georeferenced, so per-image coordinates CANNOT be derived from the "
            "rasters. Region-level grouping is the only available spatial "
            "structure unless an external gazetteer is used.")
    elif not out["usable_for_distances"]:
        out["notes"].append(
            f"a CRS is present on {out['n_with_crs']}/{out['n_sampled']} sampled "
            f"tiles but only {out['n_distinct_bounds']} distinct extent(s) were "
            f"seen, so inter-image distances are not recoverable from the "
            f"georeferencing alone")
    else:
        out["notes"].append(
            f"georeferenced rasters confirmed: {out['n_with_crs']}/{out['n_sampled']} "
            f"sampled tiles carry a CRS with {out['n_distinct_bounds']} distinct "
            f"extents -> per-image distances are computable (tier T1)")
        out["notes"].append(
            f"CRS diversity: {out['n_distinct_crs']} distinct CRS across the sample "
            f"({out['n_geographic_crs_tiles']} tiles in a geographic CRS). "
            f"Distances must therefore be computed AFTER reprojecting to a common "
            f"frame; subtracting coordinates across CRS is not a distance.")
        g = out.get("gsd_metres") or {}
        out["notes"].append(
            f"ground sampling distance (metres, from the transforms): "
            f"min={g.get('min')} max={g.get('max')} distinct={g.get('distinct_rounded')}")

    logger.info("geotiff probe: found=%d sampled=%d crs=%d distinct_bounds=%d "
                "distinct_crs=%d geographic=%d usable=%s",
                out["n_tif_found"], out["n_sampled"], out["n_with_crs"],
                out["n_distinct_bounds"], out["n_distinct_crs"],
                out["n_geographic_crs_tiles"], out["usable_for_distances"])
    if out.get("gsd_metres"):
        logger.info("gsd (m): min=%s max=%s distinct=%s",
                    out["gsd_metres"]["min"], out["gsd_metres"]["max"],
                    out["gsd_metres"]["distinct_rounded"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default=None,
                    help="override: metadata table to inspect")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("spatial_audit", LOGS_DIR / "spatial_metadata_audit.log")
    logger.info("=" * 72)
    logger.info("SPATIAL METADATA AUDIT  %s", utcnow())
    logger.info("=" * 72)

    result: dict = {
        "audited_at_utc": utcnow(),
        "tier": None,
        "evidence": [],
        "candidate_columns": {},
        "per_image_coordinates": False,
        "region_level_coordinates": False,
        "notes": [],
    }

    # ---- locate candidate metadata tables --------------------------------- #
    tables: list[Path] = []
    if args.metadata_csv:
        tables = [Path(args.metadata_csv)]
    else:
        tables = [p for p in METADATA_DIR.glob("*.csv")
                  if p.name not in {"class_statistics.csv", "region_statistics.csv",
                                    "duplicates_exact.csv", "duplicates_near.csv"}]
        inv = METADATA_DIR / "dataset_inventory.csv"
        if inv.exists():
            tables.insert(0, inv)

    if not tables:
        result["notes"].append(
            "no metadata table found; run scripts/build_inventory.py first")
        result["tier"] = "T3"
        write_json(METADATA_DIR / "spatial_metadata_audit.json", result)
        logger.warning("no tables to audit -> tier T3")
        return 0

    # ---- inspect each table ---------------------------------------------- #
    for t in tables:
        try:
            df = pd.read_csv(t, keep_default_na=False, dtype=str, nrows=200_000)
        except Exception as exc:                        # noqa: BLE001
            result["evidence"].append(f"{t.name}: unreadable ({exc})")
            continue

        cols_lower = {c.lower().strip(): c for c in df.columns}
        lon_col = next((cols_lower[h] for h in LON_HINTS if h in cols_lower), None)
        lat_col = next((cols_lower[h] for h in LAT_HINTS if h in cols_lower), None)

        if lon_col and lat_col:
            try:
                lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(float)
                lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(float)
                ok = np.isfinite(lon) & np.isfinite(lat)
                good, notes = plausibility(lon[ok], lat[ok])
                result["candidate_columns"][t.name] = {
                    "lon": lon_col, "lat": lat_col,
                    "n_valid": int(ok.sum()), "n_total": int(len(df)),
                    "plausible": bool(good), "notes": notes,
                }
                if good and ok.sum() > 0:
                    is_inventory = t.name == "dataset_inventory.csv"
                    if is_inventory:
                        result["per_image_coordinates"] = True
                    else:
                        result["region_level_coordinates"] = True
                    result["evidence"].append(
                        f"{t.name}: usable coordinates in '{lon_col}'/'{lat_col}' "
                        f"({int(ok.sum())}/{len(df)} valid)")
                else:
                    result["evidence"].append(
                        f"{t.name}: coordinate columns present but not plausible: {notes}")
            except Exception as exc:                    # noqa: BLE001
                result["evidence"].append(f"{t.name}: coordinate parse failed ({exc})")
            continue

        # ---- look for coordinates embedded in a text column --------------- #
        found = False
        for c in df.columns:
            if df[c].dtype != object:
                continue
            sample = df[c].dropna().astype(str).head(200)
            hits = 0
            for v in sample:
                m = GEO_PATTERNS[0].search(v)
                if m:
                    a, b = float(m.group(1)), float(m.group(2))
                    if -180 <= a <= 180 and -90 <= b <= 90:
                        hits += 1
            if hits >= max(5, int(0.5 * len(sample))):
                result["evidence"].append(
                    f"{t.name}: embedded coordinates detected in column '{c}' "
                    f"({hits}/{len(sample)} sampled values parse as lon/lat) — "
                    f"requires a dedicated parser before use")
                found = True
        if not found:
            result["evidence"].append(f"{t.name}: no coordinate columns detected")

    # ---- probe the rasters themselves ------------------------------------- #
    # A CSV-only audit would miss georeferenced GeoTIFFs, which would silently
    # understate the achievable tier. Check the actual imagery.
    geo = probe_geotiff_georeferencing(logger)
    result["geotiff_probe"] = geo
    if geo["usable_for_distances"]:
        result["per_image_coordinates"] = True
        result["evidence"].append(
            f"raster probe: {geo['n_with_crs']}/{geo['n_sampled']} sampled tiles "
            f"carry a CRS with {geo['n_distinct_bounds']} distinct extents "
            f"-> per-image coordinates recoverable from the GeoTIFFs")
    else:
        result["evidence"].append(
            "raster probe: " + "; ".join(geo["notes"]))

    # ---- decide the tier -------------------------------------------------- #
    if result["per_image_coordinates"]:
        tier = "T1"
        result["notes"].append(
            "Per-image coordinates available: continuous distances computable and "
            "the dependence range r is estimable.")
    elif result["region_level_coordinates"]:
        tier = "T2"
        result["notes"].append(
            "Only region-level coordinates: distances are region-scale; within-region "
            "resolution is unavailable, so block sizes cannot be finer than the region "
            "scale. State this limitation wherever distance-resolved results appear.")
    else:
        tier = "T3"
        result["notes"].append(
            "No usable coordinates: spatial structure must be represented by region / "
            "site / acquisition groupings. Under this tier the study MUST NOT report a "
            "numeric dependence range r, MUST NOT claim a distance-similarity "
            "relationship, and MUST NOT assert that any block size is optimal "
            "(research_protocol.md §5).")

    result["tier"] = tier
    write_json(METADATA_DIR / "spatial_metadata_audit.json", result)

    md = [
        "# Spatial Metadata Audit\n",
        f"**Generated**: {utcnow()}  ",
        f"**Coordinate tier determined**: **{tier}**\n",
        "## Why this determination matters\n",
        "The study's distance-resolved analyses (RQ2, and the block-size definition "
        "in S2/S3) depend on the coordinate resolution actually present in the "
        "archive. Assuming a tier the data do not support would produce a numeric "
        "dependence range that is not grounded in measurement. The tier is therefore "
        "determined from evidence and enforced downstream.\n",
        "| Tier | Meaning | Permitted claims |",
        "|---|---|---|",
        "| T1 | per-image coordinates | continuous distances; r estimable |",
        "| T2 | region-level coordinates | region-scale distances; coarse blocks only |",
        "| T3 | no usable coordinates | grouping-level analysis; **no numeric r** |",
        "",
        "## Evidence\n",
    ]
    for e in result["evidence"]:
        md.append(f"- {e}")
    md.append("")

    geo = result.get("geotiff_probe", {})
    if geo:
        md.append("## Raster georeferencing probe\n")
        md.append("Determined from the imagery itself, not from the file extension. "
                  "A CRS is only treated as usable if the extents actually differ "
                  "between tiles.\n")
        md.append("| Quantity | Value |")
        md.append("|---|---|")
        md.append(f"| rasterio available | {geo.get('rasterio_available')} |")
        md.append(f"| .tif files found | {geo.get('n_tif_found')} |")
        md.append(f"| sampled | {geo.get('n_sampled')} |")
        md.append(f"| with CRS | {geo.get('n_with_crs')} |")
        md.append(f"| with non-identity transform | {geo.get('n_with_nonidentity_transform')} |")
        md.append(f"| with plausible geographic bounds | {geo.get('n_with_plausible_bounds')} |")
        md.append(f"| distinct extents | {geo.get('n_distinct_bounds')} |")
        md.append(f"| **usable for distances** | **{geo.get('usable_for_distances')}** |")
        md.append("")
        if geo.get("examples"):
            md.append("Sampled tiles:\n")
            md.append("| file | CRS | transform (a,b,c,d,e,f) | bounds (L,B,R,T) | error |")
            md.append("|---|---|---|---|---|")
            for ex in geo["examples"]:
                md.append(
                    f"| {ex['path']} | {ex['crs'] or '-'} | "
                    f"{ex['transform'] or '-'} | {ex['bounds'] or '-'} | "
                    f"{ex['error'] or '-'} |")
            md.append("")

    if result["candidate_columns"]:
        md.append("## Candidate coordinate columns\n")
        md.append("| Table | lon col | lat col | valid/total | plausible | notes |")
        md.append("|---|---|---|---|---|---|")
        for tname, info in result["candidate_columns"].items():
            md.append(f"| {tname} | {info['lon']} | {info['lat']} | "
                      f"{info['n_valid']}/{info['n_total']} | {info['plausible']} | "
                      f"{'; '.join(info['notes']) or '-'} |")
        md.append("")
    md.append("## Consequence\n")
    for n in result["notes"]:
        md.append(f"- {n}")
    md.append("")
    (METADATA_DIR / "spatial_metadata_audit.md").write_text("\n".join(md),
                                                           encoding="utf-8")

    print()
    print("[PHASE] Spatial metadata audit")
    print(f"[STATUS] PASS (tier {tier})")
    print(f"[FILES] {METADATA_DIR / 'spatial_metadata_audit.md'}")
    print(f"[NEXT] freeze external geographic regions")
    logger.info("tier = %s", tier)
    return 0


if __name__ == "__main__":
    sys.exit(main())

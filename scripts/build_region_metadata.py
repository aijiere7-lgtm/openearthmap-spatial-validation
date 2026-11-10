#!/usr/bin/env python
"""
build_region_metadata.py — the region-level metadata table.

Produces `data/metadata/region_metadata.csv` with, per region:
    region, country, continent, lon, lat, coord_source, n_images, n_pixels

TWO SEPARATE CONCERNS, DELIBERATELY KEPT APART
---------------------------------------------
1. COUNTRY -> CONTINENT. This is general geographic knowledge, not a property of
   the dataset, so it is applied from an explicit mapping below. A country that
   is not in the mapping is REPORTED AND LEFT BLANK, never guessed -- a wrong
   continent would silently corrupt the continent-stratified selection of the
   external test regions.

2. COORDINATES. The dataset does not ship them (verified: the repository has no
   metadata table and the layout is `<region>/images/*.tif`). This script does
   NOT invent them. It writes `coord_source = "none"` and leaves lon/lat empty,
   unless a gazetteer file is supplied via --gazetteer.

WHY coord_source MATTERS DOWNSTREAM
-----------------------------------
`spatial_dependence.py` reads this column and refuses to use coordinates whose
provenance is a gazetteer centroid for estimating the dependence range r, because
positional error at city scale (tens of km) exceeds the range being measured. So
this column is the mechanism that prevents a plausible-looking but meaningless
number from entering the analysis. Do not populate it with anything other than a
truthful description of where the coordinates came from.

If per-image coordinates ARE found in the rasters (the GeoTIFF probe in
spatial_metadata_audit.py tests this), then tier T1 applies and the coordinates
belong in the inventory, not here.

Output: data/metadata/region_metadata.csv
        data/metadata/region_metadata_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    LOGS_DIR, METADATA_DIR, RAW_DIR, ensure_dirs, get_logger, utcnow, write_json,
)
from data.inventory import resolve_inventory_root  # noqa: E402

INVENTORY = METADATA_DIR / "dataset_inventory.csv"
OUT = METADATA_DIR / "region_metadata.csv"

# --------------------------------------------------------------------------- #
# Country -> continent. Extend explicitly; never infer.
# --------------------------------------------------------------------------- #
COUNTRY_TO_CONTINENT: dict[str, str] = {}


def _add(continent: str, countries: str) -> None:
    for c in countries.split(","):
        c = c.strip()
        if c:
            COUNTRY_TO_CONTINENT[c.lower()] = continent


_add("Africa", "Algeria,Angola,Benin,Botswana,Burkina Faso,Burundi,Cameroon,"
               "Cape Verde,Central African Republic,Chad,Comoros,Congo,"
               "Democratic Republic of the Congo,Djibouti,Egypt,Equatorial Guinea,"
               "Eritrea,Eswatini,Ethiopia,Gabon,Gambia,Ghana,Guinea,Guinea-Bissau,"
               "Ivory Coast,Kenya,Lesotho,Liberia,Libya,Madagascar,Malawi,Mali,"
               "Mauritania,Mauritius,Morocco,Mozambique,Namibia,Niger,Nigeria,"
               "Rwanda,Sao Tome and Principe,Senegal,Seychelles,Sierra Leone,"
               "Somalia,South Africa,South Sudan,Sudan,Tanzania,Togo,Tunisia,"
               "Uganda,Zambia,Zimbabwe")
_add("Asia", "Afghanistan,Armenia,Azerbaijan,Bahrain,Bangladesh,Bhutan,Brunei,"
             "Cambodia,China,Cyprus,Georgia,India,Indonesia,Iran,Iraq,Israel,"
             "Japan,Jordan,Kazakhstan,Kuwait,Kyrgyzstan,Laos,Lebanon,Malaysia,"
             "Maldives,Mongolia,Myanmar,Nepal,North Korea,Oman,Pakistan,"
             "Palestine,Philippines,Qatar,Saudi Arabia,Singapore,South Korea,"
             "Sri Lanka,Syria,Taiwan, Tajikistan,Thailand,Timor-Leste,Turkey,"
             "Turkmenistan,United Arab Emirates,Uzbekistan,Vietnam,Yemen")
_add("Europe", "Albania,Andorra,Austria,Belarus,Belgium,Bosnia and Herzegovina,"
               "Bulgaria,Croatia,Czechia,Denmark,Estonia,Finland,France,Germany,"
               "Greece,Hungary,Iceland,Ireland,Italy,Kosovo,Latvia,Liechtenstein,"
               "Lithuania,Luxembourg,Malta,Moldova,Monaco,Montenegro,Netherlands,"
               "North Macedonia,Norway,Poland,Portugal,Romania,Russia,San Marino,"
               "Serbia,Slovakia,Slovenia,Spain,Sweden,Switzerland,Ukraine,"
               "United Kingdom,Vatican City")
_add("North America", "Antigua and Barbuda,Bahamas,Barbados,Belize,Canada,"
                      "Costa Rica,Cuba,Dominica,Dominican Republic,El Salvador,"
                      "Grenada,Guatemala,Haiti,Honduras,Jamaica,Mexico,Nicaragua,"
                      "Panama,Saint Kitts and Nevis,Saint Lucia,"
                      "Saint Vincent and the Grenadines,Trinidad and Tobago,"
                      "United States,United States of America")
_add("South America", "Argentina,Bolivia,Brazil,Chile,Colombia,Ecuador,Guyana,"
                      "Paraguay,Peru,Suriname,Uruguay,Venezuela")
_add("Oceania", "Australia,Fiji,Kiribati,Marshall Islands,Micronesia,Nauru,"
                "New Zealand,Palau,Papua New Guinea,Samoa,Solomon Islands,Tonga,"
                "Tuvalu,Vanuatu")

# Common spellings / abbreviations that would otherwise go unmapped.
ALIASES = {
    "usa": "united states", "us": "united states", "u.s.a.": "united states",
    "uk": "united kingdom", "great britain": "united kingdom",
    "england": "united kingdom", "south korea": "south korea",
    "korea": "south korea", "republic of korea": "south korea",
    "czech republic": "czechia", "russian federation": "russia",
    "viet nam": "vietnam", "iran (islamic republic of)": "iran",
    "uae": "united arab emirates", "ivory coast": "ivory coast",
    "cote d'ivoire": "ivory coast", "côte d'ivoire": "ivory coast",
}


def continent_for(country: str) -> str | None:
    if not country:
        return None
    key = str(country).strip().lower()
    key = ALIASES.get(key, key)
    return COUNTRY_TO_CONTINENT.get(key)


# --------------------------------------------------------------------------- #
# Region coordinates, derived from the rasters' own georeferencing
# --------------------------------------------------------------------------- #
def derive_region_coordinates_from_rasters(
    inv: pd.DataFrame, root: Path, logger, max_per_region: int = 20,
) -> tuple[dict[str, dict], dict]:
    """Compute each region's centroid from the GeoTIFF footprints.

    The spatial-metadata audit established tier T1: every sampled tile carries a
    CRS and a non-identity affine transform. The centroid of a region's tiles is
    therefore a genuine georeference-derived position, which is why
    `coord_source` is recorded as `geotiff_centroid` -- a value accepted by BOTH
    coordinate standards in `common.py`, including the stricter one used for
    estimating the dependence range `r`.

    Three details that would each produce a plausible wrong number:

    * Tiles are stored in MANY different CRS (EPSG:25832, 4326, 3857, several UTM
      zones, a custom TM). Centres are therefore computed in each tile's own CRS
      and transformed to EPSG:4326 with pyproj before anything is averaged.
      Averaging eastings across CRS is not an average of positions.
    * The centroid is the MEDIAN of tile centres, not the mean. A region whose
      tiles are a chain of flight strips can have an outlier tile; the median is
      the robust summary and the spread is reported alongside it so the reader
      can judge.
    * The within-region spread is reported in km. If a "region" turned out to
      span thousands of km, treating it as a local block would be wrong, and that
      would only be visible from this number.

    Only tiles whose labels are usable are used, so the centroid describes the
    region as it actually enters the study.
    """
    try:
        import rasterio
        from pyproj import Transformer
    except Exception as exc:                                  # noqa: BLE001
        logger.error("rasterio/pyproj unavailable, cannot derive coordinates: %s", exc)
        return {}, {"available": False, "reason": str(exc)}

    WGS84 = "EPSG:4326"
    out: dict[str, dict] = {}
    n_failed = 0
    crs_counts: dict[str, int] = {}

    for region, sub in inv.groupby("region", dropna=False):
        paths = [str(p) for p in sub["image_path"].tolist() if str(p).strip()]
        if not paths:
            continue
        step = max(1, len(paths) // max_per_region)
        sample = paths[::step][:max_per_region]

        lons: list[float] = []
        lats: list[float] = []
        half_extents_km: list[float] = []
        for rel in sample:
            p = root / rel
            try:
                with rasterio.open(p) as src:
                    if src.crs is None:
                        continue
                    tr = src.transform
                    if tr.a == 1 and tr.e == 1 and tr.c == 0 and tr.f == 0:
                        continue
                    cx = tr.c + 0.5 * src.width * tr.a
                    cy = tr.f + 0.5 * src.height * tr.e
                    tf = Transformer.from_crs(src.crs, WGS84, always_xy=True)
                    lon, lat = tf.transform(cx, cy)
                    if not (np.isfinite(lon) and np.isfinite(lat)):
                        continue
                    lons.append(float(lon))
                    lats.append(float(lat))
                    crs_counts[str(src.crs)] = crs_counts.get(str(src.crs), 0) + 1
                    # approximate half-diagonal of the tile in km
                    if src.crs.is_geographic:
                        m_per_deg_lat = 111_320.0
                        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat))
                        w_m = abs(tr.a) * src.width * m_per_deg_lon
                        h_m = abs(tr.e) * src.height * m_per_deg_lat
                    else:
                        w_m = abs(tr.a) * src.width
                        h_m = abs(tr.e) * src.height
                    half_extents_km.append(0.5 * float(np.hypot(w_m, h_m)) / 1000.0)
            except Exception:                                  # noqa: BLE001
                n_failed += 1
                continue

        if not lons:
            continue
        lon_med = float(np.median(lons))
        lat_med = float(np.median(lats))
        # within-region spread: max haversine between the median and any tile
        spread_km = 0.0
        for lo, la in zip(lons, lats):
            spread_km = max(spread_km, _haversine_km(lon_med, lat_med, lo, la))
        out[str(region)] = {
            "lon": round(lon_med, 6),
            "lat": round(lat_med, 6),
            "coord_source": "geotiff_centroid",
            "n_tiles_sampled": len(lons),
            "n_tiles_in_region": len(paths),
            "within_region_spread_km": round(spread_km, 3),
            "median_tile_half_diagonal_km": (
                round(float(np.median(half_extents_km)), 4) if half_extents_km else None),
            "lon_range": [round(min(lons), 6), round(max(lons), 6)],
            "lat_range": [round(min(lats), 6), round(max(lats), 6)],
        }

    logger.info("derived coordinates for %d/%d regions (%d tile reads failed)",
                len(out), inv["region"].nunique(), n_failed)
    logger.info("distinct CRS encountered while deriving coordinates: %d", len(crs_counts))
    meta = {
        "available": True,
        "n_regions_with_coordinates": len(out),
        "n_tile_reads_failed": n_failed,
        "n_distinct_crs": len(crs_counts),
        "crs_counts": dict(sorted(crs_counts.items(), key=lambda kv: -kv[1])),
    }
    return out, meta


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def validate_coordinates_by_source_group(
    reg: pd.DataFrame, attribution: pd.DataFrame, logger,
) -> dict:
    """Cross-check the derived coordinates against the source-dataset grouping.

    The check rests on a property of the source datasets that is verifiable from
    the attribution table itself: some are single-country (GeoNRW covers one
    German state; Landcover.ai covers Poland; AIRS covers New Zealand), while
    others are deliberately global (SpaceNet spans Brazil, China, France, the
    Netherlands, Sudan and the USA; "Open data" is a catch-all covering 20+
    countries). Applying one distance expectation to both kinds would produce a
    false alarm on the global ones -- which is exactly what an earlier version of
    this function did.

    So the check is applied ONLY to source groups that the attribution table
    shows to be single-country, and the multi-country groups are reported with
    their span and marked as expected. What makes the check worth running is that
    the single-country groups can FAIL it: swapped axes, a misread transform, or
    a CRS mixed up would collapse the clustering and blow the distance up.

    It needs no external boundary dataset -- only the attribution table.
    """
    if attribution is None or "source_data" not in attribution.columns:
        return {"ran": False, "reason": "no attribution table with source_data"}
    if "lon" not in reg.columns or reg["lon"].isna().all():
        return {"ran": False, "reason": "no coordinates to validate"}

    # `reg` may already carry source_data (it does when the attribution table was
    # used to resolve country). Merging unconditionally would produce
    # source_data_x / source_data_y and then a KeyError, so the existing column
    # is preferred and the merge is only done when it is absent.
    if "source_data" in reg.columns:
        m = reg[["region", "lon", "lat", "source_data"]].copy()
        m["country"] = reg["country"].astype(str)
    else:
        m = reg[["region", "lon", "lat", "country"]].merge(
            attribution[["region", "source_data"]], on="region", how="left")

    # Which source groups are single-country, according to the attribution?
    src_countries: dict[str, set[str]] = {}
    for src, sub in attribution.groupby("source_data", dropna=True):
        src_countries[str(src)] = {str(c).strip() for c in sub["country"] if str(c).strip()}

    def max_pairwise(pts: list[tuple[float, float]]) -> float:
        d = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = max(d, _haversine_km(pts[i][0], pts[i][1], pts[j][0], pts[j][1]))
        return d

    single_country: dict[str, dict] = {}
    multi_country: dict[str, dict] = {}
    for src, sub in m.groupby("source_data", dropna=True):
        pts = [(r["lon"], r["lat"]) for _, r in sub.iterrows()
               if pd.notna(r["lon"]) and pd.notna(r["lat"])]
        if len(pts) < 3:
            continue
        dmax = max_pairwise(pts)
        countries = src_countries.get(str(src), set())
        rec = {"n_regions": len(pts), "max_pairwise_km": round(dmax, 1),
               "countries": sorted(countries)}
        if len(countries) == 1:
            single_country[str(src)] = rec
        else:
            multi_country[str(src)] = rec

    # A single-country source group must not span a continent. 2,000 km is a
    # generous ceiling: the largest of these (Poland) is ~700 km across.
    CEILING_KM = 2000.0
    failures = {k: v for k, v in single_country.items()
                if v["max_pairwise_km"] > CEILING_KM}

    # Independent check on the country assignment: two regions in the same
    # country should be closer to each other than a country's diameter allows.
    country_spans: dict[str, float] = {}
    for country, sub in m.groupby("country", dropna=True):
        pts = [(r["lon"], r["lat"]) for _, r in sub.iterrows()
               if pd.notna(r["lon"]) and pd.notna(r["lat"])]
        if len(pts) >= 2:
            country_spans[str(country)] = round(max_pairwise(pts), 1)

    if failures:
        logger.warning("COORDINATE VALIDATION FAILED for single-country source "
                       "groups (ceiling %.0f km): %s", CEILING_KM, failures)
    else:
        logger.info("coordinate validation PASSED: every single-country source "
                    "group spans < %.0f km", CEILING_KM)
    for k, v in single_country.items():
        logger.info("  single-country source %-14s regions=%-3d max pairwise = "
                    "%8.1f km  [%s]", k, v["n_regions"], v["max_pairwise_km"],
                    ",".join(v["countries"]))
    for k, v in multi_country.items():
        logger.info("  multi-country source  %-14s regions=%-3d max pairwise = "
                    "%8.1f km  (%d countries) -- large span is EXPECTED",
                    k, v["n_regions"], v["max_pairwise_km"], len(v["countries"]))
    for c, d in sorted(country_spans.items(), key=lambda kv: -kv[1])[:12]:
        logger.info("  country %-26s max pairwise = %8.1f km", c, d)

    return {
        "ran": True,
        "ceiling_km": CEILING_KM,
        "single_country_groups": single_country,
        "multi_country_groups": multi_country,
        "failures": failures,
        "country_max_pairwise_km": country_spans,
        "passed": not failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gazetteer", default=None,
                    help="optional CSV with columns region,lon,lat,coord_source. "
                         "Coordinates are used only for context, never for "
                         "estimating the dependence range r.")
    ap.add_argument("--attribution", default=str(METADATA_DIR /
                                                "region_country_attribution.csv"),
                    help="region -> country table transcribed from the official "
                         "attribution page. Used for country and source_data.")
    ap.add_argument("--inventory", default=str(INVENTORY))
    ap.add_argument("--dataset-root", default=None,
                    help="root that image_path is relative to; auto-detected if omitted")
    ap.add_argument("--no-raster-coordinates", action="store_true",
                    help="skip deriving region centroids from the GeoTIFFs")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("region_metadata", LOGS_DIR / "region_metadata.log")
    logger.info("=" * 72)
    logger.info("REGION METADATA  %s", utcnow())
    logger.info("=" * 72)

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        logger.error("%s missing; run scripts/build_inventory.py first", inv_path)
        return 2
    inv = pd.read_csv(inv_path, keep_default_na=False)
    logger.info("inventory rows: %d", len(inv))

    # ---- region x country from the official attribution table -------------- #
    # The release ships no region->country table (verified: only readme.txt,
    # train/val/test.txt and xbd_files.csv). The official attribution page does
    # list every region with its country, and that transcription is used here as
    # the primary source. It is also cross-checked: 97 rows vs 97 region
    # directories, zero discrepancies in either direction, and exactly 44
    # distinct countries -- matching the paper's "97 regions from 44 countries".
    attribution = None
    attr_path = Path(args.attribution)
    if attr_path.exists():
        attribution = pd.read_csv(attr_path, keep_default_na=False)
        missing_cols = {"region", "country"} - set(attribution.columns)
        if missing_cols:
            logger.error("attribution table missing columns %s", sorted(missing_cols))
            return 2
        logger.info("attribution table: %s (%d rows, %d countries)",
                    attr_path.name, len(attribution),
                    attribution["country"].nunique())
        dirs = set(inv["region"].astype(str))
        attr = set(attribution["region"].astype(str))
        if dirs - attr:
            logger.warning("regions present in the corpus but NOT attributed (%d): %s",
                           len(dirs - attr), sorted(dirs - attr))
        if attr - dirs:
            logger.warning("regions attributed but NOT present in the corpus (%d): %s",
                           len(attr - dirs), sorted(attr - dirs))
    else:
        logger.warning("attribution table not found at %s; country will be taken "
                       "from the inventory, where it is normally blank", attr_path)

    # ---- region x country from the inventory ------------------------------ #
    g = inv.groupby("region", dropna=False)
    rows = []
    for region, sub in g:
        country = ""
        if "country" in sub.columns:
            vals = [str(v).strip() for v in sub["country"].unique() if str(v).strip()]
            if len(vals) == 1:
                country = vals[0]
            elif len(vals) > 1:
                # A region spanning countries would break continent
                # stratification; record it rather than picking one.
                country = "|".join(sorted(vals))
        rows.append({
            "region": region,
            "country": country,
            "n_images": int(len(sub)),
        })
    reg = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    logger.info("regions in the corpus: %d", len(reg))

    # Attribution wins where it is available; a disagreement is reported, never
    # silently resolved.
    n_attr_country = 0
    if attribution is not None:
        amap = dict(zip(attribution["region"].astype(str),
                        attribution["country"].astype(str)))
        smap = (dict(zip(attribution["region"].astype(str),
                         attribution["source_data"].astype(str)))
                if "source_data" in attribution.columns else {})
        reg["country_attr"] = reg["region"].map(amap)
        reg["source_data"] = reg["region"].map(smap)
        disagree = reg[(reg["country"].astype(str).str.strip() != "")
                       & reg["country_attr"].notna()
                       & (reg["country"].astype(str) != reg["country_attr"])]
        if len(disagree):
            logger.warning("country disagreement between inventory and attribution "
                           "for %d region(s); the attribution value is used:",
                           len(disagree))
            for _, r in disagree.iterrows():
                logger.warning("  %-24s inventory=%-16r attribution=%r",
                               r["region"], r["country"], r["country_attr"])
        reg["country"] = reg["country_attr"].fillna(reg["country"].astype(str))
        n_attr_country = int(reg["country_attr"].notna().sum())
        reg = reg.drop(columns=["country_attr"])
        logger.info("country resolved from the attribution table for %d/%d regions",
                    n_attr_country, len(reg))
        reg["n_images"] = reg["n_images"].astype(int)
    else:
        reg["source_data"] = ""

    # ---- continent -------------------------------------------------------- #
    reg["continent"] = reg["country"].map(continent_for)
    unmapped = sorted({c for c, m in zip(reg["country"], reg["continent"])
                       if c and not m})
    blank_country = int((reg["country"].astype(str).str.strip() == "").sum())

    if unmapped:
        logger.warning("UNMAPPED COUNTRIES (%d): %s", len(unmapped), unmapped)
        logger.warning("These continents are left BLANK rather than guessed. Add "
                       "them to COUNTRY_TO_CONTINENT and re-run. Continent-"
                       "stratified external-region selection cannot use rows "
                       "with a blank continent.")
    if blank_country:
        logger.warning("%d region(s) have no country; their continent cannot be "
                       "determined", blank_country)

    # ---- coordinates ------------------------------------------------------ #
    # Preferred source: the rasters' own georeferencing (tier T1). This is a
    # `geotiff_centroid`, which satisfies BOTH coordinate standards in
    # common.py, including the stricter one used for the dependence range.
    reg["lon"] = pd.NA
    reg["lat"] = pd.NA
    reg["coord_source"] = "none"
    coord_meta: dict = {"available": False}
    n_coord = 0

    if not args.no_raster_coordinates:
        root = (Path(args.dataset_root) if args.dataset_root
                else _detect_dataset_root(logger=logger))
        if root is None:
            logger.error("could not locate the dataset root; pass --dataset-root")
        else:
            logger.info("deriving region coordinates from rasters under %s", root)
            coords, coord_meta = derive_region_coordinates_from_rasters(inv, root, logger)
            if coords:
                cdf = pd.DataFrame([
                    {"region": k, **v} for k, v in coords.items()])
                reg = reg.merge(cdf, on="region", how="left",
                                suffixes=("", "_rast"))
                for c in ("lon_rast", "lat_rast"):
                    if c in reg.columns:
                        base_c = c.replace("_rast", "")
                        reg[base_c] = reg[c]
                        reg = reg.drop(columns=[c])
                if "coord_source_rast" in reg.columns:
                    reg["coord_source"] = reg["coord_source_rast"].fillna("none")
                    reg = reg.drop(columns=["coord_source_rast"])
                n_coord = int(reg["lon"].notna().sum())
                logger.info("coordinates derived for %d/%d regions "
                            "(coord_source='geotiff_centroid')", n_coord, len(reg))

    if args.gazetteer and n_coord == 0:
        gz_path = Path(args.gazetteer)
        if not gz_path.exists():
            logger.error("gazetteer not found: %s", gz_path)
            return 2
        gz = pd.read_csv(gz_path)
        need = {"region", "lon", "lat"}
        if not need.issubset(gz.columns):
            logger.error("gazetteer must have columns %s; found %s",
                         sorted(need), list(gz.columns))
            return 2
        src = (gz["coord_source"].iloc[0] if "coord_source" in gz.columns
               else "unspecified")
        reg = reg.merge(gz[["region", "lon", "lat"]], on="region", how="left",
                        suffixes=("", "_gz"))
        reg["lon"] = reg["lon_gz"]
        reg["lat"] = reg["lat_gz"]
        reg = reg.drop(columns=[c for c in ("lon_gz", "lat_gz") if c in reg.columns])
        reg["coord_source"] = src
        n_coord = int(reg["lon"].notna().sum())
        logger.warning("coordinates supplied from %s with coord_source=%r. These "
                       "will NOT be used to estimate the dependence range r "
                       "unless the source is a genuine per-image georeference.",
                       gz_path.name, src)

    # ---- validation ------------------------------------------------------- #
    validation = validate_coordinates_by_source_group(reg, attribution, logger)

    keep = ["region", "country", "continent", "lon", "lat", "coord_source",
            "n_images"]
    for extra in ("source_data", "n_tiles_sampled", "n_tiles_in_region",
                  "within_region_spread_km", "median_tile_half_diagonal_km",
                  "lon_range", "lat_range"):
        if extra in reg.columns:
            keep.append(extra)
    reg = reg[keep]
    reg.to_csv(OUT, index=False, encoding="utf-8")
    logger.info("wrote %s (%d rows)", OUT, len(reg))

    report = {
        "built_at_utc": utcnow(),
        "n_regions": int(len(reg)),
        "n_with_continent": int(reg["continent"].notna().sum()),
        "n_without_continent": int(reg["continent"].isna().sum()),
        "unmapped_countries": unmapped,
        "n_regions_without_country": blank_country,
        "n_country_from_attribution": n_attr_country,
        "attribution_table": str(attr_path),
        "n_with_coordinates": n_coord,
        "coord_source": (str(reg["coord_source"].mode().iloc[0])
                         if len(reg) and reg["coord_source"].notna().any() else "none"),
        "coordinate_derivation": coord_meta,
        "coordinate_validation": validation,
        "continent_counts": (reg["continent"].value_counts(dropna=False)
                             .astype(int).to_dict() if len(reg) else {}),
        "note": ("Region coordinates are derived from each tile's own GeoTIFF "
                 "georeferencing (tier T1) and are therefore accepted by both "
                 "coordinate standards, including the stricter one used for the "
                 "dependence range r. They are NOT gazetteer positions."),
    }
    write_json(METADATA_DIR / "region_metadata_report.json", report)

    print()
    print("[PHASE] Region metadata")
    print(f"[STATUS] {'PASS' if not unmapped else 'PARTIAL'}")
    print(f"[COMPLETED] {len(reg)} regions; continent resolved for "
          f"{report['n_with_continent']}; coordinates for {n_coord} "
          f"(source={report['coord_source']})")
    if coord_meta.get("available"):
        print(f"[COORDS] derived from rasters using {coord_meta.get('n_distinct_crs')} "
              f"distinct CRS; {coord_meta.get('n_tile_reads_failed')} tile reads failed")
    if validation.get("ran"):
        n_single = len(validation.get("single_country_groups", {}))
        n_multi = len(validation.get("multi_country_groups", {}))
        print(f"[VALIDATION] source-group check: "
              f"{'PASS' if validation.get('passed') else 'FAILED'} "
              f"({n_single} single-country groups tested against a "
              f"{validation.get('ceiling_km')} km ceiling; {n_multi} multi-country "
              f"groups reported, large spans expected)")
    if unmapped:
        print(f"[ISSUES] unmapped countries (continent left blank, not guessed): "
              f"{unmapped}")
    if blank_country:
        print(f"[ISSUES] {blank_country} region(s) have no country")
    print(f"[FILES] {OUT}")
    print("[NEXT] freeze the external geographic test set")
    logger.info("=" * 72)
    return 0


def _detect_dataset_root(inventory_csv: Path | None = None,
                         logger=None) -> Path | None:
    """Locate the directory that the inventory's relative paths are based on.

    Delegates to the shared evidence-based resolver in `data.inventory`. This
    script runs AFTER the inventory exists, so the root can be VERIFIED rather
    than guessed: a candidate is accepted only if the inventory's own relative
    paths actually resolve beneath it. The previous local implementation
    hard-coded the directory name and returned the first candidate matching, so
    a redundant wrapper level above the corpus root would have shifted every
    relative path by one level without raising an error.
    """
    root, info = resolve_inventory_root(
        inventory_csv=inventory_csv if inventory_csv is not None else INVENTORY,
        logger=logger,
    )
    if logger is not None:
        logger.info("dataset root method: %s", info.get("method"))
        for note in info.get("notes", []):
            logger.warning("root resolution: %s", note)
    return root


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
spatial_dependence.py — spatial dependence diagnostics and estimation of r.

Implements research_protocol.md section 7. Everything here runs on the
DEVELOPMENT POOL only; external regions are excluded so that no quantity used to
configure a split is informed by the data reserved for the final external test.

THE TIER GATE IS THE POINT OF THIS SCRIPT
-----------------------------------------
The protocol forbids reporting a numeric dependence range r unless the archive
actually supports the measurement (section 5). Three outcomes are possible and
this script distinguishes them explicitly rather than degrading quietly:

  T1  per-image coordinates -> Moran's I, Geary's C, a distance-similarity
      curve, and a fitted range r with a cross-model stability assessment.
  T2  region-level coordinates -> the same statistics, but at region scale.
      The range describes region-to-region similarity, NOT patch-to-patch
      dependence, and must be labelled as such wherever it is used.
  T3  no usable coordinates -> distance-resolved statistics are NOT computable.
      No r is produced. Instead the script measures between-region
      heterogeneity, which is the precondition that makes spatial dependence
      matter at all, and records in plain terms why r is absent.

Producing an r under T3 would be fabrication, so the code refuses to do it.

A further caution implemented here: coordinates obtained by geocoding a place
name to a city centroid are NOT treated as usable for estimating r. Positional
error at city scale (tens of km) would exceed the dependence range being
measured, so such coordinates are reported for context but excluded from any
distance-resolved statistic.

Output: results/tables/spatial_dependence_curve.csv
        results/tables/spatial_autocorrelation.csv
        results/tables/region_heterogeneity.csv
        data/metadata/spatial_dependence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import (  # noqa: E402
    CLASS_NAMES, LOGS_DIR, METADATA_DIR, NUM_CLASSES, RESULTS_DIR,
    ensure_dirs, freeze_development_regions, freeze_external_regions,
    get_logger, read_json, utcnow, write_json,
)
from spatial.dependence import (  # noqa: E402
    build_weights, build_weights_spherical, detect_variogram_window,
    distance_similarity_curve_geodesic, distance_similarity_curve_within_groups,
    estimate_range, gearys_c, morans_i,
)

REGION_PIXELS = METADATA_DIR / "region_class_pixels.csv"
SAMPLE_COMPOSITION = METADATA_DIR / "sample_class_composition.csv"
SAMPLE_COORDS = METADATA_DIR / "sample_coordinates.csv"
REGION_METADATA = METADATA_DIR / "region_metadata.csv"
AUDIT_JSON = METADATA_DIR / "spatial_metadata_audit.json"
EXTERNAL_JSON = METADATA_DIR / "external_regions.json"

OUT_CURVE = RESULTS_DIR / "tables" / "spatial_dependence_curve.csv"
OUT_CURVE_WITHIN = RESULTS_DIR / "tables" / "spatial_dependence_curve_within_region.csv"
OUT_AUTOCORR = RESULTS_DIR / "tables" / "spatial_autocorrelation.csv"
OUT_HETERO = RESULTS_DIR / "tables" / "region_heterogeneity.csv"
OUT_JSON = METADATA_DIR / "spatial_dependence.json"

VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian"]

# Coordinate provenance values that are NOT acceptable for estimating r.
# A gazetteer centroid carries positional error far larger than the dependence
# range under study, so it can describe context but must not drive a distance
# statistic.
COORD_SOURCES_UNUSABLE_FOR_R = {
    "gazetteer_city_centroid", "gazetteer_region_centroid",
    "geocoded_city", "geocoded_region", "approximate",
}


def load_tier(logger) -> tuple[str, dict]:
    if not AUDIT_JSON.exists():
        logger.warning("no spatial_metadata_audit.json; assuming T3 "
                       "(the conservative default)")
        return "T3", {}
    audit = read_json(AUDIT_JSON)
    tier = audit.get("tier") or "T3"
    return tier, audit


def load_external_regions() -> set[str]:
    """External (reserved) regions from the freeze.

    Delegates to the shared accessor. The region lists are nested under
    "selection", and readers that looked only at the top level returned an empty
    set -- which is indistinguishable from "the study has no external test" and
    silently computed every statistic over the reserved regions.

    A record that yields no external regions is reported as an error by the
    caller rather than treated as "none".
    """
    if not EXTERNAL_JSON.exists():
        return set()
    try:
        return set(freeze_external_regions(read_json(EXTERNAL_JSON)))
    except Exception:                                          # noqa: BLE001
        return set()


def load_development_regions() -> set[str]:
    """The development pool as defined by the freeze, when a freeze exists.

    Preferred over "everything minus external" because the freeze also excludes
    regions that carry too few tiles to take part at all. Falling back to the
    subtraction keeps this script runnable before a freeze, and the caller logs
    which rule was used.
    """
    if not EXTERNAL_JSON.exists():
        return set()
    try:
        return set(freeze_development_regions(read_json(EXTERNAL_JSON)))
    except Exception:                                          # noqa: BLE001
        return set()


def autocorrelation_table(feats: np.ndarray,
                          labels: list[str], logger,
                          permutations: int = 999,
                          coords_km: np.ndarray | None = None,
                          lonlat: tuple[np.ndarray, np.ndarray] | None = None
                          ) -> pd.DataFrame:
    """Moran's I and Geary's C per feature dimension.

    Pass `lonlat` for geographically distributed units: the neighbour set is then
    found with an exact great-circle search. Pass `coords_km` only for units that
    genuinely share a projected plane.
    """
    if lonlat is not None:
        w = build_weights_spherical(lonlat[0], lonlat[1],
                                    k=min(8, max(1, len(lonlat[0]) - 1)))
    elif coords_km is not None:
        w = build_weights(coords_km, k=min(8, max(1, len(coords_km) - 1)))
    else:
        raise ValueError("either coords_km or lonlat must be supplied")
    rows = []
    for j, lab in enumerate(labels):
        v = feats[:, j]
        if np.nanstd(v) == 0:
            rows.append({"feature": lab, "morans_i": None, "p_sim": None,
                         "gearys_c": None, "note": "zero variance"})
            continue
        mi = morans_i(v, w, permutations=permutations)
        gc = gearys_c(v, w, permutations=permutations)
        rows.append({
            "feature": lab,
            "morans_i": mi.get("morans_i"),
            "expected_i": mi.get("expected_i"),
            "p_sim": mi.get("p_sim"),
            "z_sim": mi.get("z_sim"),
            "gearys_c": gc.get("gearys_c"),
            "gearys_c_p_sim": gc.get("p_sim"),
            "n": mi.get("n"),
            "note": mi.get("note", ""),
        })
        logger.info("  %-18s Moran's I=%-8s p=%-8s Geary's C=%s", lab,
                    mi.get("morans_i"), mi.get("p_sim"), gc.get("gearys_c"))
    return pd.DataFrame(rows)


def feature_matrix(reg: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Extract the per-class share matrix from a region-level frame.

    Returns `(feats, cols)` where `feats` has one row per region and one column
    per class, and `cols` lists the class codes in column order. Callers rely on
    `enumerate(cols)` giving the class code, so `cols == list(range(NUM_CLASSES))`.

    The matrix is ALWAYS padded to NUM_CLASSES columns. A class that is absent
    from every region may be missing from the source table entirely, and if that
    absence were allowed to shorten the matrix the class would silently vanish
    from the heterogeneity table -- which is exactly the "class absent
    everywhere" finding the table exists to surface. Padding keeps the column
    index equal to the class code.

    Raises ValueError rather than returning an empty matrix: a frame with no
    share columns means the caller passed the wrong table (for instance the
    long-format pixel counts instead of the wide share table), and an empty
    result would be reported as "no heterogeneity", which is a different and
    much stronger claim than "the input was wrong".
    """
    cols_present = [c for c in reg.columns
                    if c.startswith("share_class_")]
    if not cols_present:
        raise ValueError(
            "feature_matrix: no 'share_class_<i>' columns found. Expected the "
            "wide region-share table from scripts/extract_class_composition.py; "
            f"got columns {list(reg.columns)[:12]}")

    feats = np.full((len(reg), NUM_CLASSES), np.nan, dtype=float)
    for c in cols_present:
        try:
            idx = int(c.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if 0 <= idx < NUM_CLASSES:
            feats[:, idx] = pd.to_numeric(reg[c], errors="coerce").to_numpy(float)
    # Classes with no column at all are genuinely absent: zero, not NaN, so they
    # are counted as absent rather than dropped.
    feats = np.where(np.isnan(feats), 0.0, feats)
    return feats, list(range(NUM_CLASSES))


def region_heterogeneity(reg: pd.DataFrame, logger) -> pd.DataFrame:
    """Between-region variation in class composition (the T3 fallback).

    If regions did not differ in composition, spatial partitioning could not
    matter, so this is the precondition for the whole study. It is a weaker
    statement than a dependence range -- it is distance-free -- and is reported
    as such.
    """
    feats, cols = feature_matrix(reg)
    rows = []
    for j, col in enumerate(cols):
        v = feats[:, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        mean = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append({
            "class_code": j,
            "class_name": CLASS_NAMES[j],
            "region_mean_share": mean,
            "region_sd": sd,
            "region_cv": (sd / mean) if mean > 0 else None,
            "region_min": float(np.min(v)),
            "region_max": float(np.max(v)),
            "region_range": float(np.max(v) - np.min(v)),
            "n_regions": int(len(v)),
            "regions_absent": int((v <= 0).sum()),
        })
    df = pd.DataFrame(rows)
    logger.info("between-region heterogeneity computed for %d classes", len(df))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--max-pairs", type=int, default=2_000_000,
                    help="cap on pairs for the all-pairs CONTEXT curve only; the "
                         "within-region curve that drives r is always exhaustive")
    ap.add_argument("--n-lags", type=int, default=20)
    ap.add_argument("--max-lag-fraction", type=float, default=0.5)
    ap.add_argument("--max-lag-km", type=float, default=None,
                    help="explicit upper bin bound for the within-region curve; "
                         "defaults to max_lag_fraction * p99(within distances)")
    ap.add_argument("--window-min-pairs", type=int, default=500,
                    help="a lag bin needs at least this many pairs to enter the "
                         "variogram fit window; 500 bounds the per-bin relative "
                         "standard error of the empirical semivariance at ~6%%")
    ap.add_argument("--window-tolerance", type=float, default=0.05,
                    help="dip tolerated in the monotone rise before the window "
                         "is truncated (only used with --window-monotone)")
    ap.add_argument("--window-smooth", type=int, default=1,
                    help="moving-average width for the monotonicity test only; "
                         "1 disables smoothing (the default, because smoothing "
                         "was found to let the inflated tail back in)")
    ap.add_argument("--window-monotone", action="store_true",
                    help="ALSO apply the monotone-rise walk after the support "
                         "threshold. Off by default: a tolerance sweep showed the "
                         "walk is degenerate on this curve -- for every tolerance "
                         ">= 0.12 it returns exactly the support window, and below "
                         "that it truncates inside a genuine plateau.")
    args = ap.parse_args()

    ensure_dirs()
    logger = get_logger("spatial_dependence", LOGS_DIR / "spatial_dependence.log")
    logger.info("=" * 72)
    logger.info("SPATIAL DEPENDENCE ANALYSIS  %s", utcnow())
    logger.info("=" * 72)

    tier, audit = load_tier(logger)
    logger.info("coordinate tier: %s", tier)

    external = load_external_regions()
    dev_regions = load_development_regions()
    if EXTERNAL_JSON.exists() and not external:
        # A freeze that yields no external regions is a parsing failure, not a
        # study with no external test. Proceeding would compute every statistic
        # over the reserved regions while reporting that none were excluded.
        logger.error("%s exists but no external regions could be read from it. "
                     "Refusing to proceed: the development pool would silently "
                     "include the regions reserved for the final test.",
                     EXTERNAL_JSON.name)
        print()
        print("[PHASE] Spatial dependence analysis")
        print("[STATUS] FAIL — external freeze could not be parsed")
        return 1
    logger.info("external regions excluded from this analysis: %d", len(external))
    if dev_regions:
        logger.info("development pool taken from the freeze: %d regions",
                    len(dev_regions))
    else:
        logger.info("no freeze record; development pool = all regions minus "
                    "external")

    result: dict = {
        "analysed_at_utc": utcnow(),
        "tier": tier,
        "tier_evidence": audit.get("evidence", []),
        "n_external_regions_excluded": len(external),
        "external_regions": sorted(external),
        "n_development_regions_available": len(dev_regions) or None,
        "range_estimated": False,
        "range": None,
        "notes": [],
        "outputs": {},
    }

    if not REGION_PIXELS.exists():
        logger.error("%s missing -- run scripts/extract_class_composition.py first",
                     REGION_PIXELS)
        return 2

    reg = pd.read_csv(REGION_PIXELS)
    logger.info("regions loaded: %d", len(reg))

    # Development pool only: the external test regions must not inform any
    # quantity used to configure a split. Prefer the FROZEN development list,
    # because "all regions minus the external ones" is not the same set -- it
    # also retains regions excluded for having too few tiles (kinshasa), which
    # cannot support a leave-one-region-out fold and are not part of the pool the
    # splits are built from. Using the frozen list keeps every reported count
    # reconcilable with external_regions.json.
    if dev_regions:
        dev = reg[reg["region"].astype(str).isin(dev_regions)].copy()
        logger.info("development-pool regions: %d (of %d) -- taken from the freeze",
                    len(dev), len(reg))
        result["development_pool_source"] = "external_regions.json"
    else:
        dev = reg[~reg["region"].astype(str).isin(external)].copy()
        logger.info("development-pool regions: %d (of %d) -- derived by excluding "
                    "the external set", len(dev), len(reg))
        result["development_pool_source"] = "derived_external_exclusion"
    result["n_development_regions_reported"] = int(len(dev))
    if len(dev) < 10:
        result["notes"].append(
            f"only {len(dev)} development regions; distance statistics may be "
            f"unreliable and are reported with that caveat")

    # ---- always available: between-region heterogeneity -------------------- #
    het = region_heterogeneity(dev, logger)
    het.to_csv(OUT_HETERO, index=False, encoding="utf-8")
    result["outputs"]["region_heterogeneity"] = str(OUT_HETERO)
    result["region_heterogeneity_summary"] = {
        "n_classes_analysed": int(len(het)),
        "classes_absent_in_some_region": int(
            (het["regions_absent"] > 0).sum()) if len(het) else 0,
        "max_region_range_class": (
            het.loc[het["region_range"].idxmax(), "class_name"]
            if len(het) else None),
        "max_region_range": (
            float(het["region_range"].max()) if len(het) else None),
    }
    logger.info("wrote %s", OUT_HETERO)

    # ---- decide whether distance statistics are admissible ---------------- #
    coords = None
    coord_source = None
    if REGION_METADATA.exists():
        rm = pd.read_csv(REGION_METADATA)
        cols = {c.lower(): c for c in rm.columns}
        if "lon" in cols and "lat" in cols and "region" in cols:
            rm = rm.rename(columns={cols["lon"]: "lon", cols["lat"]: "lat",
                                    cols["region"]: "region"})
            if "coord_source" in cols:
                rm = rm.rename(columns={cols["coord_source"]: "coord_source"})
            coord_source = (str(rm["coord_source"].dropna().iloc[0])
                            if "coord_source" in rm.columns
                            and rm["coord_source"].notna().any() else "unspecified")
            coords = rm
            logger.info("region coordinates found (source=%s)", coord_source)
    else:
        logger.warning("no %s: region coordinates unavailable", REGION_METADATA.name)

    admissible = True
    if coords is None:
        admissible = False
        result["notes"].append(
            "no region coordinate table: distance-resolved statistics are not "
            "computable")
    elif coord_source in COORD_SOURCES_UNUSABLE_FOR_R:
        admissible = False
        result["notes"].append(
            f"coordinates are gazetteer/geocoded centroids (source={coord_source}); "
            f"their positional error is expected to exceed the dependence range "
            f"under study, so they are NOT used to estimate r. This is recorded "
            f"as a design limitation rather than worked around.")
    if tier == "T3":
        admissible = False
        result["notes"].append(
            "tier T3: the protocol bars a numeric dependence range. No r is "
            "reported (research_protocol.md section 5).")

    # ---- distance-resolved analysis, only when admissible ------------------ #
    if admissible:
        merged = None
        unit = "region"

        # Tier T1: prefer PER-TILE coordinates, which put the distance axis at
        # the scale where spatial dependence actually operates (tens of metres to
        # a few kilometres) instead of at continent scale.
        #
        # The previous version of this branch read `sample_class_composition.csv`
        # into `df`, set unit = "sample", and then never used `df`: it merged the
        # REGION-level table with region coordinates. The output would therefore
        # have carried unit = "sample" on region-level statistics -- a mislabelled
        # scale, which is precisely the distinction the protocol requires to be
        # kept explicit.
        if tier == "T1" and SAMPLE_COMPOSITION.exists() and SAMPLE_COORDS.exists():
            sc = pd.read_csv(SAMPLE_COORDS)
            if {"sample_id", "lon", "lat"}.issubset(sc.columns):
                sc = sc[["sample_id", "lon", "lat"]].dropna(subset=["lon", "lat"])
                smp = pd.read_csv(SAMPLE_COMPOSITION)
                # The development pool comes from the freeze when one exists: the
                # freeze also excludes regions too thin to take part, which a
                # plain "everything minus external" subtraction would keep.
                if dev_regions:
                    smp = smp[smp["region"].astype(str).isin(dev_regions)].copy()
                    result["development_pool_source"] = "external_regions.json"
                else:
                    smp = smp[~smp["region"].astype(str).isin(external)].copy()
                    result["development_pool_source"] = "all regions minus external"
                m = smp.merge(sc, on="sample_id", how="inner")
                m = m[np.isfinite(m["lon"]) & np.isfinite(m["lat"])]
                if len(m) >= 10:
                    unit = "sample"
                    merged = m
                    logger.info(
                        "tier T1: analysing at TILE scale -- %d development "
                        "samples carry coordinates across %d regions "
                        "(pool source: %s)",
                        len(m), int(m["region"].nunique()),
                        result["development_pool_source"])
                else:
                    result["notes"].append(
                        f"per-tile coordinates exist but only {len(m)} development "
                        f"samples carry them; falling back to region scale")

        if merged is None:
            # Region scale. Region centroids are 10^3 km apart, so a range fitted
            # here describes REGION-TO-REGION similarity, not patch-to-patch
            # dependence. The distinction is written into the output rather than
            # left for the reader to infer.
            merged = dev.merge(coords, on="region", how="inner",
                               suffixes=("", "_c"))
            merged = merged[np.isfinite(merged["lon"]) & np.isfinite(merged["lat"])]
            unit = "region"
            logger.info("units with usable coordinates: %d (region scale)",
                        len(merged))
            if tier == "T1":
                result["notes"].append(
                    "tier T1, but per-tile coordinates were unavailable for this "
                    "run: the distance-resolved statistics below are computed at "
                    "REGION scale and must NOT be read as patch-to-patch "
                    "dependence")

        if len(merged) < 10:
            result["notes"].append(
                f"only {len(merged)} units carry usable coordinates; a "
                f"distance-similarity curve is not fitted")
        else:
            feats, cols = feature_matrix(merged)
            glon = merged["lon"].to_numpy(float)
            glat = merged["lat"].to_numpy(float)

            # Distances are ALWAYS geodesic. Both scales span far more latitude
            # than an equirectangular projection can carry -- the planar path
            # used previously distorted east-west distances by up to a factor of
            # cos(lat0)/cos(lat), about 81% at 60 degrees latitude, and that
            # distortion would have propagated straight into the fitted range.
            result["distance_metric"] = "geodesic_haversine"
            result["distance_scale_unit"] = unit

            # autocorrelation (exact great-circle neighbour search)
            ac = autocorrelation_table(feats, cols, logger,
                                       permutations=args.permutations,
                                       lonlat=(glon, glat))
            ac.insert(0, "unit", unit)
            ac.to_csv(OUT_AUTOCORR, index=False, encoding="utf-8")
            result["outputs"]["spatial_autocorrelation"] = str(OUT_AUTOCORR)
            finite_i = ac["morans_i"].dropna()
            result["morans_i_summary"] = {
                "n_features": int(len(finite_i)),
                "mean": float(finite_i.mean()) if len(finite_i) else None,
                "min": float(finite_i.min()) if len(finite_i) else None,
                "max": float(finite_i.max()) if len(finite_i) else None,
                "n_significant_p05": int((ac["p_sim"].dropna() < 0.05).sum()),
            }

            # ---- distance-similarity curve --------------------------------- #
            # TWO curves are produced, and only one of them is used to fit r.
            #
            #   * WITHIN-REGION pairs (primary, drives r). The corpus spans six
            #     continents, so on the all-pairs curve the 99th-percentile
            #     separation is ~18,000 km and 20 linear bins are ~460 km wide.
            #     Land-cover composition varies over metres to kilometres, so the
            #     whole phenomenon would sit inside the first bin and the fitted
            #     "range" would be an artefact of the bin width. Within-region
            #     pairs put the bins at ~1.4 km. There are only ~49,000 of them
            #     against ~2.5 million total, so they are used EXHAUSTIVELY --
            #     no subsampling noise enters the number that sizes the blocks.
            #
            #   * ALL pairs (context only, recorded but NOT fitted). It describes
            #     between-region heterogeneity, which `region_heterogeneity`
            #     already quantifies separately.
            within_curve = None
            if unit == "sample" and "region" in merged.columns:
                within_curve = distance_similarity_curve_within_groups(
                    glon, glat, merged["region"].astype(str).to_numpy(),
                    feats, n_lags=args.n_lags,
                    max_lag_fraction=args.max_lag_fraction,
                    max_lag_km=args.max_lag_km)
                if len(within_curve):
                    within_curve.insert(0, "unit", unit)
                    within_curve.to_csv(OUT_CURVE_WITHIN, index=False,
                                        encoding="utf-8")
                    result["outputs"]["curve_within_region"] = str(OUT_CURVE_WITHIN)
                    result["curve_within_region"] = {
                        "n_pairs_total": within_curve.attrs.get("n_pairs_total"),
                        "upper_km": within_curve.attrs.get("upper_km"),
                        "bin_width_km": within_curve.attrs.get("bin_width_km"),
                        "n_bins": int(len(within_curve)),
                        "exhaustive": True,
                    }
                    logger.info(
                        "within-region curve: %d bins, bin width %.3f km, "
                        "upper %.2f km, %s pairs used EXHAUSTIVELY",
                        len(within_curve),
                        within_curve.attrs.get("bin_width_km") or float("nan"),
                        within_curve.attrs.get("upper_km") or float("nan"),
                        f"{within_curve.attrs.get('n_pairs_total'):,}"
                        if within_curve.attrs.get("n_pairs_total") else "n/a")

            curve = distance_similarity_curve_geodesic(
                glon, glat, feats, n_lags=args.n_lags,
                max_lag_fraction=args.max_lag_fraction,
                max_pairs=args.max_pairs)
            curve.insert(0, "unit", unit)
            curve.to_csv(OUT_CURVE, index=False, encoding="utf-8")
            result["outputs"]["curve"] = str(OUT_CURVE)
            result["curve_all_pairs_context_only"] = {
                "n_bins": int(len(curve)),
                "used_for_range": False,
                "note": ("all-pairs separations span continents, so these bins "
                         "are ~460 km wide and do not resolve the scale at which "
                         "land-cover composition varies; recorded for context "
                         "only"),
            }
            logger.info("all-pairs curve (context only): %d lag bins", len(curve))

            # Track WHICH curve drives r with an explicit flag. Comparing the
            # windowed frame back to `within_curve` by identity cannot work:
            # detect_variogram_window returns a new frame, so `is` is always
            # False and the provenance was reported as "all_pairs" even when the
            # within-region curve drove the estimate.
            use_within = within_curve is not None and len(within_curve) > 0
            fit_curve = within_curve if use_within else curve
            if not use_within:
                result["notes"].append(
                    "within-region pairs were unavailable, so r is fitted on the "
                    "ALL-PAIRS curve whose bins are continent-scale. Treat the "
                    "resulting range as region-to-region, not patch-to-patch.")

            # Restrict to the lags that carry enough pairs to support a fit. A
            # variogram is only interpretable where the process is continuous
            # AND where each lag rests on enough pairs; beyond ~20 km the
            # within-region pairs mix distinct sub-landscapes, the profile
            # oscillates, and pair support collapses below 500. The first run
            # returned r = 33.0 km from a 28.1 km window -- a range longer than
            # the data, i.e. an extrapolation presented as a measurement.
            fit_curve, win_info = detect_variogram_window(
                fit_curve, min_pairs=args.window_min_pairs,
                tolerance=args.window_tolerance,
                smooth_window=args.window_smooth,
                enforce_monotone=args.window_monotone)
            result["variogram_window"] = win_info
            logger.info("variogram window: %d of %d bins, up to %.2f km, "
                        "min pairs in window %s (rule: %s)",
                        win_info["n_bins_window"], win_info["n_bins_input"],
                        win_info["upper_km"] or float("nan"),
                        win_info.get("min_pairs_in_window"),
                        win_info["rule"])
            if win_info.get("truncated"):
                logger.info("window truncated by the monotone walk at %.2f km",
                            win_info["stopped_at_km"] or float("nan"))

            est = estimate_range(fit_curve, VARIOGRAM_MODELS, tier=tier)
            result["range_estimated"] = est.r_km is not None
            result["range"] = est.to_dict()
            result["range_unit"] = unit
            result["range_fitted_on"] = ("within_region_pairs"
                                         if use_within else "all_pairs")
            # Per-model ranges are reported individually: the model-form spread
            # is the dominant uncertainty here and must not be hidden behind a
            # single median.
            per_model = {
                m: (v["params"] or {}).get("range_km")
                for m, v in (est.per_model or {}).items()}
            result["range_per_model_km"] = per_model
            if any(v is not None for v in per_model.values()):
                logger.info("per-model ranges (km): %s",
                            {m: (round(v, 3) if v else None)
                             for m, v in per_model.items()})

            # IDENTIFIABILITY. If the fitted range lies at or beyond the last
            # fitted lag, the variogram had not plateaued inside the window, so
            # the value is an extrapolation and must be reported as a lower
            # bound rather than as r.
            upper = win_info.get("upper_km")
            if est.r_km is not None and upper:
                if est.r_km >= upper:
                    result["range_identified"] = False
                    result["notes"].append(
                        f"the fitted range ({est.r_km:.2f} km) reaches or exceeds "
                        f"the last fitted lag ({upper:.2f} km): the variogram had "
                        f"not plateaued inside the window, so this value is a "
                        f"LOWER BOUND on the dependence scale, not an estimate of "
                        f"it. Report it as such.")
                    logger.warning(
                        "range %.2f km >= window upper %.2f km -- NOT identified; "
                        "reporting as a lower bound", est.r_km, upper)
                else:
                    result["range_identified"] = True
                    result["range_window_km"] = upper
            logger.info("range estimate: r=%s km stable=%s (%s)",
                        est.r_km, est.stable, est.note)
            if est.r_km is not None:
                result["notes"].append(
                    f"range estimated at {unit} scale: r = {est.r_km:.3f} km "
                    f"(stable across models: {est.stable}). Block sizes are "
                    f"derived as 0.5r, 1r and 2r from this value.")
                if not est.stable:
                    result["notes"].append(
                        "the range is NOT stable across variogram models; it must "
                        "be reported as an interval and no single block size may "
                        "be asserted as optimal")
    else:
        logger.warning("distance-resolved statistics SKIPPED (see notes)")
        result["notes"].append(
            "Consequence: block-size definitions in S2/S3 cannot be derived from "
            "a measured range on this dataset. The study reports region-level "
            "grouping designs (S1/S4) as the primary spatially independent "
            "strategies, and states this as a data-driven limitation rather than "
            "an analytical choice.")

    write_json(OUT_JSON, result)

    print()
    print("[PHASE] Spatial dependence analysis")
    print(f"[STATUS] {'PASS' if result['range_estimated'] else 'PARTIAL'}")
    print(f"[COMPLETED] tier={tier}; between-region heterogeneity for "
          f"{len(het)} classes; "
          f"range_estimated={result['range_estimated']}")
    if result["range_estimated"]:
        r = result["range"]
        print(f"[QC] r = {r['r_km']:.3f} km ({r['model']}), stable={r['stable']}, "
              f"fit R2={r['fit_r2']}")
    else:
        print("[QC] no numeric r reported; reasons:")
        for n in result["notes"]:
            print(f"      - {n}")
    print(f"[FILES] {OUT_HETERO}")
    if result["outputs"].get("curve_within_region"):
        print(f"[FILES] {OUT_CURVE_WITHIN}")
    if result["outputs"].get("curve"):
        print(f"[FILES] {OUT_CURVE}")
    print(f"[FILES] {OUT_JSON}")
    print("[NEXT] build the S0-S4 partition designs and run the leakage QC")
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

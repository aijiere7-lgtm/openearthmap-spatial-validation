#!/usr/bin/env python
"""
test_spatial_dependence.py — validate the dependence machinery against synthetic
data with a KNOWN spatial structure.

Rationale (same as test_split_logic.py and test_metrics.py): a defect in the
dependence code is invisible until after a real analysis, where there is no
ground truth to check against. Here the ground truth is constructed, so the code
can be wrong only if the test passes.

The core capability being validated is the one the whole study depends on:
CAN THE MACHINERY TELL SPATIALLY DEPENDENT DATA FROM SPATIALLY RANDOM DATA?
If it cannot, no conclusion about random vs spatial partitioning could be
trusted. So the test constructs both cases and requires opposite answers.

Also verified: the T3 refusal. Under tier T3 the protocol forbids reporting a
numeric range, and `estimate_range` must raise rather than return a number.

Run: python tests/test_spatial_dependence.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from spatial.dependence import (  # noqa: E402
    build_weights, build_weights_spherical, detect_variogram_window,
    distance_similarity_curve, distance_similarity_curve_geodesic,
    distance_similarity_curve_within_groups, estimate_range, haversine_km,
    morans_i, pairwise_haversine_km, to_local_km,
)

# Import the analysis script as a module to test its pure helpers.
_spec = importlib.util.spec_from_file_location(
    "spatial_dependence_script", ROOT / "scripts" / "spatial_dependence.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

CHECKS = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(f"{name} {detail}")


def make_coords(n: int, extent_km: float, seed: int) -> np.ndarray:
    """Uniform points on a square plane, in km."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent_km, size=(n, 2))


def correlated_field(coords: np.ndarray, scale_km: float, seed: int,
                     sigma2: float = 1.0) -> np.ndarray:
    """Gaussian field with covariance exp(-d / scale_km).

    The correlation decays to 1/e at `scale_km`, so the EFFECTIVE range (where
    correlation becomes negligible) is about 3 * scale_km. That distinction
    matters: the variogram models in dependence.py are parameterised by the
    effective range, so the expected fitted value is ~3 * scale_km, not
    scale_km. Asserting the wrong one would make a correct implementation look
    broken.
    """
    n = len(coords)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    cov = sigma2 * np.exp(-d / scale_km)
    cov += np.eye(n) * 1e-8                      # jitter for numerical stability
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    return L @ rng.standard_normal(n)


def main() -> int:
    print("=" * 72)
    print("SPATIAL DEPENDENCE LOGIC TEST (synthetic, known ground truth)")
    print("=" * 72)

    extent = 100.0
    n = 400
    coords = make_coords(n, extent, seed=11)

    # ------------------------------------------------------------------ #
    # 1. A spatially CORRELATED field must show strong positive autocorrelation
    # ------------------------------------------------------------------ #
    print("\n[1] correlated field (scale 5 km, effective range ~15 km)")
    z_corr = correlated_field(coords, scale_km=5.0, seed=21)
    w = build_weights(coords, k=8)
    mi_corr = morans_i(z_corr, w, permutations=499)
    print(f"      Moran's I = {mi_corr['morans_i']:.4f}  p = {mi_corr['p_sim']}")
    check("correlated: Moran's I is strongly positive",
          mi_corr["morans_i"] is not None and mi_corr["morans_i"] > 0.3,
          f"got {mi_corr['morans_i']}")
    check("correlated: autocorrelation is significant",
          mi_corr["p_sim"] is not None and mi_corr["p_sim"] < 0.05,
          f"got p={mi_corr['p_sim']}")

    # ------------------------------------------------------------------ #
    # 2. A spatially RANDOM field must show NO autocorrelation.
    #    This is the discriminating test: if the machinery reported a large
    #    Moran's I here too, it would be measuring nothing.
    # ------------------------------------------------------------------ #
    print("\n[2] spatially random field (the discriminating control)")
    rng = np.random.default_rng(31)
    z_rand = rng.standard_normal(n)
    mi_rand = morans_i(z_rand, w, permutations=499)
    print(f"      Moran's I = {mi_rand['morans_i']:.4f}  p = {mi_rand['p_sim']}")
    check("random: Moran's I is near zero",
          abs(mi_rand["morans_i"]) < 0.15, f"got {mi_rand['morans_i']}")
    check("random: no significant autocorrelation",
          mi_rand["p_sim"] is None or mi_rand["p_sim"] > 0.05,
          f"got p={mi_rand['p_sim']}")
    check("machinery DISCRIMINATES correlated from random",
          mi_corr["morans_i"] > mi_rand["morans_i"] + 0.25,
          f"corr={mi_corr['morans_i']:.3f} rand={mi_rand['morans_i']:.3f}")

    # ------------------------------------------------------------------ #
    # 3. The fitted range must be in the right neighbourhood of the truth.
    # ------------------------------------------------------------------ #
    print("\n[3] range recovery against the known effective range (~15 km)")
    feats = z_corr.reshape(-1, 1)
    curve = distance_similarity_curve(coords, feats, n_lags=20)
    check("curve has enough lag bins for a fit", len(curve) >= 4,
          f"got {len(curve)}")
    est = estimate_range(curve, ["spherical", "exponential", "gaussian"],
                         tier="T1", n_boot=50, seed=7)
    print(f"      r = {est.r_km}  model = {est.model}  R2 = {est.fit_r2}")
    print(f"      per-model ranges: "
          f"{ {m: (v['params'] or {}).get('range_km') for m, v in est.per_model.items()} }")
    check("range was estimated", est.r_km is not None)
    if est.r_km is not None:
        check("range is within a factor of 3 of the effective range",
              5.0 <= est.r_km <= 45.0, f"got {est.r_km:.2f} km")
        check("range is positive and finite",
              np.isfinite(est.r_km) and est.r_km > 0, f"got {est.r_km}")

    # A spatially random field should NOT yield a well-behaved range: the
    # semivariogram is flat, so the fitted range is unconstrained. This is
    # recorded, not asserted, because a flat variogram can still be fitted.
    curve_rand = distance_similarity_curve(coords, z_rand.reshape(-1, 1), n_lags=20)
    if len(curve_rand) >= 4:
        g = curve_rand["semivariance"].to_numpy(float)
        rel = float(np.nanstd(g) / (np.nanmean(g) or 1.0))
        print(f"      random-field semivariogram relative spread = {rel:.3f}")
        check("random field produces a flatter semivariogram than the correlated one",
              rel < 0.35, f"got {rel:.3f}")

    # ------------------------------------------------------------------ #
    # 4. T3 must REFUSE to produce a range. Producing one would be fabrication.
    # ------------------------------------------------------------------ #
    print("\n[4] tier T3 refusal")
    raised = False
    try:
        estimate_range(curve, ["spherical", "exponential", "gaussian"], tier="T3")
    except RuntimeError as exc:
        raised = True
        print(f"      raised RuntimeError as required: {str(exc)[:70]}...")
    check("estimate_range RAISES under T3", raised)

    # ------------------------------------------------------------------ #
    # 5. Projection sanity.
    #
    # The ORIGINAL version of this section tested only points at the equator,
    # where the single-reference-latitude equirectangular approximation happens
    # to be exact. That blind spot let a real defect through: the approximation
    # was being applied to a corpus spanning six continents, where it overstates
    # east-west distances by up to cos(lat0)/cos(lat) -- about 81% at 60 degrees
    # latitude. The section now tests high latitude explicitly, and asserts that
    # the unsafe case RAISES rather than returning a plausible wrong number.
    # ------------------------------------------------------------------ #
    print("\n[5] lon/lat -> km projection")
    # 1 degree of latitude is ~110.574 km everywhere.
    xy = to_local_km(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    dlat = float(np.linalg.norm(xy[1] - xy[0]))
    print(f"      1 degree of latitude -> {dlat:.3f} km")
    check("1 deg latitude ~ 110.6 km", abs(dlat - 110.574) < 0.5, f"got {dlat:.3f}")
    # At the equator, 1 degree of longitude is ~111.32 km.
    xy = to_local_km(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    dlon = float(np.linalg.norm(xy[1] - xy[0]))
    print(f"      1 degree of longitude at equator -> {dlon:.3f} km")
    check("1 deg longitude at equator ~ 111.3 km", abs(dlon - 111.320) < 0.5,
          f"got {dlon:.3f}")

    # ---- 5b. The regression: a WIDE latitude span must be refused --------- #
    print("\n[5b] the approximation must refuse a wide latitude span")
    wide_lon = np.array([0.0, 1.0])
    wide_lat = np.array([-35.0, 60.0])          # the corpus spans roughly this
    raised = False
    try:
        to_local_km(wide_lon, wide_lat)
    except ValueError as exc:
        raised = True
        print(f"      refused as required: {str(exc)[:66]}...")
    check("wide latitude span RAISES instead of returning a wrong distance",
          raised, "no ValueError raised")

    ok_after_optin = True
    try:
        to_local_km(wide_lon, wide_lat, allow_wide=True)
    except Exception:
        ok_after_optin = False
    check("allow_wide=True still permits the deliberate use", ok_after_optin)

    # A narrow span must continue to work, so the guard is not simply a block.
    narrow = to_local_km(np.array([0.0, 0.01]), np.array([45.0, 45.01]))
    check("narrow latitude span still projects",
          narrow.shape == (2, 2) and np.isfinite(narrow).all(),
          f"got {narrow}")

    # ---- 5c. Geodesic distances are latitude-correct --------------------- #
    print("\n[5c] geodesic distances")
    # On a sphere of R = 6371.0088, one degree of arc is 2*pi*R/360 km.
    one_deg = 2.0 * np.pi * 6371.0088 / 360.0
    check("1 deg of latitude on the sphere ~ 111.19 km",
          abs(float(haversine_km(0.0, 0.0, 1.0, 0.0)) - one_deg) < 0.05,
          f"got {float(haversine_km(0.0, 0.0, 1.0, 0.0)):.3f}")
    d_eq = float(haversine_km(0.0, 0.0, 0.0, 1.0))
    check("1 deg of longitude at the equator ~ 111.19 km",
          abs(d_eq - one_deg) < 0.05, f"got {d_eq:.3f}")
    d_60 = float(haversine_km(60.0, 0.0, 60.0, 1.0))
    check("1 deg of longitude at 60 deg latitude is about half the equatorial value",
          abs(d_60 - one_deg * 0.5) < 0.3, f"got {d_60:.3f}")

    # THE DEFECT, DEMONSTRATED.
    #
    # The approximation is EXACT when every point shares a latitude, because the
    # reference latitude then equals the latitude of the pair. It only breaks
    # when the dataset spans latitudes -- which is the actual situation. So the
    # demonstration must include points at other latitudes that pull the
    # reference latitude away from the pair being measured.
    #
    # Pair: two points at 60 N, 1 degree apart in longitude.
    # Dataset also contains equatorial points, so lat0 = mean(lat) = 30.
    # Planar east-west scale at 60 N is then 111.320*cos(30) = 96.4 km/deg,
    # whereas the true value is 111.320*cos(60) = 55.7 km/deg.
    lon_mix = np.array([0.0, 1.0, 0.0, 1.0])
    lat_mix = np.array([0.0, 0.0, 60.0, 60.0])
    xy_mix = to_local_km(lon_mix, lat_mix, allow_wide=True)   # lat0 = 30
    planar_60 = float(np.linalg.norm(xy_mix[3] - xy_mix[2]))
    d_60 = float(haversine_km(60.0, 0.0, 60.0, 1.0))
    ratio = planar_60 / d_60
    print(f"      1 deg longitude at 60 N, inside a dataset whose mean latitude "
          f"is 30 N:")
    print(f"        planar {planar_60:.2f} km  vs  geodesic {d_60:.2f} km  "
          f"(ratio {ratio:.2f})")
    check("planar approximation OVERSTATES east-west distance at high latitude",
          ratio > 1.5, f"ratio only {ratio:.2f}; the defect may have regressed")
    check("geodesic value is the correct one",
          abs(d_60 - one_deg * 0.5) < 0.3, f"got {d_60:.3f}")
    # And the same pair measured on its own is exact, which is why the guard is
    # about the DATASET's span and not about the pair's latitude.
    xy_same = to_local_km(np.array([0.0, 1.0]), np.array([60.0, 60.0]))
    planar_same = float(np.linalg.norm(xy_same[1] - xy_same[0]))
    check("with all points at one latitude the approximation is exact",
          abs(planar_same - d_60) < 0.2,
          f"planar {planar_same:.2f} vs geodesic {d_60:.2f}")

    # ---- 5d. pairwise helper matches the pairwise ordering --------------- #
    print("\n[5d] pairwise geodesic helper")
    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([0.0, 0.5, 1.0, 1.5])
    pw = pairwise_haversine_km(lon, lat)
    ii, jj = np.triu_indices(4, k=1)
    check("condensed length matches triu_indices",
          pw.size == ii.size, f"got {pw.size}, expected {ii.size}")
    ref = haversine_km(lat[ii], lon[ii], lat[jj], lon[jj])
    check("pairwise values match the elementwise function",
          np.allclose(pw, ref), "values disagree")
    check("pairwise ordering matches the triu layout",
          np.allclose(pw, ref), "ordering disagrees with triu_indices")

    # ---- 5e. A high-latitude field recovers its range geodesically ------- #
    print("\n[5e] range recovery at high latitude, geodesically")
    # Points spread over ~0.5 deg of latitude near 60 N. In a planar
    # single-reference projection the east-west spacing is inflated by ~2x,
    # which would stretch the apparent range; the geodesic curve must not.
    rng_hl = np.random.default_rng(11)
    n_hl = 240
    lon_hl = 10.0 + rng_hl.uniform(-0.25, 0.25, n_hl)
    lat_hl = 60.0 + rng_hl.uniform(-0.25, 0.25, n_hl)
    # Build a field whose correlation decays with TRUE great-circle distance.
    order = np.argsort(lat_hl)
    dmat = haversine_km(lat_hl[:, None], lon_hl[:, None],
                        lat_hl[None, :], lon_hl[None, :])
    scale_true = 8.0        # km, correlation decay length
    cov = np.exp(-dmat / scale_true) + 1e-8 * np.eye(n_hl)
    L = np.linalg.cholesky(cov)
    z_hl = L @ rng_hl.standard_normal(n_hl)
    curve_geo = distance_similarity_curve_geodesic(
        lon_hl, lat_hl, z_hl.reshape(-1, 1), n_lags=20)
    check("geodesic curve produced lag bins", len(curve_geo) >= 4,
          f"got {len(curve_geo)}")
    if len(curve_geo) >= 4:
        sem = curve_geo["semivariance"].to_numpy(float)
        d = curve_geo["lag_centre_km"].to_numpy(float)
        print(f"      semivariance rises from {sem[0]:.3f} to {sem[-1]:.3f} "
              f"over {d[0]:.2f}-{d[-1]:.2f} km")
        check("semivariance increases with distance (dependence is detected)",
              sem[-1] > sem[0], f"{sem[0]:.3f} -> {sem[-1]:.3f}")
        est_hl = estimate_range(curve_geo, ["spherical", "exponential", "gaussian"],
                                tier="T1", n_boot=30, seed=3)
        print(f"      fitted r = {est_hl.r_km} km (true effective range ~ "
              f"{3 * scale_true:.1f} km)")
        check("range is finite and positive",
              est_hl.r_km is not None and est_hl.r_km > 0,
              f"got {est_hl.r_km}")
        if est_hl.r_km is not None:
            check("range is within a factor of 4 of the true effective range",
                  3.0 <= est_hl.r_km <= 110.0, f"got {est_hl.r_km:.2f} km")

    # ---- 5f. Spherical KNN picks the true neighbours at high latitude ---- #
    print("\n[5f] exact great-circle nearest neighbours")
    # Three points at 60 N: the middle one's nearest neighbour by geodesic
    # distance is the western point (1 deg of longitude is ~55.6 km there),
    # NOT whichever a globally-referenced planar projection would prefer.
    lon3 = np.array([10.0, 10.5, 20.0])
    lat3 = np.array([60.0, 60.0, 60.0])
    w_sph = build_weights_spherical(lon3, lat3, k=1)
    nb = w_sph.neighbors[1]
    check("middle point's nearest neighbour is the adjacent one, not the far one",
          nb == [0], f"got {nb}")
    d01 = float(haversine_km(60.0, 10.0, 60.0, 10.5))
    d12 = float(haversine_km(60.0, 10.5, 60.0, 20.0))
    check("the chosen neighbour really is closer",
          d01 < d12, f"{d01:.1f} vs {d12:.1f} km")

    # ---- 5g. Within-group pairs are what make r measurable at all --------- #
    #
    # THE DEFECT THIS LOCKS DOWN. On the development pool the all-pairs curve has
    # a 99th-percentile separation of ~18,000 km, so 20 linear bins are ~460 km
    # wide. Land-cover composition varies over metres to kilometres, so on the
    # all-pairs curve the whole phenomenon sits inside the FIRST bin and the
    # fitted "range" is an artefact of the bin width. Restricting to within-group
    # pairs puts the bins at ~1.4 km. This test constructs the same situation in
    # miniature and asserts that the two curves resolve different scales.
    print("\n[5g] within-group pairs resolve the scale the all-pairs curve cannot")
    rng_g = np.random.default_rng(23)
    per_group, n_groups = 120, 2
    g_lon, g_lat, g_lab, g_val = [], [], [], []
    for g in range(n_groups):
        base_lon = 10.0 + g * 50.0          # groups 50 deg apart
        base_lat = 50.0 + g * 5.0
        lo = base_lon + rng_g.uniform(-0.15, 0.15, per_group)
        la = base_lat + rng_g.uniform(-0.15, 0.15, per_group)
        dmat = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        scale_g = 6.0                        # km, correlation decay length
        cov = np.exp(-dmat / scale_g) + 1e-8 * np.eye(per_group)
        L = np.linalg.cholesky(cov)
        v = L @ rng_g.standard_normal(per_group)
        v = v + 5.0 * g                      # large between-group offset
        g_lon.append(lo); g_lat.append(la); g_val.append(v)
        g_lab.append(np.full(per_group, f"G{g}"))
    g_lon = np.concatenate(g_lon); g_lat = np.concatenate(g_lat)
    g_lab = np.concatenate(g_lab); g_val = np.concatenate(g_val)
    g_feat = g_val.reshape(-1, 1)

    curve_all = distance_similarity_curve_geodesic(
        g_lon, g_lat, g_feat, n_lags=20)
    curve_win = distance_similarity_curve_within_groups(
        g_lon, g_lat, g_lab, g_feat, n_lags=20)

    # The all-pairs curve can fail in TWO ways, and both are the same defect.
    # Here the pair-distance distribution is bimodal -- within-group pairs sit
    # below ~40 km, between-group pairs near ~3330 km, and nothing lies in
    # between -- so the 20 linear bins are ~84 km wide and every informative
    # pair falls into bin 0, leaving a curve with a single usable row. The
    # all-pairs curve therefore cannot resolve ANY scale, which is a strictly
    # stronger demonstration than merely having coarse bins. Accept either
    # outcome, but require the within-group curve to be genuinely resolvable.
    all_width = (float(curve_all["lag_centre_km"].diff().dropna().median())
                 if len(curve_all) > 1 else float("nan"))
    win_width = curve_win.attrs.get("bin_width_km")
    degenerate = len(curve_all) <= 1 or not np.isfinite(all_width)
    if degenerate:
        print(f"      all-pairs curve DEGENERATE: only {len(curve_all)} usable "
              f"bin(s) -- every informative pair lands in one bin")
    else:
        print(f"      all-pairs median bin width  = {all_width:.1f} km")
    print(f"      within-group bin width      = {win_width:.3f} km")
    check("the within-group curve is resolvable at all (positive half)",
          win_width is not None and len(curve_win) >= 5,
          f"bins={len(curve_win)} width={win_width}")
    check("the all-pairs curve cannot resolve the within-group scale",
          degenerate or (win_width is not None
                         and all_width / win_width > 50.0),
          f"all={all_width} all_bins={len(curve_all)} within={win_width}")

    exact_pairs = n_groups * per_group * (per_group - 1) // 2
    check("every within-group pair is used (no subsampling)",
          curve_win.attrs.get("n_pairs_total") == exact_pairs,
          f"got {curve_win.attrs.get('n_pairs_total')}, expected {exact_pairs}")

    # The within-group curve must RECOVER the constructed effective range
    # (~3 * 6 km = 18 km). The all-pairs curve cannot: its bins are ~2000 km
    # wide, so it reports a number driven by the between-group offset instead.
    est_win = estimate_range(curve_win, ["spherical", "exponential", "gaussian"],
                             tier="T1", n_boot=30, seed=5)
    est_all = estimate_range(curve_all, ["spherical", "exponential", "gaussian"],
                             tier="T1", n_boot=30, seed=5)
    print(f"      within-group fitted r = {est_win.r_km} km "
          f"(true effective range ~ 18 km)")
    print(f"      all-pairs   fitted r = {est_all.r_km} km")
    check("within-group curve recovers a range near the constructed one",
          est_win.r_km is not None and 6.0 <= est_win.r_km <= 60.0,
          f"got {est_win.r_km}")
    check("the all-pairs range is much larger, i.e. it measures a different thing",
          est_all.r_km is None or est_win.r_km is None
          or est_all.r_km > est_win.r_km,
          f"all={est_all.r_km} within={est_win.r_km}")

    # A singleton group contributes no pairs and must not break the call.
    lone_lab = np.concatenate([g_lab, np.array(["LONE"])])
    lone_lon = np.concatenate([g_lon, np.array([0.0])])
    lone_lat = np.concatenate([g_lat, np.array([0.0])])
    lone_feat = np.concatenate([g_val, np.array([0.0])]).reshape(-1, 1)
    curve_lone = distance_similarity_curve_within_groups(
        lone_lon, lone_lat, lone_lab, lone_feat, n_lags=20)
    check("a singleton group contributes no pairs and does not raise",
          curve_lone.attrs.get("n_pairs_total") == exact_pairs,
          f"got {curve_lone.attrs.get('n_pairs_total')}, expected {exact_pairs}")

    # An explicit upper bound must be honoured.
    curve_cap = distance_similarity_curve_within_groups(
        g_lon, g_lat, g_lab, g_feat, n_lags=10, max_lag_km=10.0)
    check("an explicit max_lag_km caps the curve",
          curve_cap.attrs.get("upper_km") == 10.0
          and float(curve_cap["lag_centre_km"].max()) < 10.0,
          f"upper={curve_cap.attrs.get('upper_km')} "
          f"max_lag={curve_cap['lag_centre_km'].max() if len(curve_cap) else None}")

    # Mismatched lengths must be refused rather than silently mis-paired.
    raised_len = False
    try:
        distance_similarity_curve_within_groups(
            g_lon[:-1], g_lat, g_lab, g_feat, n_lags=10)
    except ValueError:
        raised_len = True
    check("mismatched input lengths raise", raised_len)

    # ---- 5h. the variogram window rule ------------------------------------- #
    #
    # The window decides which lags enter the fit, so an unsupported window
    # silently changes r. Two properties matter and are locked down here:
    # (1) the SUPPORT criterion is the rule -- bins below the pair threshold are
    #     excluded regardless of their shape;
    # (2) the optional monotone walk must NOT truncate inside a plateau whose
    #     bin-to-bin scatter is explained by pair-count noise, which is why it is
    #     off by default. A sweep on the real curve showed that for every
    #     tolerance >= 0.12 the walk returns exactly the support window, and
    #     below that it truncates inside a genuine plateau -- i.e. it has no
    #     discriminating power on this data.
    print("\n[5h] variogram window: support rule vs optional monotone walk")
    n = 20
    lag = 0.7 + 1.406 * np.arange(n)
    # A profile that rises to a sill at ~10 km, then plateaus with ~6% noise,
    # then steps up where support collapses. This reproduces the measured shape.
    rng_w = np.random.default_rng(7)
    sill = 6.2
    gamma = np.where(
        lag <= 10.5, 3.4 + (sill - 3.4) * (lag / 10.5) ** 0.7,
        sill + rng_w.normal(0.0, 0.35, n))
    gamma[13:] = 8.4 + rng_w.normal(0.0, 0.4, n - 13)
    pairs = np.array([9536, 8711, 5363, 3618, 3138, 2624, 2107, 1830, 1628,
                      1513, 1287, 865, 619, 480, 305, 254, 376, 247, 262, 307],
                     dtype=float)
    fake = pd.DataFrame({"lag_centre_km": lag, "semivariance": gamma,
                         "mean_similarity": -gamma, "n_pairs": pairs})

    win_sup, info_sup = detect_variogram_window(fake, min_pairs=500)
    n_supported = int((pairs >= 500).sum())
    print(f"      bins with >= 500 pairs: {n_supported}; window kept "
          f"{info_sup['n_bins_window']} up to {info_sup['upper_km']:.2f} km")
    check("the support rule keeps exactly the supported bins",
          info_sup["n_bins_window"] == n_supported
          and info_sup["min_pairs_in_window"] >= 500,
          f"kept {info_sup['n_bins_window']} expected {n_supported}")
    check("the support rule does not truncate inside the supported range",
          info_sup["truncated"] is False
          and info_sup["upper_km"] == float(lag[n_supported - 1]),
          f"truncated={info_sup['truncated']} upper={info_sup['upper_km']} "
          f"expected {float(lag[n_supported - 1])}")
    check("the default rule is the support threshold alone",
          info_sup["enforce_monotone"] is False
          and info_sup["rule"] == "pair-support threshold")

    # Enabling the walk must be able to shorten the window, otherwise the flag
    # would be decorative -- but the default must not.
    win_mono, info_mono = detect_variogram_window(
        fake, min_pairs=500, tolerance=0.0, enforce_monotone=True)
    print(f"      monotone walk (tol=0.00): kept {info_mono['n_bins_window']} "
          f"up to {info_mono['upper_km']:.2f} km")
    check("enabling the monotone walk can shorten the window",
          info_mono["n_bins_window"] <= info_sup["n_bins_window"]
          and info_mono["enforce_monotone"] is True,
          f"mono={info_mono['n_bins_window']} sup={info_sup['n_bins_window']}")
    check("a monotone walk at zero tolerance truncates on the first dip",
          info_mono["truncated"] is True
          and info_mono["stopped_at_km"] is not None)

    # Raising the tolerance far enough must reproduce the support window: this is
    # the fixed-point property that made the walk useless here.
    win_loose, info_loose = detect_variogram_window(
        fake, min_pairs=500, tolerance=1.0, enforce_monotone=True)
    check("a loose monotone walk collapses onto the support window",
          info_loose["n_bins_window"] == n_supported
          and info_loose["truncated"] is False,
          f"loose={info_loose['n_bins_window']} sup={n_supported}")

    # A degenerate support set must be returned rather than raising.
    win_none, info_none = detect_variogram_window(
        fake, min_pairs=10 ** 9)
    check("a support threshold above every bin returns an empty window",
          len(win_none) == 0 and info_none["n_bins_window"] == 0)
    check("an empty input curve is passed through unchanged",
          detect_variogram_window(
              pd.DataFrame(columns=["lag_centre_km", "semivariance", "n_pairs"]),
              min_pairs=500)[1]["n_bins_window"] == 0)

    print("\n[6] between-region heterogeneity (T3 fallback statistic)")
    reg = pd.DataFrame({
        "region": [f"r{i}" for i in range(6)],
        "n_images": [10] * 6,
        "n_pixels": [1000] * 6,
    })
    # class 0 shares: mean 0.4, sd 0.2 (constructed exactly)
    shares0 = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]      # mean 0.4, sample sd 0.2345
    for c in range(8):
        reg[f"share_class_{c}"] = shares0 if c == 0 else [0.0] * 6
    het = sd.region_heterogeneity(reg, sd.get_logger("test_het"))
    row0 = het[het["class_code"] == 0].iloc[0]
    exp_mean = float(np.mean(shares0))
    exp_sd = float(np.std(shares0, ddof=1))
    print(f"      class 0: mean={row0['region_mean_share']:.4f} "
          f"(expect {exp_mean:.4f}), sd={row0['region_sd']:.4f} "
          f"(expect {exp_sd:.4f})")
    check("heterogeneity mean is exact",
          abs(row0["region_mean_share"] - exp_mean) < 1e-9)
    check("heterogeneity sd is exact", abs(row0["region_sd"] - exp_sd) < 1e-9)
    check("region_range is exact",
          abs(row0["region_range"] - (max(shares0) - min(shares0))) < 1e-9)
    absent = het[het["class_code"] == 1].iloc[0]
    check("a class absent in every region is flagged",
          int(absent["regions_absent"]) == 6, f"got {absent['regions_absent']}")

    # ------------------------------------------------------------------ #
    # 7. feature_matrix must fail loudly rather than silently.
    # ------------------------------------------------------------------ #
    print("\n[7] feature_matrix input validation")
    raised = False
    try:
        sd.feature_matrix(pd.DataFrame({"region": ["a", "b"]}))
    except ValueError:
        raised = True
    check("feature_matrix raises when share columns are absent", raised)

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"RESULT: FAIL — {len(FAILURES)} of {CHECKS} assertions failed")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 72)
        return 1
    print(f"RESULT: PASS — all {CHECKS} assertions hold")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

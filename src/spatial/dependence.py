"""
dependence.py — spatial dependence diagnostics and estimation of the
dependence range r.

Implements research_protocol.md §7. All computation is confined to the
DEVELOPMENT POOL: external regions are excluded so that no information about
them enters any quantity used to configure a split.

Under coordinate tier T3 (no usable coordinates) this module must not be asked
for a numeric range. `estimate_range` raises in that case rather than returning
a number that is not grounded in measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common import NUM_CLASSES
from spatial.geometry import (  # noqa: F401  (re-exported for compatibility)
    DEFAULT_MAX_LAT_SPAN_DEG, EARTH_RADIUS_KM, chord_to_great_circle_km,
    equirectangular_km, haversine_km, latitude_span_deg, pairwise_haversine_km,
    to_unit_sphere,
)


# --------------------------------------------------------------------------- #
# Spatial weights
# --------------------------------------------------------------------------- #
def build_weights(coords_km: np.ndarray, k: int = 8, row_standardise: bool = True):
    """k-nearest-neighbour spatial weights via libpysal."""
    from libpysal.weights import KNN

    w = KNN.from_array(coords_km, k=min(k, max(1, len(coords_km) - 1)))
    if row_standardise:
        w.transform = "r"
    return w


def build_weights_spherical(lon: np.ndarray, lat: np.ndarray, k: int = 8,
                            row_standardise: bool = True):
    """k-nearest-neighbour spatial weights with EXACT great-circle neighbours.

    The neighbour search runs on 3-D unit-sphere coordinates, so the neighbour
    SET is correct at any latitude. Building the weights in a single-latitude
    equirectangular plane -- as this project previously did -- distorts the
    search space by cos(lat0)/cos(lat), which can reorder near neighbours at high
    latitude and therefore changes which units Moran's I treats as adjacent.
    """
    from libpysal.weights import W
    from scipy.spatial import cKDTree

    xy = to_unit_sphere(lon, lat)
    n = xy.shape[0]
    if n < 2:
        raise ValueError("at least 2 units are required to build weights")
    kk = min(int(k), n - 1)
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=kk + 1)  # includes the point itself
    neighbours: dict[int, list[int]] = {}
    for i in range(n):
        nb = [int(j) for j in np.atleast_1d(idx[i]) if int(j) != i][:kk]
        neighbours[i] = nb
    w = W(neighbours, silence_warnings=True)
    if row_standardise:
        w.transform = "r"
    return w


def to_local_km(lon: np.ndarray, lat: np.ndarray,
                max_lat_span_deg: float = DEFAULT_MAX_LAT_SPAN_DEG,
                allow_wide: bool = False) -> np.ndarray:
    """Project lon/lat to a LOCAL equirectangular km plane.

    This previously read "adequate at these scales" and was applied to a corpus
    spanning six continents, where a single reference latitude distorts east-west
    distances by up to a factor of cos(lat0)/cos(lat) -- about 81% at 60 degrees
    latitude. It now refuses spans beyond `max_lat_span_deg` unless `allow_wide`
    is passed, so the failure cannot recur silently.

    For anything wider than a degree or so, use `pairwise_haversine_km`, which is
    exact on the sphere and no more expensive.
    """
    return equirectangular_km(lon, lat, max_lat_span_deg=max_lat_span_deg,
                              allow_wide=allow_wide)


# --------------------------------------------------------------------------- #
# Moran's I
# --------------------------------------------------------------------------- #
def morans_i(values: np.ndarray, w, permutations: int = 999) -> dict[str, Any]:
    """Global Moran's I with a permutation-based pseudo p-value."""
    from esda.moran import Moran

    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"morans_i": None, "p_sim": None, "z_sim": None,
                "note": "too few finite values"}

    mi = Moran(v[ok], w, permutations=permutations)
    return {
        "morans_i": float(mi.I),
        "expected_i": float(mi.EI),
        "p_sim": float(mi.p_sim),
        "z_sim": float(mi.z_sim),
        "n": int(ok.sum()),
    }


def gearys_c(values: np.ndarray, w, permutations: int = 999) -> dict[str, Any]:
    from esda.geary import Geary

    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"gearys_c": None, "p_sim": None}
    gc = Geary(v[ok], w, permutations=permutations)
    return {"gearys_c": float(gc.C), "p_sim": float(gc.p_sim), "n": int(ok.sum())}


# --------------------------------------------------------------------------- #
# Distance-resolved similarity
# --------------------------------------------------------------------------- #
def _sample_pair_index(n: int, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the upper-triangle pair index, subsampled when necessary.

    Subsampling happens BEFORE distances are computed so that only the retained
    pairs are ever evaluated; this keeps the cost bounded for a global corpus.
    """
    ii, jj = np.triu_indices(n, k=1)
    if ii.size > max_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(ii.size, size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    return ii, jj


def distance_similarity_curve_within_groups(
    lon: np.ndarray,
    lat: np.ndarray,
    groups: np.ndarray,
    features: np.ndarray,
    n_lags: int = 20,
    max_lag_fraction: float = 0.5,
    max_lag_km: float | None = None,
    min_pairs_per_bin: int = 10,
    exhaustive_max_pairs: int = 5_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Distance-similarity curve using ONLY pairs drawn from the same group.

    WHY THIS IS THE RIGHT ESTIMATOR FOR r, AND WHY THE ALL-PAIRS CURVE IS NOT
    -----------------------------------------------------------------------
    The corpus spans six continents. Measured on the development pool:

        all pairs        p99 separation = 18,382 km  -> 20 linear bins over
                         0..0.5*p99 are 459 km wide
        within-region    p99 separation =     56 km  -> 20 linear bins are
                         1.4 km wide

    Land-cover composition is spatially dependent over metres to a few
    kilometres. On the all-pairs curve the entire phenomenon therefore falls
    inside the FIRST bin, and the fitted "range" becomes an artefact of the bin
    width -- a number that looks like a measurement and is not one. The
    all-pairs curve is additionally dominated by between-region contrast (only
    ~2% of pairs are within a region), so it describes continent-to-continent
    heterogeneity, which `region_heterogeneity` already measures separately.

    Restricting to within-group pairs also makes the computation EXACT rather
    than subsampled: the development pool yields ~49,000 within-region pairs
    against ~2.5 million total, so every within-region pair is used and there is
    no subsampling noise in the curve that determines the block sizes.

    `groups` is typically the region. Bin upper bound is `max_lag_km` when given,
    otherwise `max_lag_fraction * p99(within-group distances)` -- truncating the
    unstable tail is standard practice.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    groups = np.asarray(groups)
    if not (lon.size == lat.size == groups.size):
        raise ValueError("lon, lat and groups must have the same length")

    ii_parts: list[np.ndarray] = []
    jj_parts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 2:
            continue
        a, b = np.triu_indices(idx.size, k=1)
        gi, gj = idx[a], idx[b]
        if gi.size > exhaustive_max_pairs:
            pick = rng.choice(gi.size, size=exhaustive_max_pairs, replace=False)
            gi, gj = gi[pick], gj[pick]
        ii_parts.append(gi)
        jj_parts.append(gj)

    empty = pd.DataFrame(columns=["lag_centre_km", "mean_similarity",
                                  "n_pairs", "semivariance"])
    if not ii_parts:
        return empty
    ii = np.concatenate(ii_parts)
    jj = np.concatenate(jj_parts)
    d = haversine_km(lat[ii], lon[ii], lat[jj], lon[jj])
    if d.size == 0:
        return empty

    upper = (float(max_lag_km) if max_lag_km is not None
             else float(np.nanpercentile(d, 99)) * max_lag_fraction)
    if not np.isfinite(upper) or upper <= 0:
        return empty

    f = np.asarray(features, dtype=float)
    sd = f.std(axis=0)
    sd[sd == 0] = 1.0
    f = (f - f.mean(axis=0)) / sd
    diff = np.linalg.norm(f[ii] - f[jj], axis=1)

    edges = np.linspace(0.0, upper, n_lags + 1)
    rows = []
    for k in range(n_lags):
        m = (d >= edges[k]) & (d < edges[k + 1])
        if m.sum() < min_pairs_per_bin:
            continue
        rows.append({
            "lag_centre_km": float(0.5 * (edges[k] + edges[k + 1])),
            "mean_similarity": float(-diff[m].mean()),
            "semivariance": float(0.5 * (diff[m] ** 2).mean()),
            "n_pairs": int(m.sum()),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_pairs_total"] = int(d.size)
    out.attrs["upper_km"] = upper
    out.attrs["bin_width_km"] = float(edges[1] - edges[0]) if len(edges) > 1 else None
    return out


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average with edge padding; returns x unchanged if k <= 1."""
    x = np.asarray(x, dtype=float)
    if k <= 1 or x.size < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k, dtype=float) / k, mode="valid")


def detect_variogram_window(
    curve: pd.DataFrame,
    min_pairs: int = 500,
    tolerance: float = 0.05,
    smooth_window: int = 1,
    enforce_monotone: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict a distance-similarity curve to the lags that support a fit.

    WHY A WINDOW IS NEEDED, AND WHY IT MUST BE CHOSEN FROM THE DATA
    ---------------------------------------------------------------
    A variogram is only interpretable where the spatial process is continuous and
    where each lag is estimated from enough pairs to be meaningful. Both fail
    here, and both were measured rather than assumed.

    MEASURED PROFILE (development pool, within-region pairs, 1.406 km bins):

        lag km   0.70   6.33  10.54  13.35  17.57  20.38  27.41
        gamma    3.37   5.25   6.15   6.52   5.89   8.67   7.92
        n_pairs  9536   3138   1830   1513    619    305    307

    The semivariance rises steadily to a sill near 6.2 by ~10.5 km, hovers around
    6.0-6.7 through ~19 km, then STEPS UP to ~8.4 and oscillates. The step
    coincides exactly with pair support collapsing below ~500, and with the
    distance at which within-region pairs start crossing between sub-landscapes:
    41 of 62 development regions contain more than one spatial cluster at 2 km
    single-linkage, and 3.4% of tiles have no same-region neighbour within 2 km.
    Beyond that point the semivariance is no longer a function of distance alone.

    WHY THE THRESHOLD IS 500 PAIRS RATHER THAN A HAND-PICKED DISTANCE
    ---------------------------------------------------------------
    The empirical semivariance at a lag is an average over n pairs, so its
    relative standard error is of order sqrt(2/n). At n = 500 that is ~6%; at
    n = 250 it is ~9%, and at n = 100 it is ~14%. Requiring 500 pairs therefore
    bounds the per-bin noise at a stated level instead of relying on where the
    profile "looks flat". On this dataset the rule retains 13 of 20 bins and
    lands at 17.57 km, which keeps the plateau and excludes the inflated tail --
    but it lands there as a CONSEQUENCE of the support criterion, not by being
    tuned to this data.

    WHY THE MONOTONICITY WALK IS DISABLED BY DEFAULT
    ------------------------------------------------
    The walk was swept over tolerance in {0, 0.02, ..., 0.20} x smoothing in
    {off, 3-bin}, and its behaviour on this curve is degenerate:

        tolerance  0.000  0.050  0.069  0.070  0.120  >=0.120
        bins kept      8     10     10     12     13       13
        upper km   10.54  13.35  13.35  16.17  17.57    17.57

    For every tolerance >= 0.12 the walk returns EXACTLY the support-threshold
    window, so it carries no information of its own; and for tolerance < 0.12 it
    truncates INSIDE a genuine plateau, because a plateau whose bin-to-bin
    scatter is the ~6% predicted by sqrt(2/n) necessarily contains dips of that
    size. There is therefore no tolerance that separates the plateau from the
    inflated tail on this data, and a rule with no discriminating power must not
    be used to set the window. `enforce_monotone=True` remains available for
    profiles that genuinely rise and then decline within the supported region.

    Returns (windowed_curve, info).
    """
    info: dict[str, Any] = {
        "min_pairs": int(min_pairs), "tolerance": float(tolerance),
        "smooth_window": int(smooth_window),
        "enforce_monotone": bool(enforce_monotone),
        "n_bins_input": int(len(curve)), "n_bins_window": 0,
        "upper_km": None, "truncated": False, "stopped_at_km": None,
        "rule": ("pair-support threshold" if not enforce_monotone else
                 "pair-support threshold, then monotone rise"),
    }
    if curve is None or not len(curve):
        return curve, info
    c = curve[curve["n_pairs"] >= min_pairs].sort_values("lag_centre_km")
    if len(c) < 3:
        info["n_bins_window"] = int(len(c))
        info["upper_km"] = (float(c["lag_centre_km"].max()) if len(c) else None)
        info["min_pairs_in_window"] = (int(c["n_pairs"].min()) if len(c) else None)
        return c.reset_index(drop=True), info

    if not enforce_monotone:
        # The support criterion is the whole rule. Return it unchanged so that
        # the fitted range is a function of measured support and model form
        # only, never of a shape parameter.
        win = c.reset_index(drop=True)
        info["n_bins_window"] = int(len(win))
        info["upper_km"] = float(win["lag_centre_km"].max())
        info["min_pairs_in_window"] = int(win["n_pairs"].min())
        return win, info

    # Optional safeguard: walk forward while the curve keeps rising, using a
    # running maximum so that one noisy dip does not truncate a plateau.
    g_raw = c["semivariance"].to_numpy(float)
    g = _moving_average(g_raw, int(smooth_window))
    keep = [0]
    for i in range(1, len(g)):
        ref = float(np.max(g[keep]))
        if g[i] >= ref * (1.0 - tolerance):
            keep.append(i)
        else:
            break
    win = c.iloc[keep].reset_index(drop=True)
    info["n_bins_window"] = int(len(win))
    info["upper_km"] = float(win["lag_centre_km"].max())
    info["truncated"] = bool(len(win) < len(c))
    info["stopped_at_km"] = (float(c["lag_centre_km"].iloc[keep[-1] + 1])
                             if keep[-1] + 1 < len(c) else None)
    info["min_pairs_in_window"] = int(win["n_pairs"].min())
    return win, info


def _curve_from_pairs(ii: np.ndarray, jj: np.ndarray, d: np.ndarray,
                      features: np.ndarray, n_lags: int,
                      max_lag_fraction: float) -> pd.DataFrame:
    """Bin pair distances and compute the similarity / semivariance per bin."""
    f = np.asarray(features, dtype=float)
    sd = f.std(axis=0)
    sd[sd == 0] = 1.0
    f = (f - f.mean(axis=0)) / sd
    diff = np.linalg.norm(f[ii] - f[jj], axis=1)

    if d.size == 0:
        return pd.DataFrame(columns=["lag_centre_km", "mean_similarity",
                                     "n_pairs", "semivariance"])
    dmax = float(np.nanpercentile(d, 99))
    if not np.isfinite(dmax) or dmax <= 0:
        return pd.DataFrame(columns=["lag_centre_km", "mean_similarity",
                                     "n_pairs", "semivariance"])
    edges = np.linspace(0, dmax * max_lag_fraction, n_lags + 1)
    rows = []
    for k in range(n_lags):
        m = (d >= edges[k]) & (d < edges[k + 1])
        if m.sum() < 10:
            continue
        rows.append({
            "lag_centre_km": float(0.5 * (edges[k] + edges[k + 1])),
            "mean_similarity": float(-diff[m].mean()),
            "semivariance": float(0.5 * (diff[m] ** 2).mean()),
            "n_pairs": int(m.sum()),
        })
    return pd.DataFrame(rows)


def distance_similarity_curve(
    coords_km: np.ndarray,
    features: np.ndarray,
    n_lags: int = 20,
    max_lag_fraction: float = 0.5,
    max_pairs: int = 2_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Similarity between sample pairs as a function of separation distance.

    Similarity is the negative mean Euclidean distance in standardised feature
    space, so a monotone decrease in similarity with distance is the signature
    of spatial dependence. Pair enumeration is capped and subsampled so this
    stays tractable at n ~ 5000 (n^2/2 ~ 12.5M pairs).

    `coords_km` must be a PROJECTED coordinate array. For lon/lat in degrees use
    `distance_similarity_curve_geodesic`, which does not distort distances.
    """
    coords_km = np.asarray(coords_km, dtype=float)
    n = len(coords_km)
    if n < 10:
        return pd.DataFrame(columns=["lag_centre_km", "mean_similarity",
                                     "n_pairs", "semivariance"])
    ii, jj = _sample_pair_index(n, max_pairs, seed)
    d = np.linalg.norm(coords_km[ii] - coords_km[jj], axis=1)
    return _curve_from_pairs(ii, jj, d, features, n_lags, max_lag_fraction)


def distance_similarity_curve_geodesic(
    lon: np.ndarray,
    lat: np.ndarray,
    features: np.ndarray,
    n_lags: int = 20,
    max_lag_fraction: float = 0.5,
    max_pairs: int = 2_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """As `distance_similarity_curve`, but with exact great-circle distances.

    This is the correct entry point for lon/lat in degrees over a wide extent.
    Distances are computed only for the retained pairs, so the geodesic formula
    costs no more than the planar one at the subsampling rates used here.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = lon.size
    if n < 10:
        return pd.DataFrame(columns=["lag_centre_km", "mean_similarity",
                                     "n_pairs", "semivariance"])
    ii, jj = _sample_pair_index(n, max_pairs, seed)
    d = haversine_km(lat[ii], lon[ii], lat[jj], lon[jj])
    return _curve_from_pairs(ii, jj, d, features, n_lags, max_lag_fraction)


# --------------------------------------------------------------------------- #
# Range estimation
# --------------------------------------------------------------------------- #
@dataclass
class RangeEstimate:
    r_km: float | None
    r_ci_km: tuple[float, float] | None
    model: str | None
    fit_r2: float | None
    per_model: dict[str, Any]
    stable: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_km": self.r_km,
            "r_ci_km": list(self.r_ci_km) if self.r_ci_km else None,
            "model": self.model,
            "fit_r2": self.fit_r2,
            "per_model": self.per_model,
            "stable": self.stable,
            "note": self.note,
        }


def _fit_variogram(lag: np.ndarray, gamma: np.ndarray, model: str,
                   weights: np.ndarray | None = None):
    """Fit a parametric variogram model. Returns (params, r2, error).

    The fit is WEIGHTED by the number of pairs behind each bin when `weights` is
    supplied. An unweighted fit lets the sparse far bins dominate the residual:
    on the development pool the first bin holds ~16,700 pairs while bins beyond
    20 km hold a few hundred, so an unweighted fit is effectively fitting the
    noise in the tail. Weighting the empirical variogram by pair count is the
    standard treatment.
    """
    from scipy.optimize import curve_fit

    sill0 = float(np.nanmax(gamma))
    range0 = float(np.nanmax(lag) / 3.0)

    if model == "spherical":
        def f(h, nugget, sill, rng):
            h = np.asarray(h, float)
            out = np.where(h >= rng, sill,
                           nugget + (sill - nugget) *
                           (1.5 * h / rng - 0.5 * (h / rng) ** 3))
            return out
        p0 = [float(np.nanmin(gamma)), sill0, range0]
    elif model == "exponential":
        def f(h, nugget, sill, rng):
            return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * np.asarray(h, float) / rng))
        p0 = [float(np.nanmin(gamma)), sill0, range0]
    elif model == "gaussian":
        def f(h, nugget, sill, rng):
            return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * (np.asarray(h, float) / rng) ** 2))
        p0 = [float(np.nanmin(gamma)), sill0, range0]
    else:
        raise ValueError(f"unknown variogram model: {model}")

    sigma = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        # curve_fit minimises sum(((y - f) / sigma)^2), so sigma = 1/sqrt(w)
        # gives a weight proportional to w.
        sigma = 1.0 / np.sqrt(w)

    try:
        popt, _ = curve_fit(f, lag, gamma, p0=p0, sigma=sigma,
                            absolute_sigma=False, maxfev=20000,
                            bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]))
    except Exception as exc:                                   # noqa: BLE001
        return None, None, f"fit failed: {exc}"

    pred = f(lag, *popt)
    ss_res = float(np.sum((gamma - pred) ** 2))
    ss_tot = float(np.sum((gamma - gamma.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"nugget": float(popt[0]), "sill": float(popt[1]),
            "range_km": float(popt[2])}, r2, None


def estimate_range(
    curve: pd.DataFrame,
    models: list[str],
    tier: str,
    n_boot: int = 200,
    seed: int = 0,
) -> RangeEstimate:
    """Estimate the dependence range and assess whether it is stable.

    Stability matters more than the point value: if plausible models disagree
    substantially, the range is reported as an interval and the study must not
    assert a single optimal block size (protocol §7.2).
    """
    if tier == "T3":
        raise RuntimeError(
            "Coordinate tier T3: the dependence range cannot be estimated. The "
            "protocol forbids reporting a numeric r under this tier "
            "(research_protocol.md §5).")

    if curve.empty or len(curve) < 4:
        return RangeEstimate(None, None, None, None, {},
                             False, "insufficient lag points for a fit")

    lag = curve["lag_centre_km"].to_numpy(float)
    gamma = curve["semivariance"].to_numpy(float)
    ok = np.isfinite(lag) & np.isfinite(gamma)
    lag, gamma = lag[ok], gamma[ok]
    # Weight by pair count when the curve provides it: bins near the origin hold
    # thousands of pairs and the tail holds hundreds, so an unweighted fit would
    # be dominated by the noisiest part of the profile.
    w = (curve["n_pairs"].to_numpy(float)[ok]
         if "n_pairs" in curve.columns else None)

    per: dict[str, Any] = {}
    ranges: list[float] = []
    for m in models:
        params, r2, err = _fit_variogram(lag, gamma, m, weights=w)
        per[m] = {"params": params, "r2": r2, "error": err}
        if params and r2 is not None and r2 > 0.0:
            ranges.append(params["range_km"])

    if not ranges:
        return RangeEstimate(None, None, None, None, per, False,
                             "no model produced a usable fit")

    # IDENTIFIABILITY, PER MODEL. A fitted range is only a measurement if the
    # model actually attains its sill INSIDE the fitted window. If r >= the last
    # fitted lag, the model is still climbing at the window edge and r is an
    # extrapolation. This must be checked for EVERY model, not just the selected
    # one: on the development pool the exponential fit returns 23.4 km from a
    # window that ends at 17.6 km, and admitting that value into the stability
    # test would inflate the reported spread from 0.20 to 0.93 of the median and
    # wrongly declare the range unstable.
    upper = float(lag.max()) if len(lag) else None
    identified: dict[str, bool] = {}
    for m, v in per.items():
        r_m = (v["params"] or {}).get("range_km")
        if r_m is None or upper is None:
            identified[m] = False
        else:
            identified[m] = bool(r_m < upper)
        v["identified_within_window"] = identified[m]
    id_ranges = [per[m]["params"]["range_km"] for m in models
                 if identified.get(m) and per[m]["params"]
                 and per[m]["r2"] is not None and per[m]["r2"] > 0.0]
    if id_ranges:
        ranges = id_ranges

    r_med = float(np.median(ranges))
    best = max((m for m in models if per[m]["params"] and per[m]["r2"] is not None),
               key=lambda m: per[m]["r2"], default=None)

    spread = (max(ranges) - min(ranges)) / r_med if r_med > 0 else np.inf
    stable = spread <= 0.5 and len(ranges) >= 2

    # Bootstrap the chosen model's range by resampling lag points.
    ci = None
    if best is not None and len(lag) >= 6:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(lag), len(lag))
            if len(np.unique(lag[idx])) < 4:
                continue
            p, r2b, e = _fit_variogram(
                lag[idx], gamma[idx], best,
                weights=(w[idx] if w is not None else None))
            if p and r2b is not None and r2b > 0:
                boots.append(p["range_km"])
        if len(boots) >= 20:
            ci = (float(np.percentile(boots, 2.5)),
                  float(np.percentile(boots, 97.5)))

    excluded = sorted(m for m in models if not identified.get(m, False)
                      and per[m]["params"] is not None)
    note = ("range is stable across the tested variogram models"
            if stable else
            f"range is NOT stable across models (spread {spread:.2f} of the "
            f"median); report as an interval and do not assert a single optimal "
            f"block size")
    if excluded:
        note += (f"; {len(excluded)} of {len(models)} models were EXCLUDED from "
                 f"the interval because their fitted range reaches or exceeds "
                 f"the last fitted lag ({upper:.2f} km), i.e. they had not "
                 f"attained their sill inside the window and are extrapolations: "
                 f"{excluded}")

    return RangeEstimate(
        r_km=r_med,
        r_ci_km=ci,
        model=best,
        fit_r2=float(per[best]["r2"]) if best else None,
        per_model=per,
        stable=bool(stable),
        note=note,
    )

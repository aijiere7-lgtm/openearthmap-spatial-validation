"""
geometry.py — canonical geodesic and projection helpers.

WHY THIS MODULE EXISTS
----------------------
The project contained TWO divergent distance implementations:

  * `src/splits/strategies.py` defined a correct `haversine_km`.
  * `src/spatial/dependence.py` projected lon/lat with a single-reference-latitude
    equirectangular formula, `x = lon * 111.320 * cos(mean(lat))`, and described
    it as "adequate at these scales".

That description is FALSE for this corpus. OpenEarthMap spans six continents,
from roughly -35 to +60 degrees latitude. The east-west scale factor of an
equirectangular projection referenced to a single latitude is `cos(lat0)`, while
the true factor at latitude `lat` is `cos(lat)`, so distances are distorted by

        error factor = cos(lat0) / cos(lat)

At lat0 = 25 degrees and a pair at 60 degrees latitude the factor is 0.906/0.5 =
1.81, i.e. east-west distances are overestimated by about 81%. The distance axis
of the distance-similarity curve feeds `estimate_range`, so that distortion would
have propagated into the dependence range r -- the central derived quantity of
RQ2 -- while looking entirely plausible.

Correct geodesic distance is not more expensive, so there is no reason to accept
the approximation. One implementation now lives here and both modules import it.

CONVENTIONS
-----------
  * `haversine_km` takes arguments in (lat, lon) order, matching the pre-existing
    `strategies.haversine_km` signature. Reversing this order is a silent
    correctness bug, so the order is documented and asserted in tests.
  * "xy" arrays are (lon, lat) in degrees, matching the inventory and the region
    metadata tables.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# The equirectangular approximation is only offered for genuinely local extents.
# At latitude lat0 the relative error over a half-span of d degrees is
# approximately tan(lat0) * d * pi/180; 1 degree at lat0 = 45 gives ~0.9%.
DEFAULT_MAX_LAT_SPAN_DEG = 1.0


# --------------------------------------------------------------------------- #
# Great-circle distance
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorised over numpy arrays.

    Argument order is (lat, lon) -- see the module docstring.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_haversine_km(lon, lat) -> np.ndarray:
    """Condensed pairwise great-circle distances, in `np.triu_indices(n, 1)` order.

    The ordering matches `np.linalg.norm(coords[ii] - coords[jj], axis=1)` for
    `ii, jj = np.triu_indices(n, k=1)`, so it can be substituted directly for the
    planar distance vector without reordering anything.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape != lat.shape:
        raise ValueError(f"lon and lat must have the same shape; "
                         f"got {lon.shape} and {lat.shape}")
    n = lon.size
    if n < 2:
        return np.zeros(0, dtype=float)

    ii, jj = np.triu_indices(n, k=1)
    # Chunked so that a global corpus at n ~ 3000 (4.5M pairs, 3 arrays) does not
    # allocate more than a few tens of MB at once.
    out = np.empty(ii.size, dtype=float)
    step = 500_000
    for start in range(0, ii.size, step):
        stop = min(start + step, ii.size)
        out[start:stop] = haversine_km(
            lat[ii[start:stop]], lon[ii[start:stop]],
            lat[jj[start:stop]], lon[jj[start:stop]],
        )
    return out


# --------------------------------------------------------------------------- #
# Exact nearest-neighbour support
# --------------------------------------------------------------------------- #
def to_unit_sphere(lon, lat) -> np.ndarray:
    """Map lon/lat degrees to 3-D unit-sphere Cartesian coordinates.

    Used for nearest-neighbour search. On the unit sphere the straight-line chord
    length between two points is `2*sin(theta/2)`, which is strictly monotone in
    the great-circle angle `theta`. A KD-tree built on these coordinates therefore
    returns the TRUE great-circle nearest neighbour, with no projection error --
    unlike a KD-tree built in a single-latitude equirectangular plane, which
    distorts the search space by the same `cos(lat0)/cos(lat)` factor.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)
    return np.column_stack([
        np.cos(phi) * np.cos(lam),
        np.cos(phi) * np.sin(lam),
        np.sin(phi),
    ])


def chord_to_great_circle_km(chord) -> np.ndarray:
    """Convert a unit-sphere chord length to a great-circle distance in km."""
    c = np.clip(np.asarray(chord, dtype=float), 0.0, 2.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(c / 2.0)


# --------------------------------------------------------------------------- #
# Local planar projection (restricted)
# --------------------------------------------------------------------------- #
def latitude_span_deg(lat) -> float:
    lat = np.asarray(lat, dtype=float)
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return 0.0
    return float(lat.max() - lat.min())


def equirectangular_km(lon, lat, lat0: float | None = None,
                       max_lat_span_deg: float = DEFAULT_MAX_LAT_SPAN_DEG,
                       allow_wide: bool = False) -> np.ndarray:
    """Project lon/lat to a local equirectangular km plane.

    VALID ONLY for a small latitude span. The guard exists because this function
    was previously applied to a corpus spanning six continents, where it
    overestimated east-west distances by up to ~81%. Pass `allow_wide=True` only
    when the distortion has been reasoned about explicitly and the result is not
    used as a distance axis.

    Reference latitude defaults to the mean, which minimises the error over the
    span. The relative error at the edge of the span is approximately
    `tan(lat0) * (span/2) * pi/180`.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    span = latitude_span_deg(lat)
    if span > max_lat_span_deg and not allow_wide:
        raise ValueError(
            f"equirectangular projection requested over a latitude span of "
            f"{span:.2f} deg, which exceeds the {max_lat_span_deg:.2f} deg "
            f"validity limit. East-west distances would be distorted by up to "
            f"a factor of cos(lat0)/cos(lat). Use haversine_km / "
            f"pairwise_haversine_km instead, or pass allow_wide=True if the "
            f"distortion is acceptable for this use."
        )
    lat0 = float(np.nanmean(lat)) if lat0 is None else float(lat0)
    kx = 111.320 * np.cos(np.radians(lat0))
    ky = 110.574
    return np.column_stack([lon * kx, lat * ky])


def equirectangular_km_of_pair(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Planar distance using the MIDPOINT latitude of each pair.

    Accurate to well under 1% for pairs up to a few hundred km apart at any
    latitude, because the reference latitude is local to the pair rather than
    global. Provided for callers that need a planar metric on a genuinely local
    problem; `pairwise_haversine_km` remains the general answer.
    """
    lat_mid = 0.5 * (np.asarray(lat1, float) + np.asarray(lat2, float))
    kx = 111.320 * np.cos(np.radians(lat_mid))
    ky = 110.574
    dx = (np.asarray(lon2, float) - np.asarray(lon1, float)) * kx
    dy = (np.asarray(lat2, float) - np.asarray(lat1, float)) * ky
    return np.sqrt(dx * dx + dy * dy)

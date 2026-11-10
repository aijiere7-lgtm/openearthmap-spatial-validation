"""
strategies.py — construction of validation partitions (S0-S4, EXT) and
measurement of spatial independence.

Governing rules (research_protocol.md §6, §8, §9):
  * Spatial independence is MEASURED, never inferred from a strategy's name.
    Every partition carries a diagnostics record, and that record is the
    evidence for any independence claim.
  * The external test is selected by a deterministic, seeded, PERFORMANCE-BLIND
    procedure and then frozen. Nothing in this module consults model accuracy.
  * Matched-sample-size partitions are mandatory wherever a spatial design
    reduces the training count, so that a change in the estimate cannot be
    attributed to sample count.
  * Nothing is silently dropped. Samples excluded by a buffer are recorded with
    their exclusion reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from common import NUM_CLASSES


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
# Canonical implementations live in `spatial.geometry`. They are imported rather
# than redefined here because the project previously carried TWO divergent
# distance implementations: this module's correct `haversine_km`, and a
# single-reference-latitude equirectangular projection in `spatial.dependence`
# that distorted east-west distances by up to ~81% at high latitude. One
# implementation removes the possibility of them drifting apart again.
from spatial.geometry import (  # noqa: E402
    EARTH_RADIUS_KM, chord_to_great_circle_km, haversine_km, to_unit_sphere,
)


def nearest_distance_km(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    """For each row of src_xy, the great-circle distance (km) to the nearest row of dst_xy.

    Coordinates are (lon, lat) in degrees.

    The KD-tree is built on 3-D unit-sphere Cartesian coordinates, NOT in a
    planar projection. On the unit sphere the chord length is `2*sin(theta/2)`,
    which is strictly monotone in the great-circle angle, so the returned
    neighbour is the TRUE nearest neighbour and the distance is exact.

    The previous implementation built the tree in a single-latitude
    equirectangular plane. That distorts the search space by `cos(lat0)/cos(lat)`
    -- about 81% in the east-west direction at 60 degrees latitude -- so for a
    corpus spanning six continents both the neighbour identity and the distance
    could be wrong, silently. These distances set the buffer widths and the
    external-region selection, so an error here propagates into the partitions
    themselves.
    """
    from scipy.spatial import cKDTree

    src_xy = np.asarray(src_xy, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)
    if len(src_xy) == 0 or len(dst_xy) == 0:
        return np.full(len(src_xy), np.nan)

    s = to_unit_sphere(src_xy[:, 0], src_xy[:, 1])
    d = to_unit_sphere(dst_xy[:, 0], dst_xy[:, 1])
    tree = cKDTree(d)
    chord, _ = tree.query(s, k=1)
    return chord_to_great_circle_km(chord)


# --------------------------------------------------------------------------- #
# Partition containers
# --------------------------------------------------------------------------- #
@dataclass
class Partition:
    """One validation design applied to one set of samples.

    `assignment` is a DataFrame with at least:
        sample_id, split, fold, strategy, block_id, distance_to_training
    """
    strategy: str
    assignment: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def train_ids(self) -> set[str]:
        return set(self.assignment.loc[self.assignment["split"] == "train", "sample_id"])

    @property
    def val_ids(self) -> set[str]:
        return set(self.assignment.loc[self.assignment["split"] == "val", "sample_id"])

    def fold(self, k: int) -> tuple[set[str], set[str]]:
        a = self.assignment
        tr = set(a.loc[(a["split"] == "train") & (a["fold"] != k), "sample_id"])
        va = set(a.loc[(a["split"] == "val") & (a["fold"] == k), "sample_id"])
        return tr, va


# --------------------------------------------------------------------------- #
# S0 — random class-stratified split
# --------------------------------------------------------------------------- #
def random_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    n_train_target: int | None = None,
    stratify_col: str = "dominant_class",
) -> pd.DataFrame:
    """Class-stratified random K-fold assignment.

    Stratification is on the patch's dominant class, so every fold sees a
    comparable class mix. When `n_train_target` is given, the training set is
    subsampled to exactly that size — this is the matched-sample-size control
    required by protocol §8.2, and it is the only difference from the
    unmatched random split.
    """
    rng = np.random.default_rng(seed)
    out = df[["sample_id"]].copy()
    out["fold"] = -1
    out["split"] = "unused"

    strata = df[stratify_col].to_numpy() if stratify_col in df.columns \
        else np.zeros(len(df), dtype=int)

    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        perm = rng.permutation(idx)
        # round-robin so folds stay balanced even for small strata
        for j, i in enumerate(perm):
            out.iloc[i, out.columns.get_loc("fold")] = j % n_folds

    out["split"] = "val"  # every sample is validation in exactly one fold

    if n_train_target is not None:
        # Keep a fixed validation set (fold 0) and cap the training pool.
        val_mask = out["fold"] == 0
        n_val = int(val_mask.sum())
        pool = np.where(~val_mask)[0]
        keep = rng.permutation(pool)[: max(0, n_train_target)]
        allowed = np.zeros(len(out), dtype=bool)
        allowed[keep] = True
        allowed |= val_mask.to_numpy()
        out.loc[~allowed, "split"] = "excluded_matched_sample_size"
    return out


# --------------------------------------------------------------------------- #
# S1 — group / scene split
# --------------------------------------------------------------------------- #
def group_folds(df: pd.DataFrame, group_col: str, n_folds: int,
                seed: int) -> pd.DataFrame:
    """Grouped K-fold: no group may straddle a fold boundary."""
    rng = np.random.default_rng(seed)
    out = df[["sample_id"]].copy()
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)
    perm = rng.permutation(len(uniq))
    gfold = {uniq[p]: (j % n_folds) for j, p in enumerate(perm)}
    out["fold"] = [gfold[g] for g in groups]
    out["split"] = "val"
    return out


# --------------------------------------------------------------------------- #
# S2 — spatial block cross-validation
# --------------------------------------------------------------------------- #
def spatial_block_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    block_size_km: float | None = None,
    n_blocks: int | None = None,
    coord_cols: tuple[str, str] = ("lon", "lat"),
) -> pd.DataFrame:
    """Blocks defined on a regular grid over the study extent.

    A regular grid is used rather than k-means so that the block geometry is
    independent of sample density — otherwise dense areas would produce small
    blocks and the effective isolation distance would vary unintentionally.

    Either `block_size_km` or `n_blocks` must be given. `block_size_km` is the
    parameter that matters for the isolation-scale analysis.
    """
    lon = df[coord_cols[0]].to_numpy(dtype=float)
    lat = df[coord_cols[1]].to_numpy(dtype=float)
    if np.isnan(lon).any() or np.isnan(lat).any():
        raise ValueError("spatial_block_folds requires complete coordinates")

    lat0 = float(np.mean(lat))
    kx = 111.320 * np.cos(np.radians(lat0))
    ky = 110.574

    x = (lon - lon.min()) * kx
    y = (lat - lat.min()) * ky

    if block_size_km is None:
        if n_blocks is None:
            raise ValueError("provide block_size_km or n_blocks")
        extent = max(x.max(), y.max())
        block_size_km = max(extent / np.sqrt(n_blocks), 1e-6)

    bx = np.floor(x / block_size_km).astype(int)
    by = np.floor(y / block_size_km).astype(int)
    block_id = [f"b{bx[i]}_{by[i]}" for i in range(len(df))]

    out = df[["sample_id"]].copy()
    out["block_id"] = block_id

    uniq = np.unique(block_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    bfold = {uniq[p]: (j % n_folds) for j, p in enumerate(perm)}
    out["fold"] = [bfold[b] for b in block_id]
    out["split"] = "val"
    return out


def apply_buffer(
    assign: pd.DataFrame,
    coords: pd.DataFrame,
    buffer_km: float,
    val_fold: int,
) -> pd.DataFrame:
    """Exclude from training any sample within `buffer_km` of a validation sample.

    The exclusion is recorded with its reason rather than deleted, so the size
    of the buffer's effect is measurable and auditable (protocol §8).
    """
    out = assign.copy()
    val_mask = out["fold"] == val_fold
    val_xy = coords.loc[val_mask.to_numpy(), ["lon", "lat"]].to_numpy(dtype=float)
    all_xy = coords[["lon", "lat"]].to_numpy(dtype=float)

    d = nearest_distance_km(all_xy, val_xy)
    out["distance_to_val_km"] = d
    train_mask = (~val_mask).to_numpy()
    too_close = train_mask & (d < buffer_km)
    out.loc[too_close, "split"] = "excluded_buffer"
    return out


# --------------------------------------------------------------------------- #
# S4 — leave-one-region-out
# --------------------------------------------------------------------------- #
def leave_one_region_out(df: pd.DataFrame, region_col: str = "region",
                         folds_per_region: int = 1) -> pd.DataFrame:
    """Each region is held out in turn. `fold` indexes the held-out region."""
    out = df[["sample_id"]].copy()
    regions = df[region_col].astype(str).to_numpy()
    uniq = sorted(set(regions))
    rmap = {r: i for i, r in enumerate(uniq)}
    out["fold"] = [rmap[r] for r in regions]
    out["split"] = "val"
    out["region_held_out"] = regions
    return out


# --------------------------------------------------------------------------- #
# EXT — external geographic test selection (performance-blind)
# --------------------------------------------------------------------------- #
def _class_pixel_pivot(class_share: pd.DataFrame,
                       num_classes: int = NUM_CLASSES) -> pd.DataFrame:
    """Normalise a region x class pixel table to a (region x 0..n-1) frame.

    TWO LAYOUTS ARE ACCEPTED, because the producer and the consumer had drifted
    apart: `extract_class_composition.py` writes a WIDE table (`region`,
    `px_class_0` ... `px_class_7`), while this module's docstring described a LONG
    table (`region`, `class_id`, `pixel_count`). Nothing failed until the freeze
    was actually attempted, at which point `pivot_table` raised
    `KeyError: 'pixel_count'`. Accepting both here removes the mismatch as a
    category rather than patching one side of it.
    """
    lowered = {str(c).lower(): c for c in class_share.columns}
    if {"class_id", "pixel_count"} <= set(lowered):
        cs = class_share.rename(columns={v: k for k, v in lowered.items()})
        piv = cs.pivot_table(index="region", columns="class_id",
                             values="pixel_count", aggfunc="sum")
    else:
        px_cols = [f"px_class_{k}" for k in range(num_classes)]
        missing = [c for c in px_cols if c not in class_share.columns]
        if missing:
            raise KeyError(
                f"class-share table is neither long (region/class_id/pixel_count) "
                f"nor wide ({', '.join(px_cols[:2])}, ...); missing {missing}. "
                f"Columns present: {list(class_share.columns)}")
        piv = class_share.set_index("region")[px_cols].copy()
        piv.columns = list(range(num_classes))
    return piv.reindex(columns=range(num_classes)).fillna(0)


def select_external_regions(
    region_table: pd.DataFrame,
    class_share: pd.DataFrame,
    seed: int,
    target_n: tuple[int, int] = (12, 15),
    stratify_col: str = "continent",
    min_class_px: int = 1000,
    maximise_min_distance: bool = True,
    acceptable_coord_sources: Iterable[str] | None = None,
    min_tiles: int | None = None,
) -> dict[str, Any]:
    """Deterministically choose whole regions for the external geographic test.

    `region_table` needs columns: region, country, [stratify_col], lon, lat and
    optionally coord_source. `class_share` needs: region, class_id, pixel_count.

    Selection uses ONLY geographic dispersion, continent and class composition.
    No model, no accuracy, no preliminary result enters this function — that is
    the point of the freeze.

    COORDINATE HANDLING IS EXPLICIT, NOT SILENT. A region without usable
    coordinates cannot participate in a distance-based selection, so it is
    excluded — but the exclusion is counted, attributed to a reason, and
    returned in `coordinate_report`. The earlier implementation simply called
    `dropna(subset=["lon", "lat"])` and, when nothing survived, reported
    "no eligible regions after class-presence filter", which named the wrong
    cause and would have hidden a partial exclusion behind a plausible one.

    `acceptable_coord_sources` lets the caller apply the appropriate standard.
    Region-level dispersion tolerates gazetteer centroids, because the distances
    are inter-region (10^3-10^4 km) and the positional error is 10^1-10^2 km.
    Estimating the dependence range r does NOT, and that is enforced separately
    in `spatial_dependence.py`. See `common.COORD_SOURCES_OK_FOR_*`.
    """
    rng = np.random.default_rng(seed)

    # --- hard filter: every class must be present above the minimum count
    piv = _class_pixel_pivot(class_share)
    eligible = set(piv.index[piv.min(axis=1) >= min_class_px])

    # --- optional guard: a region must carry enough tiles to participate ----- #
    # A region with three tiles satisfies the class-presence filter (3 tiles are
    # ~3M pixels) but cannot serve as an external test set, and cannot serve as a
    # validation fold either -- under leave-one-region-out it would be a
    # three-tile fold whose estimate is pure noise. The threshold therefore has
    # ONE meaning: how many tiles a region needs before it takes part in the
    # study at all. It is applied to BOTH pools below, not just to the selection
    # pool, so that the two counts cannot be read as using different rules.
    #
    # It is currently dormant -- no class-filter passer has fewer than 8 tiles --
    # which is exactly why it belongs in the code rather than in a comment:
    # lowering `min_class_presence_px` would activate it.
    thin_all: set[str] = set()
    if min_tiles is not None and "n_images" in class_share.columns:
        per_region_tiles = class_share.groupby("region")["n_images"].max()
        thin_all = set(per_region_tiles.index[per_region_tiles < int(min_tiles)])
    thin_regions = sorted(eligible & thin_all)
    eligible -= thin_all

    n_in_table = int(len(region_table))
    cand = region_table[region_table["region"].isin(eligible)].copy()
    n_after_class = int(len(cand))

    # --- coordinate availability, counted and attributed ------------------- #
    # The counts form an arithmetic chain that must balance:
    #   n_after_class = n_no_coordinates + n_unacceptable_source + n_usable
    # A report whose counts disagree with its own exclusion list is worse than
    # no report, so the invariant is asserted rather than assumed.
    has_xy = (cand["lon"].notna() & cand["lat"].notna()) \
        if {"lon", "lat"} <= set(cand.columns) else pd.Series(False, index=cand.index)
    by_source: dict[str, int] = {}
    if "coord_source" in cand.columns:
        by_source = (cand["coord_source"].astype(str).value_counts().to_dict())
    n_non_null = int(has_xy.sum())

    source_ok = pd.Series(True, index=cand.index)
    if acceptable_coord_sources is not None and "coord_source" in cand.columns:
        source_ok = cand["coord_source"].astype(str).isin(set(acceptable_coord_sources))

    # Two distinct reasons a region cannot be used, reported separately: the
    # coordinate is absent, versus the coordinate exists but its provenance does
    # not meet the standard being applied. They call for different remedies.
    unusable = ~(has_xy & source_ok)
    n_unacceptable_source = int((has_xy & ~source_ok).sum())
    n_no_coordinates = int((~has_xy).sum())
    n_usable = int((has_xy & source_ok).sum())

    coord_report = {
        "n_regions_in_table": n_in_table,
        "n_after_class_presence_filter": n_after_class,
        "n_with_non_null_coordinates": n_non_null,
        "n_rejected_for_coord_source": n_unacceptable_source,
        "n_excluded_no_coordinates": n_no_coordinates,
        "n_usable_for_selection": n_usable,
        "coord_source_counts": by_source,
        "acceptable_coord_sources": (sorted(acceptable_coord_sources)
                                     if acceptable_coord_sources is not None
                                     else None),
        "counts_balance": (n_no_coordinates + n_unacceptable_source + n_usable
                           == n_after_class),
        "min_tiles_applied": min_tiles,
        "n_excluded_too_few_tiles": len(thin_all),
        "excluded_thin_regions": sorted(thin_all),
        "n_thin_regions_that_were_in_selection_pool": len(thin_regions),
    }
    if not coord_report["counts_balance"]:
        raise AssertionError(
            f"coordinate accounting does not balance: "
            f"{n_no_coordinates} absent + {n_unacceptable_source} unacceptable "
            f"source + {n_usable} usable != {n_after_class} after the class "
            f"filter. The report would be self-contradictory.")

    if n_after_class == 0:
        return {"error": (f"no region survives the class-presence filter "
                          f"(every class must reach {min_class_px} px); "
                          f"{n_in_table} regions were supplied"),
                "coordinate_report": coord_report}

    dropped = cand[unusable]
    if len(dropped):
        # Named explicitly, because a partial exclusion changes which regions
        # were even candidates and therefore what the selection means.
        coord_report["excluded_regions"] = sorted(dropped["region"].astype(str))
        parts = []
        if n_no_coordinates:
            parts.append(f"{n_no_coordinates} with no coordinate at all")
        if n_unacceptable_source:
            parts.append(f"{n_unacceptable_source} whose coordinate provenance "
                         f"does not meet the applied standard")
        coord_report["exclusion_note"] = (
            f"{len(dropped)} region(s) were excluded from selection: "
            f"{' and '.join(parts)}. The selection is therefore made over the "
            f"remaining {n_usable} region(s), and the excluded regions are "
            f"neither external nor development.")

    cand = cand[~unusable]
    if cand.empty:
        return {
            "error": (f"no region has coordinates usable for selection: "
                      f"{n_after_class} region(s) passed the class-presence "
                      f"filter, of which {n_no_coordinates} carried no coordinate "
                      f"and {n_unacceptable_source} carried a coordinate whose "
                      f"provenance is not acceptable for this standard. "
                      f"coord_source counts: {by_source}. This is a COORDINATE "
                      f"problem, not a class-coverage problem -- note the "
                      f"distinction when reading this message."),
            "coordinate_report": coord_report,
        }

    strat = cand[stratify_col].astype(str) if stratify_col in cand.columns \
        else pd.Series(["ALL"] * len(cand), index=cand.index)
    cand["_strat"] = strat

    lo, hi = target_n
    strata = sorted(cand["_strat"].unique())
    per_stratum = max(1, int(np.ceil(lo / max(len(strata), 1))))

    chosen: list[str] = []
    for s in strata:
        sub = cand[cand["_strat"] == s]
        if sub.empty:
            continue
        n_take = min(per_stratum, len(sub))
        if maximise_min_distance and len(chosen) > 0:
            # greedily take the region furthest from the current selection
            sel_xy = cand[cand["region"].isin(chosen)][["lon", "lat"]].to_numpy(float)
            for _ in range(n_take):
                rem = sub[~sub["region"].isin(chosen)]
                if rem.empty:
                    break
                d = nearest_distance_km(rem[["lon", "lat"]].to_numpy(float), sel_xy)
                pick = rem.iloc[int(np.argmax(d))]["region"]
                chosen.append(pick)
                sel_xy = cand[cand["region"].isin(chosen)][["lon", "lat"]].to_numpy(float)
        else:
            take = sub.sample(n=n_take, random_state=int(rng.integers(1 << 31)))
            chosen.extend(take["region"].tolist())

    chosen = sorted(set(chosen))
    if len(chosen) > hi:
        chosen = chosen[:hi]

    # THE DEVELOPMENT POOL IS NOT THE CLASS-FILTERED CANDIDATE POOL.
    #
    # The class-presence filter exists so that the EXTERNAL test can evaluate
    # every class. It does not apply to the development pool: a region lacking
    # Bareland is perfectly usable for training and validation. Deriving the
    # development pool as `cand - chosen` (as this did) discarded 24 of 75
    # regions for a reason that does not apply to them, and reported
    # n_development = 39 when the pipeline actually trains on 62 -- a figure that
    # would otherwise have been quoted in the manuscript as fact.
    #
    # The development pool is therefore: every region in the table, minus the
    # external ones, minus those with too few tiles, minus those whose
    # coordinates cannot support a spatial split.
    dev_pool = set(region_table["region"].astype(str)) - set(chosen)
    dev_pool -= thin_all
    if {"lon", "lat"} <= set(region_table.columns):
        rt_xy = region_table.assign(
            region=region_table["region"].astype(str)).set_index("region")
        ok_xy = rt_xy["lon"].notna() & rt_xy["lat"].notna()
        if "coord_source" in rt_xy.columns and acceptable_coord_sources is not None:
            ok_xy = ok_xy & rt_xy["coord_source"].astype(str).isin(
                set(acceptable_coord_sources))
        dev_pool &= set(rt_xy.index[ok_xy])
    dev = sorted(dev_pool)

    def share(regions: Iterable[str]) -> dict[int, float]:
        sub = piv.loc[piv.index.isin(list(regions))]
        tot = float(sub.to_numpy().sum())
        if tot <= 0:
            return {c: 0.0 for c in range(NUM_CLASSES)}
        return {c: float(sub[c].sum()) / tot * 100.0 for c in range(NUM_CLASSES)}

    return {
        "algorithm": "continent-stratified greedy max-min-distance",
        "seed": seed,
        "input_pool_size": int(len(cand)),
        "n_selection_pool": int(len(cand)),
        "selection_pool_note": (
            "The selection pool is restricted to regions where EVERY class "
            "reaches min_class_px, because the external test must be able to "
            "evaluate every class. The development pool is NOT restricted this "
            "way; see development_regions."),
        "class_presence_filter_min_px": min_class_px,
        "target_n_range": list(target_n),
        "external_regions": chosen,
        "development_regions": dev,
        "n_external": len(chosen),
        "n_development": len(dev),
        "coordinate_report": coord_report,
        "class_share_external_pct": share(chosen),
        "class_share_development_pct": share(dev),
        "class_share_divergence_pp": {
            c: round(share(chosen).get(c, 0.0) - share(dev).get(c, 0.0), 4)
            for c in range(NUM_CLASSES)
        },
    }



# --------------------------------------------------------------------------- #
# Leakage diagnostics moved to src/splits/diagnostics.py
#
# The implementation that previously lived here read a literal `split == 'train'`
# label. The compact partition representation produced by random_folds /
# group_folds / spatial_block_folds / leave_one_region_out labels EVERY sample
# 'val' (a sample is validation in exactly one fold and training in the others),
# so that read returned an empty training set and silently nulled every
# diagnostic. The corrected, fold-aware implementation is in
# src/splits/diagnostics.py. Import it from there.
# --------------------------------------------------------------------------- #

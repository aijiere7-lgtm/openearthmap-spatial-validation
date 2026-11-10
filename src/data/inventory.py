"""
inventory.py — dataset discovery, inventory construction and quality control.

Design constraints this module respects:
  * The corpus never enters memory. Statistics are accumulated incrementally,
    one image at a time (host RAM is 15.82 GB; 5,000 x 1024 x 1024 x 3 images
    are ~15.7 GB as uint8 before masks). See DECISIONS D-006.
  * The archive layout is DISCOVERED, not assumed. Nothing about directory
    names, filename conventions or the presence of a metadata table is taken
    for granted.
  * Nothing is silently deleted or excluded. Anomalies are recorded as rows and
    surfaced in the QC report.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from common import (
    CLASS_NAMES, DATA_DIR, LANDCOVER_LABEL_VALUES, NUM_CLASSES,
    PUBLISHED_PIXEL_SHARE_PCT, RAW_DIR,
)

_LANDCOVER_VALUE_SET = set(LANDCOVER_LABEL_VALUES)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# A directory is treated as holding imagery when its name matches one of these.
IMAGE_DIR_HINTS = {"image", "images", "img", "rgb", "aerial", "photos"}
LABEL_DIR_HINTS = {"label", "labels", "mask", "masks", "gt", "annotation",
                   "annotations", "semantic", "seg"}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    root: Path
    image_dir: Path | None = None
    label_dir: Path | None = None
    # Multi-region corpora lay imagery out as <region>/images + <region>/labels.
    # In that case image_dir/label_dir stay None and region_dirs carries the
    # per-region directories, because collapsing them to a single directory
    # would silently truncate the corpus to one region.
    region_dirs: dict[str, tuple[Path, Path]] = field(default_factory=dict)
    multi_region: bool = False
    split_files: dict[str, Path] = field(default_factory=dict)
    metadata_files: list[Path] = field(default_factory=list)
    n_image_files: int = 0
    n_label_files: int = 0
    top_level: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "label_dir": str(self.label_dir) if self.label_dir else None,
            "multi_region": self.multi_region,
            "n_region_dirs": len(self.region_dirs),
            "region_dirs": {
                k: [str(a), str(b)] for k, (a, b) in sorted(self.region_dirs.items())
            },
            "split_files": {k: str(v) for k, v in self.split_files.items()},
            "metadata_files": [str(p) for p in self.metadata_files],
            "n_image_files": self.n_image_files,
            "n_label_files": self.n_label_files,
            "top_level": self.top_level,
            "notes": self.notes,
        }


def _walk_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def discover_layout(root: Path) -> Layout:
    """Identify image/label directories, split files and metadata tables."""
    lay = Layout(root=root)
    if not root.exists():
        lay.notes.append(f"root does not exist: {root}")
        return lay

    lay.top_level = sorted(p.name for p in root.iterdir())

    all_files = list(_walk_files(root))
    by_ext: Counter[str] = Counter(p.suffix.lower() for p in all_files)

    # --- image and label directories --------------------------------------
    # Two layouts are possible and they MUST be distinguished, because the
    # single-directory assumption silently truncates a multi-region corpus.
    # A corpus laid out as <region>/images + <region>/labels presents 97
    # candidate image directories; "pick the one with the most files" would
    # select exactly one region and then report that region's tile count as the
    # dataset size -- a wrong answer produced without any error. Detection is
    # therefore by STRUCTURE (how many parents carry both an image and a label
    # subdirectory), never by file count.
    dir_counts: Counter[Path] = Counter(
        p.parent for p in all_files if p.suffix.lower() in IMAGE_EXT
    )
    img_by_parent: dict[Path, Path] = {}
    lbl_by_parent: dict[Path, Path] = {}
    for d, n in dir_counts.items():
        if not n:
            continue
        nm = d.name.lower()
        if nm in IMAGE_DIR_HINTS:
            img_by_parent[d.parent] = d
        if nm in LABEL_DIR_HINTS:
            lbl_by_parent[d.parent] = d

    paired_parents = sorted(set(img_by_parent) & set(lbl_by_parent))
    if len(paired_parents) >= 2:
        lay.multi_region = True
        for parent in paired_parents:
            lay.region_dirs[parent.name] = (img_by_parent[parent], lbl_by_parent[parent])
        lay.n_image_files = sum(dir_counts[d] for d in img_by_parent.values())
        lay.n_label_files = sum(dir_counts[d] for d in lbl_by_parent.values())
        lay.notes.append(
            f"multi-region layout detected: {len(paired_parents)} directories carry "
            f"both an image and a label subdirectory; imagery is enumerated per "
            f"region rather than from a single directory"
        )
        n_img_dirs_only = len(set(img_by_parent) - set(lbl_by_parent))
        n_lbl_dirs_only = len(set(lbl_by_parent) - set(img_by_parent))
        if n_img_dirs_only or n_lbl_dirs_only:
            lay.notes.append(
                f"unpaired directories ignored: {n_img_dirs_only} image-only, "
                f"{n_lbl_dirs_only} label-only"
            )
    else:
        # Single-directory layout (or a single region): previous behaviour.
        candidates: dict[str, list[tuple[Path, int]]] = {"image": [], "label": []}
        for d, n in dir_counts.items():
            nm = d.name.lower()
            if nm in IMAGE_DIR_HINTS:
                candidates["image"].append((d, n))
            if nm in LABEL_DIR_HINTS:
                candidates["label"].append((d, n))
        for key in ("image", "label"):
            if candidates[key]:
                candidates[key].sort(key=lambda t: (-t[1], str(t[0])))
                setattr(lay, f"{key}_dir", candidates[key][0][0])
                lay.notes.append(
                    f"{key}_dir chosen: {candidates[key][0][0]} "
                    f"({candidates[key][0][1]} files); other candidates: "
                    f"{[str(c[0]) for c in candidates[key][1:]]}"
                )
        if lay.image_dir:
            lay.n_image_files = dir_counts[lay.image_dir]
        if lay.label_dir:
            lay.n_label_files = dir_counts[lay.label_dir]

    # Fallback: infer from extension population if hints failed.
    if lay.image_dir is None and not lay.multi_region:
        ext_rank = sorted(
            ((e, n) for e, n in by_ext.items() if e in IMAGE_EXT),
            key=lambda t: -t[1],
        )
        if ext_rank:
            ext = ext_rank[0][0]
            dirs: Counter[Path] = Counter(
                p.parent for p in all_files if p.suffix.lower() == ext
            )
            if dirs:
                lay.image_dir = dirs.most_common(1)[0][0]
                lay.notes.append(
                    f"image_dir inferred from dominant extension {ext}: "
                    f"{lay.image_dir}"
                )

    # --- split files
    for p in all_files:
        stem = p.stem.lower()
        if p.suffix.lower() in {".txt", ".lst", ".csv"} and stem in {
            "train", "val", "valid", "validation", "test", "trainval"
        }:
            key = {"valid": "val", "validation": "val"}.get(stem, stem)
            lay.split_files[key] = p

    # --- metadata tables
    for p in all_files:
        if p.suffix.lower() == ".csv" and "xbd_files" not in p.name.lower():
            lay.metadata_files.append(p)
        elif p.suffix.lower() in {".json", ".geojson"} and p.stat().st_size < 20_000_000:
            lay.metadata_files.append(p)

    if not lay.split_files:
        lay.notes.append("no official split files found")
    if not lay.metadata_files:
        lay.notes.append("no metadata table found")

    return lay


# --------------------------------------------------------------------------- #
# Corpus root resolution
# --------------------------------------------------------------------------- #
# ONE resolver, shared by every script. Four scripts previously carried their own
# copy of this logic and three of them hard-coded the directory name
# "OpenEarthMap" while accepting the FIRST candidate that matched. That produced
# a silent wrong-answer failure: `extract_class_composition.py` selected the
# outer `data/raw/OpenEarthMap` instead of the inner
# `data/raw/OpenEarthMap/OpenEarthMap_wo_xBD`, and because the inventory's
# `label_path` values are relative to the INNER directory, every one of 2,687
# paths failed to resolve and the script reported "no samples produced pixel
# counts" -- with no indication of why. The root is therefore resolved by
# EVIDENCE (do the inventory's own relative paths resolve beneath it?) rather
# than by name or by file count.

def candidate_corpus_roots(bases: Iterable[Path] | None = None,
                           max_depth: int = 3) -> list[Path]:
    """Enumerate plausible corpus roots under the data directories.

    Deliberately GENERIC: nothing about directory names is assumed. Extraction
    nesting varies (a mirror may add a redundant level), so the search descends
    a bounded number of levels instead of naming expected folders.
    """
    if bases is None:
        bases = [RAW_DIR, DATA_DIR / "interim", DATA_DIR / "processed"]
    out: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        base = Path(base)
        if not base.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(base, 0)]
        while stack:
            d, depth = stack.pop()
            try:
                key = d.resolve()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if depth >= max_depth:
                continue
            try:
                subs = sorted((p for p in d.iterdir() if p.is_dir()),
                              key=lambda p: p.name)
            except OSError:
                continue
            for s in subs:
                stack.append((s, depth + 1))
    return out


def corpus_root_score(root: Path, relpaths: Iterable[str]) -> int:
    """Count how many inventory-relative paths actually exist beneath `root`."""
    hits = 0
    for rel in relpaths:
        try:
            if (root / Path(rel)).exists():
                hits += 1
        except OSError:
            continue
    return hits


def detect_corpus_root_by_structure(bases: Iterable[Path] | None = None,
                                    min_paired: int = 2) -> Path | None:
    """Structure-only detection, for use BEFORE an inventory exists.

    The corpus root is the directory containing the most region directories that
    each carry BOTH an image and a label subdirectory. Detection is by STRUCTURE
    (how many parents carry both), never by file count: a corpus laid out as
    <region>/images + <region>/labels presents 97 candidate image directories,
    and "pick the one with the most files" would silently truncate it to one
    region. The deepest qualifying directory wins, so a redundant outer wrapper
    level is not mistaken for the corpus.
    """
    best: tuple[int, int, Path] | None = None
    for cand in candidate_corpus_roots(bases):
        try:
            subs = [d for d in cand.iterdir() if d.is_dir()]
        except OSError:
            continue
        n_paired = sum(
            1 for d in subs
            if (d / "images").is_dir() and (d / "labels").is_dir()
        )
        if n_paired < min_paired:
            continue
        depth = len(cand.parts)
        # Most paired regions wins; on a tie the DEEPEST directory wins, because
        # an outer wrapper that merely contains the real corpus root would
        # otherwise be selected and would shift every relative path by one level.
        if best is None or (n_paired, depth) > (best[0], best[1]):
            best = (n_paired, depth, cand)
    if best is not None:
        return best[2]

    # Fallback: a single-directory corpus (no per-region structure).
    for cand in candidate_corpus_roots(bases):
        if (cand / "images").is_dir():
            return cand
    return None


def read_inventory_relpaths(inventory_csv: Path, columns: Iterable[str] = (
        "label_path", "image_path"), n_sample: int = 24) -> tuple[list[str], bool]:
    """Sample non-empty relative paths from an inventory, spread across rows.

    Sampling is EVENLY SPACED rather than "the first N". Taking the head would
    concentrate the sample inside whichever region sorts first, so the test would
    be decided by a handful of regions -- the same artefact that was found and
    fixed in `spatial_metadata_audit.py`'s CRS probe.

    Returns (paths, absolute_seen). Absolute paths are reported rather than
    silently joined, because a join would produce a meaningless path.
    """
    import csv as _csv

    values: list[str] = []
    absolute_seen = False
    with Path(inventory_csv).open("r", newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            for col in columns:
                v = (row.get(col) or "").strip()
                if not v:
                    continue
                if Path(v).is_absolute():
                    absolute_seen = True
                values.append(v)
                break
    if not values:
        return [], absolute_seen
    if len(values) <= n_sample:
        return values, absolute_seen
    step = len(values) / n_sample
    picked = [values[min(int(i * step), len(values) - 1)] for i in range(n_sample)]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in picked:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out, absolute_seen


def resolve_inventory_root(inventory_csv: Path | None = None,
                           explicit: str | Path | None = None,
                           bases: Iterable[Path] | None = None,
                           n_sample: int = 24,
                           logger=None) -> tuple[Path | None, dict[str, Any]]:
    """Resolve the directory that an inventory's relative paths are based on.

    Evidence first, structure second. Returns (root, info). `info["method"]`
    records which rule decided, so a caller can log the provenance instead of
    presenting a guess as a fact.
    """
    info: dict[str, Any] = {
        "method": None, "root": None, "n_candidates": 0, "n_sampled": 0,
        "n_hits": 0, "runner_up": None, "runner_up_hits": None,
        "absolute_paths_seen": False, "notes": [],
    }

    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(f"explicit corpus root does not exist: {p}")
        info.update(method="explicit", root=str(p))
        return p, info

    relpaths: list[str] = []
    if inventory_csv is not None and Path(inventory_csv).exists():
        relpaths, absolute_seen = read_inventory_relpaths(inventory_csv,
                                                          n_sample=n_sample)
        info["absolute_paths_seen"] = absolute_seen
        if absolute_seen:
            info["notes"].append(
                "inventory contains absolute paths; the root is not needed to "
                "resolve them and resolve_inventory_path() honours them directly"
            )

    cands = candidate_corpus_roots(bases)
    info["n_candidates"] = len(cands)
    info["n_sampled"] = len(relpaths)

    if relpaths:
        scored: list[tuple[int, int, Path]] = []
        for cand in cands:
            hits = corpus_root_score(cand, relpaths)
            if hits:
                # Most hits wins; on a tie prefer the SHALLOWER directory. A
                # shallower base that still resolves every sampled path is
                # equally usable, and preferring it keeps the choice stable when
                # a nested duplicate tree exists.
                scored.append((hits, -len(cand.parts), cand))
        if scored:
            scored.sort(key=lambda t: (-t[0], -t[1], str(t[2])))
            best_hits, _, best = scored[0]
            info.update(method="inventory_paths", root=str(best),
                        n_hits=best_hits)
            if len(scored) > 1:
                info["runner_up"] = str(scored[1][2])
                info["runner_up_hits"] = scored[1][0]
            if best_hits < len(relpaths):
                info["notes"].append(
                    f"only {best_hits}/{len(relpaths)} sampled paths resolved; "
                    "the inventory may reference files that are no longer on disk"
                )
            if logger:
                logger.info(
                    "corpus root resolved by path evidence: %s (%d/%d sampled "
                    "paths resolve; %d candidate(s) scored)",
                    best, best_hits, len(relpaths), len(scored),
                )
                if info["runner_up"]:
                    logger.info(
                        "  runner-up rejected: %s (%d/%d paths resolve)",
                        info["runner_up"], info["runner_up_hits"], len(relpaths),
                    )
            return best, info

    # No inventory evidence available (or nothing resolved): fall back to
    # structure and SAY SO, because a structure guess cannot be verified.
    struct = detect_corpus_root_by_structure(bases)
    if struct is not None:
        info.update(method="structure", root=str(struct))
        info["notes"].append(
            "no inventory path evidence available; the root was inferred from "
            "directory structure and could not be verified against the inventory"
        )
        if logger:
            logger.warning(
                "corpus root inferred from structure only: %s (no inventory "
                "path evidence to verify it against)", struct,
            )
        return struct, info

    info["notes"].append("no corpus root could be resolved")
    if logger:
        logger.error("could not resolve a corpus root under the data directories")
    return None, info


def resolve_inventory_path(root: Path | None, rel: str) -> Path:
    """Join an inventory-relative path onto a root, honouring absolute paths."""
    p = Path(rel)
    if p.is_absolute() or root is None:
        return p
    return root / p


# --------------------------------------------------------------------------- #
# Region / country / source attribution
# --------------------------------------------------------------------------- #
def load_region_metadata(lay: Layout) -> dict[str, dict[str, Any]]:
    """Return {sample_key: {region, country, source, ...}} if a table exists.

    The key is the file stem, which is what pairs an image with its label.
    Returns an empty dict when no usable table is present — the caller must then
    fall back to filename-derived attribution and record that as a limitation.
    """
    import pandas as pd

    for p in lay.metadata_files:
        if p.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(p, keep_default_na=False, dtype=str)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        name_col = next((cols[c] for c in
                         ("filename", "file", "name", "image", "image_name",
                          "sample_id", "id") if c in cols), None)
        if name_col is None:
            continue
        rec: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = Path(str(row[name_col])).stem
            rec[key] = {c: str(row[cols[c]]) for c in cols}
        return rec
    return {}


def derive_region_from_name(stem: str) -> str:
    """Best-effort region extraction from a filename stem.

    OpenEarthMap filenames generally begin with a region token followed by
    numeric tile indices. This is a FALLBACK: when a metadata table exists, its
    value is authoritative and this function is not used. Any use of this
    fallback is recorded as a limitation in the QC report.
    """
    parts = stem.replace("-", "_").split("_")
    # Take leading non-numeric tokens as the region name.
    out: list[str] = []
    for tok in parts:
        if tok.isdigit():
            break
        out.append(tok)
    return "_".join(out) if out else stem


# --------------------------------------------------------------------------- #
# Per-sample measurement
# --------------------------------------------------------------------------- #
def _read_label(path: Path) -> np.ndarray | None:
    suf = path.suffix.lower()
    try:
        if suf in {".tif", ".tiff"}:
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1)
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im)
    except Exception:
        return None


def _read_image_shape_channels(path: Path) -> tuple[int, int, int, str] | None:
    suf = path.suffix.lower()
    try:
        if suf in {".tif", ".tiff"}:
            import rasterio
            with rasterio.open(path) as src:
                return src.height, src.width, src.count, str(src.dtypes[0])
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
            if arr.ndim == 2:
                return arr.shape[0], arr.shape[1], 1, str(arr.dtype)
            return arr.shape[0], arr.shape[1], arr.shape[2], str(arr.dtype)
    except Exception:
        return None


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def perceptual_hash(path: Path, hash_size: int = 16) -> str | None:
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
@dataclass
class InventoryResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    class_pixel_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_CLASSES + 256, dtype=np.int64)
    )
    label_value_counter: Counter = field(default_factory=Counter)
    per_region: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def build_inventory(
    lay: Layout,
    cfg: dict[str, Any],
    logger,
    limit: int | None = None,
) -> InventoryResult:
    """Stream over the dataset and accumulate inventory + statistics."""
    res = InventoryResult()
    if lay.image_dir is None and not lay.multi_region:
        res.problems.append("no image directory identified; cannot build inventory")
        return res

    qc = cfg.get("qc", {})
    do_sha = bool(qc.get("compute_sha256", True))
    do_phash = bool(qc.get("compute_perceptual_hash", True))
    ph_size = int(qc.get("phash_size", 16))

    meta = load_region_metadata(lay)
    if meta:
        region_source = "metadata_table"
    elif lay.multi_region:
        # The region directory name IS the authoritative region attribution in
        # this layout. Filename parsing is not needed and would disagree with
        # it: the directory is "warminsko-mazurskie" while the stem parses to
        # "warminsko_mazurskie". Recorded so the provenance is explicit.
        region_source = "directory_name"
    else:
        region_source = "filename_parse"
        res.problems.append(
            "no usable metadata table found and no per-region directory layout: "
            "region attribution falls back to filename parsing "
            "(recorded as a limitation)"
        )

    # ---- build the (region, image, label) work list -----------------------
    work: list[tuple[str, Path, Path | None]] = []
    if lay.multi_region:
        for region in sorted(lay.region_dirs):
            idir, ldir = lay.region_dirs[region]
            lab_index = {
                p.stem: p
                for p in ldir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXT
            }
            for img in sorted(
                (p for p in idir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXT),
                key=lambda p: p.name,
            ):
                work.append((region, img, lab_index.get(img.stem)))
    else:
        label_index: dict[str, Path] = {}
        if lay.label_dir:
            label_index = {
                p.stem: p
                for p in lay.label_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXT
            }
        for img in sorted(
            (p for p in lay.image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXT),
            key=lambda p: p.name,
        ):
            work.append((derive_region_from_name(img.stem), img, label_index.get(img.stem)))

    if limit:
        work = work[:limit]
    logger.info(
        "inventory: %d image files over %d %s (region source: %s)",
        len(work),
        len(lay.region_dirs) if lay.multi_region else 1,
        "region directories" if lay.multi_region else "image directory",
        region_source,
    )

    label_value_set: set[int] = set()

    for i, (dir_region, img, lab) in enumerate(work, 1):
        stem = img.stem
        row: dict[str, Any] = {
            "sample_id": stem,
            "image_path": str(img.relative_to(lay.root)),
            "label_path": None,
            "region": None,
            "country": None,
            "source_dataset": None,
            "region_source": region_source,
            "width": None, "height": None, "channels": None, "dtype": None,
            "label_width": None, "label_height": None,
            "n_classes_in_label": None,
            "label_min": None, "label_max": None,
            "image_bytes": None, "label_bytes": None,
            "sha256": None, "phash": None,
            "label_found": False, "image_readable": False, "label_readable": False,
            "notes": "",
        }

        m = meta.get(stem, {})
        if m:
            for cand in ("region", "city", "area", "site"):
                if m.get(cand):
                    row["region"] = m[cand]
                    break
            for cand in ("country", "nation"):
                if m.get(cand):
                    row["country"] = m[cand]
                    break
            for cand in ("source", "source_dataset", "dataset", "provider"):
                if m.get(cand):
                    row["source_dataset"] = m[cand]
                    break
        if row["region"] is None:
            row["region"] = dir_region

        shp = _read_image_shape_channels(img)
        if shp:
            row["height"], row["width"], row["channels"], row["dtype"] = shp
            row["image_readable"] = True
        else:
            row["notes"] = (row["notes"] + "image unreadable; ").strip()
            res.problems.append(f"unreadable image: {img}")

        row["image_bytes"] = img.stat().st_size
        if do_sha:
            row["sha256"] = sha256_of_file(img)
        if do_phash:
            row["phash"] = perceptual_hash(img, ph_size)

        if lab is not None:
            row["label_path"] = str(lab.relative_to(lay.root))
            row["label_bytes"] = lab.stat().st_size
            arr = _read_label(lab)
            if arr is not None:
                row["label_readable"] = True
                row["label_found"] = True
                if arr.ndim == 3:
                    arr = arr[..., 0]
                row["label_height"], row["label_width"] = int(arr.shape[0]), int(arr.shape[1])
                vals, counts = np.unique(arr, return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    res.label_value_counter[int(v)] += int(c)
                    if 0 <= int(v) < len(res.class_pixel_counts):
                        res.class_pixel_counts[int(v)] += int(c)
                label_value_set.update(int(v) for v in vals.tolist())
                row["label_min"] = int(vals.min())
                row["label_max"] = int(vals.max())
                # Count land-cover classes present. The label integer is not the
                # class index (0 is a residual "unknown" category), so the count
                # must be taken over the land-cover values, not over 0..NUM_CLASSES.
                row["n_classes_in_label"] = int(
                    sum(1 for v in vals.tolist() if v in _LANDCOVER_VALUE_SET)
                )
            else:
                row["notes"] = (row["notes"] + "label unreadable; ").strip()
                res.problems.append(f"unreadable label: {lab}")
        else:
            row["notes"] = (row["notes"] + "label missing; ").strip()
            res.problems.append(f"missing label for {stem}")

        # per-region accumulation
        rk = row["region"] or "UNKNOWN"
        pr = res.per_region.setdefault(rk, {
            "region": rk, "country": row["country"], "n_images": 0,
            "n_labels": 0, "n_bad": 0,
        })
        pr["n_images"] += 1
        if row["label_found"]:
            pr["n_labels"] += 1
        if not row["image_readable"] or not row["label_readable"]:
            pr["n_bad"] += 1

        res.rows.append(row)
        if i % 250 == 0:
            logger.info("  ... %d/%d processed", i, len(work))

    res.label_value_set = label_value_set  # type: ignore[attr-defined]
    return res


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #
def find_duplicates(rows: list[dict[str, Any]], hamming_threshold: int = 6,
                    logger=None) -> tuple[list[dict], list[dict]]:
    """Exact (SHA-256) and near (perceptual hash) duplicates."""
    exact: list[dict] = []
    seen: dict[str, str] = {}
    for r in rows:
        h = r.get("sha256")
        if not h:
            continue
        if h in seen:
            exact.append({"sha256": h, "sample_a": seen[h],
                          "sample_b": r["sample_id"]})
        else:
            seen[h] = r["sample_id"]

    near: list[dict] = []
    ph = [(r["sample_id"], r["phash"]) for r in rows if r.get("phash")]
    if ph and logger:
        logger.info("near-duplicate scan over %d perceptual hashes", len(ph))
    try:
        import imagehash
    except ImportError:
        return exact, near

    buckets: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    parsed = []
    for sid, s in ph:
        try:
            parsed.append((sid, imagehash.hex_to_hash(s)))
        except Exception:
            continue
    # LSH-style blocking on the first 8 bits keeps this tractable at n=5000.
    for sid, hh in parsed:
        buckets[int(str(hh)[:2], 16)].append((sid, hh))

    for _, items in buckets.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = items[i][1] - items[j][1]
                if d <= hamming_threshold:
                    near.append({"sample_a": items[i][0], "sample_b": items[j][0],
                                 "hamming": int(d)})
    return exact, near


# --------------------------------------------------------------------------- #
# Writing outputs
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: Iterable[dict[str, Any]],
              fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

"""
common.py — shared paths, logging, seeding and config helpers.

Design rules for this project (see research_protocol.md §17):
  * No hard-coded absolute paths in analysis code. Everything resolves from
    PROJECT_ROOT, which is derived from this file's location.
  * Every run is seeded from config and every run is logged.
  * Outputs go to results/ or data/metadata/ only; nothing is written to the
    project root.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
LITERATURE_DIR = PROJECT_ROOT / "literature"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
STATISTICS_DIR = RESULTS_DIR / "statistics"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "supplementary"

# Canonical locations of derived artefacts. Splits live under data/ so that they
# are versioned alongside the data they partition.
SPLITS_DIR = DATA_DIR / "splits"
EXTERNAL_REGIONS_JSON = METADATA_DIR / "external_regions.json"

# --------------------------------------------------------------------------- #
# Coordinate provenance: TWO DIFFERENT STANDARDS, deliberately kept separate
# --------------------------------------------------------------------------- #
# Whether a coordinate is good enough depends entirely on the scale of the
# quantity it is used to measure. Conflating the two standards would either
# (a) block a legitimate analysis, or (b) license a distance statistic whose
# positional error is the same size as the effect being measured. Both are
# failures, and they fail in opposite directions, so the two sets are named
# separately and neither is derived from the other.

# Acceptable for REGION-LEVEL DISPERSION -- selecting geographically spread
# regions for the external test, and stratifying by continent.
#
# The distances involved are inter-region, of order 1,000-10,000 km. A gazetteer
# city centroid carries positional error of order 10-50 km, which is two orders
# of magnitude smaller and therefore immaterial to the ordering it is used to
# produce.
COORD_SOURCES_OK_FOR_REGION_DISPERSION = {
    "geotiff_centroid",        # measured from the tiles' own georeferencing
    "region_centroid",         # measured from tile extents
    "gazetteer_city_centroid",  # external gazetteer
    "gazetteer_region_centroid",
    "geocoded_city",
    "geocoded_region",
    "approximate",
}

# Acceptable for estimating the DEPENDENCE RANGE r.
#
# The range is of order 1-10 km. The same 10-50 km gazetteer error is now the
# same size as, or larger than, the quantity being estimated, so a variogram
# built on it measures the positional error rather than the spatial dependence.
# Coordinates must therefore come from the imagery's own georeferencing.
COORD_SOURCES_OK_FOR_RANGE = {
    "geotiff_centroid",
    "region_centroid",
}

# Retained as the explicit denial set for r, so a caller can assert on it.
COORD_SOURCES_UNUSABLE_FOR_RANGE = (
    COORD_SOURCES_OK_FOR_REGION_DISPERSION - COORD_SOURCES_OK_FOR_RANGE
)

ALL_DIRS = [
    CONFIG_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, METADATA_DIR,
    EXPERIMENTS_DIR, CHECKPOINTS_DIR, LOGS_DIR,
    TABLES_DIR, FIGURES_DIR, PREDICTIONS_DIR, STATISTICS_DIR,
    SPLITS_DIR, MANUSCRIPT_DIR, SUPPLEMENTARY_DIR,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class _UtcFormatter(logging.Formatter):
    """Formatter whose `%(asctime)s` is UTC.

    `logging.Formatter.converter` is what decides the timezone of `asctime`;
    the default is `time.localtime`. Overriding it is the whole fix.
    """

    converter = time.gmtime


def get_logger(name: str, logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    # The console on this host is GBK, so a log message carrying a Unicode minus
    # (U+2212, which the manuscript uses in tables) raises inside logging.emit and
    # the record is lost with a traceback instead of the message that mattered.
    # Reconfiguring the stream to replace unencodable characters keeps the record.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")      # type: ignore[union-attr]
        except Exception:                              # noqa: BLE001
            pass
    # The timestamp carries a literal "Z", so it must BE UTC. logging's default
    # `%(asctime)s` uses local time, which made every log line in this project
    # read as UTC while being eight hours ahead of it (Asia/Shanghai), i.e. a
    # record that looks more precise than it is. See DECISIONS D-052. Lines
    # written before this change are local time mislabelled Z; both the log
    # files and the registry's `*_at_utc` columns (which have always used
    # `utcnow()`) are reconciled in PROGRESS.md.
    fmt = _UtcFormatter(
        "%(asctime)sZ %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every random source this project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    if path is None:
        path = CONFIG_DIR / "default.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


# --------------------------------------------------------------------------- #
# Hashing and provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json_canonical(obj: Any) -> str:
    """Hash of a JSON object in canonical form (sorted keys, no whitespace).

    Used for the external-regions freeze: the hash must not depend on
    formatting, only on content.
    """
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False))


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unavailable"


def write_json(path: Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# External-freeze accessors
#
# The freeze record nests its region lists under "selection":
#
#     {"selection": {"external_regions": [...], "development_regions": [...]},
#      "content_sha256": "..."}
#
# Two separate modules previously read only the TOP level, found nothing, and
# silently treated the study as having no external test set -- which would have
# computed every headline statistic over the reserved regions while reporting
# that none were excluded. Every reader must go through these helpers.
# --------------------------------------------------------------------------- #
def _region_list(record: Any, key: str) -> list[str]:
    """Collect region names for `key` from a freeze record, at either level."""
    if not isinstance(record, dict):
        return []
    containers = [record]
    if isinstance(record.get("selection"), dict):
        containers.append(record["selection"])
    out: list[str] = []
    for c in containers:
        v = c.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    name = item.get("region") or item.get("name")
                    if name:
                        out.append(str(name))
        if out:
            break
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(out))


def freeze_external_regions(record: Any) -> list[str]:
    """External (reserved) regions from a freeze record, at either nesting."""
    return _region_list(record, "external_regions") or _region_list(record, "regions")


def freeze_development_regions(record: Any) -> list[str]:
    """Development-pool regions from a freeze record, at either nesting."""
    return _region_list(record, "development_regions")


def freeze_content_sha256(record: dict) -> str:
    """Canonical hash of a freeze record, excluding the hash field itself."""
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    return sha256_json_canonical(body)


# --------------------------------------------------------------------------- #
# Dataset constants — verified from primary sources
# (see data/metadata/dataset_source_verification.md)
# --------------------------------------------------------------------------- #
CLASS_NAMES = [
    "Bareland",
    "Rangeland",
    "Developed space",
    "Road",
    "Tree",
    "Water",
    "Agriculture land",
    "Building",
]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- #
# LABEL ENCODING — the label integer is NOT the class index
# --------------------------------------------------------------------------- #
# This is the single most dangerous detail in the dataset, because getting it
# wrong mislabels every per-class metric while producing plausible-looking
# numbers.  The label rasters store
#
#     0 = "unknown"      1 = Bareland    2 = Grass      3 = Pavement
#     4 = Road           5 = Tree        6 = Water      7 = Cropland
#     8 = buildings
#
# so the paper's class k is stored as integer k+1.  Source: the dataset authors'
# own code, `open_earth_map/utils.py`:
#
#     class_grey_oem = {"unknown": 0, "Bareland": 1, "Grass": 2, "Pavement": 3,
#                       "Road": 4, "Tree": 5, "Water": 6, "Cropland": 7,
#                       "buildings": 8}
#
# and `open_earth_map/dataset.py`, whose default is `n_classes: int = 9` and
# whose `ToTensor(classes=np.arange(n_classes))` one-hot encodes all nine
# integers with no shift and no ignore index.  The RGB values in the same dict
# match the paper's Table 2 hex codes one-for-one and in the same order, which
# is what pins the mapping to the paper's class names.
#
# Decision (see DECISIONS D-022): the study trains and reports on the EIGHT
# land-cover classes, treating integer 0 as the ignore index, because "unknown"
# is not a land-cover class and the paper's published shares are defined over
# the eight.  The nine-class variant that the official baseline uses is reported
# as a sensitivity analysis.
LABEL_VALUE_UNKNOWN = 0
# paper class index -> the integer stored in the label raster
CLASS_INDEX_TO_LABEL_VALUE = {k: k + 1 for k in range(NUM_CLASSES)}
LABEL_VALUE_TO_CLASS_INDEX = {v: k for k, v in CLASS_INDEX_TO_LABEL_VALUE.items()}
# integers that denote a real land-cover class
LANDCOVER_LABEL_VALUES = sorted(CLASS_INDEX_TO_LABEL_VALUE.values())
# every integer the official release may contain
ALL_LABEL_VALUES_INCLUDING_UNKNOWN = [LABEL_VALUE_UNKNOWN, *LANDCOVER_LABEL_VALUES]
# the official baseline's class count, which counts "unknown" as a class
OFFICIAL_BASELINE_N_CLASSES = 9

# The ignore index used by the TRAINING pipeline.  It is deliberately NOT
# LABEL_VALUE_UNKNOWN: after remapping, labels are class indices in [0, 8), so
# the ignore index must lie outside that range.  Using 0 would silently mark
# every Bareland pixel as ignore.  `remap_labels` in data/dataset.py performs the
# raster-value -> class-index conversion, mapping the raster's "unknown" value
# onto this index.
IGNORE_INDEX = 255

# Hex colours as published by the dataset authors (paper Table 2).
CLASS_COLORS_HEX = {
    0: "800000", 1: "00FF24", 2: "949494", 3: "FFFFFF",
    4: "226126", 5: "0045FF", 6: "4BB549", 7: "DE1F07",
}

# Published pixel shares (%), used as an independent cross-check on the
# inventory. If measured shares diverge substantially, the data are not what
# they claim to be and the QC gate must fail.
PUBLISHED_PIXEL_SHARE_PCT = {
    0: 1.5, 1: 22.9, 2: 16.1, 3: 6.7, 4: 20.2, 5: 3.3, 6: 13.7, 7: 15.6,
}

# Published dataset-level facts, for cross-checking the inventory.
PUBLISHED_N_IMAGES = 5000
PUBLISHED_IMAGE_SIZE = 1024
PUBLISHED_N_REGIONS = 97
PUBLISHED_N_COUNTRIES = 44
PUBLISHED_SEGMENTS = 2_205_395
PUBLISHED_AREA_KM2 = 799

# --------------------------------------------------------------------------- #
# AUDITED RELEASE CONTENT — what the distribution actually ships
# --------------------------------------------------------------------------- #
# The published figures above describe the dataset as the authors define it.
# They do NOT describe the released archive, which is smaller for two structural
# reasons (see data/metadata/dataset_release_audit.md and decisions D-020/D-021):
#
#   * xBD RGB imagery is not distributed, so 1,162 tiles have labels but no
#     image.  The count of images that are xBD targets is 0.
#   * The official test labels are withheld so that the CodaLab leaderboard is
#     meaningful, so 1,500 tiles have no label.  Zero test tiles carry a label.
#
# Every number below was measured on this host by scripts/audit_*.py and
# scripts/region_availability.py.  The inventory cross-checks against them, so a
# future change to the corpus raises a gate failure instead of passing silently.
RELEASE_EXPECTED_N_IMAGES = 3838          # 5000 - 1162 xBD tiles
RELEASE_EXPECTED_N_LABELS = 3500          # 3000 train + 500 val
RELEASE_EXPECTED_N_USABLE = 2687          # image AND label present
RELEASE_EXPECTED_N_REGION_DIRS = 97
RELEASE_EXPECTED_N_REGIONS_WITH_IMAGERY = 75   # 97 - 22 entirely-xBD regions
RELEASE_EXPECTED_N_XBD_TILES = 1162
RELEASE_EXPECTED_N_REGIONS_ALL_XBD = 22   # regions with no imagery at all
RELEASE_EXPECTED_N_REGIONS_ELIGIBLE = 74  # >= 4 usable tiles
RELEASE_EXPECTED_SPLIT = {"train": 3000, "val": 500, "test": 1500}
# The official test split is unusable as a reference: its labels are not
# distributed.  A region-held-out external test is therefore the only option,
# not a methodological preference (D-021).
OFFICIAL_TEST_LABELS_AVAILABLE = False

# --------------------------------------------------------------------------- #
# PRE-REGISTERED ANALYSIS SCOPE — the frozen design set (D-060)
# --------------------------------------------------------------------------- #
# The protocol (research_protocol.md §13.4) fixes ONE primary contrast and
# requires every family of secondary tests to be defined BEFORE the results are
# seen. That scope is the five designs below.
#
# Plan stages 8-9 added four P1 sensitivity strategies (the block-size and
# buffer-width sweeps at 0.5 r and 2 r). When the 75-run registry is analysed,
# `paired_contrasts()` — which, without an explicit list, pairs EVERY strategy
# present in the frame — would silently inflate the family from the
# pre-registered ten contrasts to thirty-six and change every adjusted
# p-value, with the old gate unable to notice (it only checked that the
# `family` field was non-null). The scope is therefore named here once and
# imported by the analysis and figure scripts, so that a later addition to the
# grid can never widen a pre-registered family by accident (D-060).
PRE_REGISTERED_DESIGNS = (
    "S0_random_patch",
    "S2_spatial_block_1r",
    "S3_buffered_1r",
    "S4_leave_one_region_out",
    "MATCHED_S3_buffered_1r_N1324_rep0",
)

# P1 sensitivity strategies, e.g. "S2_spatial_block_0.5r".
#
# ANCHORED AT BOTH ENDS (fullmatch). The matched control's own name embeds the
# words "S3_buffered_1r" inside "MATCHED_S3_buffered_1r_N1324_rep0", and an
# unanchored search parses it as a buffer-sweep point at 1 r — merging nine
# matched-control runs into the sweep's 1 r cell. That is the same class of
# mixing defect D-057 was written to prevent, caught again before the first
# sweep figure was rendered (D-060).
_SWEEP_STRATEGY_RE = re.compile(
    r"(S2_spatial_block|S3_buffered)_(\d+(?:\.\d+)?)r")


def parse_sweep_strategy(name: str) -> tuple[str, float] | None:
    """('block'|'buffer', multiple of r) for a P1 sweep strategy, else None."""
    m = _SWEEP_STRATEGY_RE.fullmatch(str(name).strip())
    if m is None:
        return None
    family = "block" if m.group(1) == "S2_spatial_block" else "buffer"
    return family, float(m.group(2))

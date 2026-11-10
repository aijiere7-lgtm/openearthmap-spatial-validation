"""test_label_encoding.py — regression guards for the label encoding.

Why this file exists
--------------------
OpenEarthMap label rasters do NOT store class indices. The authors' own code
(`open_earth_map/utils.py`) defines

    unknown=0, Bareland=1, Grass=2, Pavement=3, Road=4,
    Tree=5, Water=6, Cropland=7, buildings=8

so paper class *k* is stored as integer *k+1* and integer 0 is a residual
category. Every plausible-but-wrong assumption here fails SILENTLY and produces
plausible-looking metrics:

* assuming `raster value == class index` labels every "unknown" pixel Bareland,
  and makes Building (value 8) an out-of-range code that is discarded;
* setting `IGNORE_INDEX = 0` collides with class index 0 (Bareland) and marks
  every Bareland pixel as ignore;
* using `[0, NUM_CLASSES)` as the validity test rejects Building.

None of these raises. They change the numbers. So they get explicit assertions.

Run:  python tests/test_label_encoding.py
Exit: 0 if all assertions pass, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    ALL_LABEL_VALUES_INCLUDING_UNKNOWN, CLASS_COLORS_HEX,
    CLASS_INDEX_TO_LABEL_VALUE, CLASS_NAMES, IGNORE_INDEX,
    LABEL_VALUE_TO_CLASS_INDEX, LABEL_VALUE_UNKNOWN, LANDCOVER_LABEL_VALUES,
    NUM_CLASSES, OFFICIAL_BASELINE_N_CLASSES,
)
from data.dataset import count_unmapped_values, remap_labels  # noqa: E402
from evaluation.metrics import ConfusionMatrix  # noqa: E402

PASS = 0
FAIL = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"  -- {detail}" if detail else ""))


def section(name: str) -> None:
    print()
    print(f"=== {name} ===")


# --------------------------------------------------------------------------- #
def test_constants_are_self_consistent() -> None:
    section("1. Encoding constants are self-consistent")

    check("NUM_CLASSES is 8 (the paper's taxonomy)",
          NUM_CLASSES == 8, f"got {NUM_CLASSES}")
    check("class names are the paper's eight, in the paper's order",
          CLASS_NAMES == ["Bareland", "Rangeland", "Developed space", "Road",
                          "Tree", "Water", "Agriculture land", "Building"],
          str(CLASS_NAMES))

    check("class index -> label value is k+1",
          CLASS_INDEX_TO_LABEL_VALUE == {k: k + 1 for k in range(NUM_CLASSES)},
          str(CLASS_INDEX_TO_LABEL_VALUE))
    check("the two maps are exact inverses",
          all(LABEL_VALUE_TO_CLASS_INDEX[v] == k
              for k, v in CLASS_INDEX_TO_LABEL_VALUE.items()))
    check("land-cover values are 1..8 and exclude the residual value",
          LANDCOVER_LABEL_VALUES == list(range(1, 9)),
          str(LANDCOVER_LABEL_VALUES))
    check("the residual value 0 is not a land-cover class",
          LABEL_VALUE_UNKNOWN == 0 and 0 not in LANDCOVER_LABEL_VALUES)
    check("every label value the release may contain is enumerated",
          ALL_LABEL_VALUES_INCLUDING_UNKNOWN == list(range(0, 9)),
          str(ALL_LABEL_VALUES_INCLUDING_UNKNOWN))
    check("the official baseline trains 9 classes (unknown counted as one)",
          OFFICIAL_BASELINE_N_CLASSES == 9)


def test_ignore_index_is_outside_the_class_range() -> None:
    section("2. IGNORE_INDEX cannot collide with a class index")

    check("IGNORE_INDEX lies outside [0, NUM_CLASSES)",
          not (0 <= IGNORE_INDEX < NUM_CLASSES), f"IGNORE_INDEX={IGNORE_INDEX}")
    # This is the specific trap: 0 is the raster's residual value AND class index
    # 0 is Bareland, so reusing the raster value as the ignore index silently
    # deletes a real class from every metric.
    check("IGNORE_INDEX is NOT the raster's residual value",
          IGNORE_INDEX != LABEL_VALUE_UNKNOWN,
          f"both are {IGNORE_INDEX}")
    check("IGNORE_INDEX is not any label value the release defines",
          IGNORE_INDEX not in ALL_LABEL_VALUES_INCLUDING_UNKNOWN)


def test_remap_labels() -> None:
    section("3. remap_labels performs the k+1 -> k conversion")

    raw = np.arange(0, 9, dtype=np.uint8).reshape(3, 3)
    out = remap_labels(raw)

    # value 0 -> ignore
    check("raster value 0 (unknown) -> ignore index",
          int(out[0, 0]) == IGNORE_INDEX, f"got {int(out[0, 0])}")
    # value k+1 -> class index k
    ok = all(int(out.reshape(-1)[v]) == v - 1 for v in range(1, 9))
    check("raster value k+1 -> class index k for k=0..7", ok,
          str(out.reshape(-1).tolist()))
    check("Building (raster 8) survives as class index 7",
          int(out.reshape(-1)[8]) == 7, f"got {int(out.reshape(-1)[8])}")
    check("Bareland (raster 1) becomes class index 0, not ignore",
          int(out.reshape(-1)[1]) == 0)

    # Genuinely unexpected codes
    weird = np.array([[9, 17, 200], [255, 1, 8]], dtype=np.uint8)
    wo = remap_labels(weird)
    check("codes the release does not define -> ignore",
          int(wo[0, 0]) == IGNORE_INDEX and int(wo[0, 1]) == IGNORE_INDEX
          and int(wo[0, 2]) == IGNORE_INDEX and int(wo[1, 0]) == IGNORE_INDEX,
          str(wo.tolist()))
    check("...while real classes in the same array are preserved",
          int(wo[1, 1]) == 0 and int(wo[1, 2]) == 7, str(wo.tolist()))

    # Non-uint8 input must not silently wrap
    big = np.array([[0, 1, 8, 9, 300]], dtype=np.int32)
    bo = remap_labels(big)
    check("non-uint8 input is handled without wrapping",
          int(bo[0, 0]) == IGNORE_INDEX and int(bo[0, 1]) == 0
          and int(bo[0, 2]) == 7 and int(bo[0, 3]) == IGNORE_INDEX
          and int(bo[0, 4]) == IGNORE_INDEX,
          str(bo.tolist()))

    # A custom ignore index must still be honoured
    co = remap_labels(raw, ignore_index=99)
    check("a caller-supplied ignore index is honoured",
          int(co[0, 0]) == 99 and int(co.reshape(-1)[1]) == 0,
          str(co.reshape(-1).tolist()))

    # The original array must not be mutated in place
    raw2 = np.arange(0, 9, dtype=np.uint8)
    before = raw2.copy()
    remap_labels(raw2)
    check("the input array is not mutated", np.array_equal(raw2, before))


def test_count_unmapped_values() -> None:
    section("4. count_unmapped_values surfaces unexpected codes")

    arr = np.array([[0, 0, 1], [8, 8, 200]], dtype=np.uint8)
    un = count_unmapped_values(arr)
    check("the residual value and unexpected codes are both reported",
          un.get(0) == 2 and un.get(200) == 1, str(un))
    check("real land-cover values are not reported as unmapped",
          1 not in un and 8 not in un, str(un))

    clean = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    check("a clean array reports nothing",
          count_unmapped_values(clean) == {}, str(count_unmapped_values(clean)))


def test_class_index_matches_paper_table_order() -> None:
    section("5. Class indices align with the paper's Table 2 order")

    # The mapping is pinned to the paper by the colours: class_grey_oem in the
    # authors' code carries the same hex values as Table 2, in the same order.
    # If CLASS_NAMES were reordered without reordering CLASS_COLORS_HEX, the
    # per-class table in the manuscript would be wrong while looking fine.
    expected_hex = ["800000", "00FF24", "949494", "FFFFFF",
                    "226126", "0045FF", "4BB549", "DE1F07"]
    check("class colour hex matches the paper's Table 2 order",
          [CLASS_COLORS_HEX[k] for k in range(NUM_CLASSES)] == expected_hex,
          str([CLASS_COLORS_HEX[k] for k in range(NUM_CLASSES)]))
    check("the number of colours equals the number of class names",
          len(CLASS_COLORS_HEX) == len(CLASS_NAMES))


def test_metrics_ignore_the_residual_value() -> None:
    section("6. ConfusionMatrix excludes the residual pixels")

    # A 3x3 label tile with two 'unknown' pixels. After remap, those are ignore
    # and must not enter the confusion matrix at all.
    raw = np.array([[0, 1, 1], [8, 8, 8], [2, 2, 0]], dtype=np.uint8)
    y = remap_labels(raw)
    y_true = y.reshape(-1)
    n_ignored = int((y_true == IGNORE_INDEX).sum())
    check("two residual pixels become ignore", n_ignored == 2, str(n_ignored))

    pred = y_true.copy()                      # perfect prediction on real classes
    cm = ConfusionMatrix(n_classes=NUM_CLASSES)
    cm.update(pred.reshape(1, -1), y_true.reshape(1, -1), ignore_index=IGNORE_INDEX)
    total = int(cm.mat.sum())
    check("ignored pixels do not enter the confusion matrix",
          total == int((y_true != IGNORE_INDEX).sum()),
          f"matrix total {total} vs kept {int((y_true != IGNORE_INDEX).sum())}")

    m = cm.all_metrics()
    check("a perfect prediction on the kept pixels scores mIoU 1.0",
          abs(float(m["miou"]) - 1.0) < 1e-9, f"got {m['miou']}")
    check("n_pixels equals the kept (non-ignored) count",
          int(m["n_pixels"]) == int((y_true != IGNORE_INDEX).sum()),
          f"got {m['n_pixels']}")
    # The end-to-end proof that raster 8 is Building: it must land in the
    # building metric, and the 'bareland' metric must be driven by raster 1.
    # raw = [[0,1,1],[8,8,8],[2,2,0]] -> raster 8 occurs 3x, raster 1 occurs 2x.
    check("raster value 8 reaches the Building metric",
          m.get("support_building") == 3, str(m.get("support_building")))
    check("raster value 1 reaches the Bareland metric",
          m.get("support_bareland") == 2, str(m.get("support_bareland")))
    check("Bareland support equals the raster-1 count, not the residual count",
          m.get("support_bareland") == 2 and m.get("iou_bareland") == 1.0,
          f"support={m.get('support_bareland')} iou={m.get('iou_bareland')}")


def main() -> int:
    print("=" * 72)
    print("LABEL ENCODING TESTS")
    print("=" * 72)
    test_constants_are_self_consistent()
    test_ignore_index_is_outside_the_class_range()
    test_remap_labels()
    test_count_unmapped_values()
    test_class_index_matches_paper_table_order()
    test_metrics_ignore_the_residual_value()

    print()
    print("=" * 72)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

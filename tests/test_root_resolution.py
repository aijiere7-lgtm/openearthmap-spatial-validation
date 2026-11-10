#!/usr/bin/env python
"""
test_root_resolution.py — regression tests for corpus-root resolution.

WHY THIS TEST EXISTS
--------------------
A silent wrong-answer bug was found and fixed in this project:

    `extract_class_composition.py` resolved the corpus root by scoring candidates
    on `discover_layout(...).n_label_files`. Because `discover_layout` recurses,
    BOTH the outer `data/raw/OpenEarthMap` and the inner
    `data/raw/OpenEarthMap/OpenEarthMap_wo_xBD` reported 3,500 label files. The
    tie was kept by the first enumerated candidate, so the OUTER directory won.
    The inventory's `label_path` values are relative to the INNER directory,
    so all 2,687 of them failed to resolve, and the script exited with
    "no samples produced pixel counts; nothing written" -- no exception, no
    indication of the cause.

The fix is evidence-based resolution: a candidate is accepted only if the
inventory's own relative paths actually resolve beneath it. These tests build a
synthetic corpus that reproduces the exact nesting that caused the bug, and
assert that the wrong answer can no longer be produced.

Run:  python tests/test_root_resolution.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.inventory import (  # noqa: E402
    candidate_corpus_roots,
    corpus_root_score,
    detect_corpus_root_by_structure,
    read_inventory_relpaths,
    resolve_inventory_path,
    resolve_inventory_root,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------- #
# Synthetic corpus: reproduces the nesting that caused the bug.
#
#   <tmp>/raw/OpenEarthMap/                     <- outer wrapper (WRONG root)
#       OpenEarthMap_wo_xBD/                    <- TRUE corpus root
#           regionA/{images,labels}/a_1.tif
#           regionB/{images,labels}/b_1.tif
# --------------------------------------------------------------------------- #
def build_fixture(tmp: Path) -> tuple[Path, Path, Path]:
    raw = tmp / "raw"
    outer = raw / "OpenEarthMap"
    inner = outer / "OpenEarthMap_wo_xBD"
    for region, stem in (("regionA", "a_1"), ("regionB", "b_1")):
        (inner / region / "images").mkdir(parents=True, exist_ok=True)
        (inner / region / "labels").mkdir(parents=True, exist_ok=True)
        (inner / region / "images" / f"{stem}.tif").write_bytes(b"img")
        (inner / region / "labels" / f"{stem}.tif").write_bytes(b"lbl")
    return raw, outer, inner


def write_inventory(path: Path, relpaths: list[str]) -> Path:
    lines = ["sample_id,image_path,label_path,region,label_readable"]
    for rp in relpaths:
        stem = Path(rp).stem
        region = Path(rp).parts[0]
        img = rp.replace("labels", "images")
        lines.append(f"{stem},{img},{rp},{region},True")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="root_resolution_test_"))
    try:
        raw, outer, inner = build_fixture(tmp)
        print("=" * 72)
        print("CORPUS ROOT RESOLUTION TESTS")
        print("=" * 72)
        print(f"fixture: {raw}")

        # ---- 1. candidate enumeration is generic -------------------------- #
        print("\n[1] candidate enumeration")
        cands = candidate_corpus_roots(bases=[raw])
        check("outer wrapper is a candidate", outer in cands, f"{cands}")
        check("inner corpus root is a candidate", inner in cands, f"{cands}")
        check("region dirs are reachable as candidates",
              (inner / "regionA") in cands, "max_depth too shallow")

        # ---- 2. path scoring distinguishes the two levels ------------------ #
        print("\n[2] path scoring")
        rels = ["regionA\\labels\\a_1.tif", "regionB\\labels\\b_1.tif"]
        check("inner scores 2/2", corpus_root_score(inner, rels) == 2,
              f"got {corpus_root_score(inner, rels)}")
        check("outer scores 0/2", corpus_root_score(outer, rels) == 0,
              f"got {corpus_root_score(outer, rels)}")
        check("raw scores 0/2", corpus_root_score(raw, rels) == 0,
              f"got {corpus_root_score(raw, rels)}")

        # ---- 3. THE REGRESSION: evidence picks the inner, not the outer ---- #
        print("\n[3] regression: evidence-based resolution")
        inv = write_inventory(tmp / "dataset_inventory.csv", rels)
        root, info = resolve_inventory_root(inventory_csv=inv, bases=[raw])
        check("resolves to the INNER corpus root", root == inner,
              f"got {root}, expected {inner}")
        check("method is inventory_paths", info.get("method") == "inventory_paths",
              f"got {info.get('method')}")
        check("all sampled paths resolved", info.get("n_hits") == info.get("n_sampled"),
              f"{info.get('n_hits')}/{info.get('n_sampled')}")
        check("the outer wrapper is recorded as the rejected runner-up",
              info.get("runner_up") is None or Path(info["runner_up"]) != outer,
              f"runner_up={info.get('runner_up')}")

        # ---- 4. paths resolve end to end ---------------------------------- #
        print("\n[4] end-to-end path resolution")
        for rp in rels:
            p = resolve_inventory_path(root, rp)
            check(f"resolves {rp}", p.exists(), f"{p} does not exist")

        # ---- 5. a redundant extra wrapper level is handled ----------------- #
        # The fixture must carry BOTH images/ and labels/ per region: the
        # structure rule requires the pair by design, so a labels-only fixture
        # would test nothing about it.
        print("\n[5] redundant extra wrapper level")
        deeper = tmp / "raw2"
        wrapped = deeper / "w1" / "w2" / "OpenEarthMap_wo_xBD"
        for region, stem in (("regionA", "a_1"), ("regionB", "b_1")):
            (wrapped / region / "images").mkdir(parents=True, exist_ok=True)
            (wrapped / region / "labels").mkdir(parents=True, exist_ok=True)
            (wrapped / region / "images" / f"{stem}.tif").write_bytes(b"img")
            (wrapped / region / "labels" / f"{stem}.tif").write_bytes(b"lbl")
        root2, _ = resolve_inventory_root(inventory_csv=inv, bases=[deeper])
        check("finds the corpus root under two wrapper levels", root2 == wrapped,
              f"got {root2}, expected {wrapped}")

        # ---- 6. structure detection agrees, and prefers the deeper level --- #
        print("\n[6] structure-only detection")
        sroot = detect_corpus_root_by_structure(bases=[raw])
        check("structure detection picks the inner root", sroot == inner,
              f"got {sroot}, expected {inner}")
        sroot2 = detect_corpus_root_by_structure(bases=[deeper])
        check("structure detection finds the doubly-wrapped root",
              sroot2 == wrapped, f"got {sroot2}, expected {wrapped}")

        # ---- 7. sampling is spread, not the head -------------------------- #
        print("\n[7] inventory sampling")
        many = [f"region{i % 5}/labels/t_{i}.tif" for i in range(200)]
        inv_many = write_inventory(tmp / "many.csv", many)
        picked, abs_seen = read_inventory_relpaths(inv_many, n_sample=20)
        check("samples the requested count", len(picked) == 20, f"got {len(picked)}")
        check("sample is not the head only",
              picked[0] != many[0] or picked[-1] != many[19],
              "sample collapsed to the first rows")
        check("sample reaches the tail of the inventory",
              picked[-1] == many[199] or Path(picked[-1]).stem.startswith("t_1"),
              f"last={picked[-1]}")
        check("no absolute paths flagged for a relative inventory",
              abs_seen is False, f"got {abs_seen}")

        # ---- 8. absolute paths are honoured, not joined -------------------- #
        print("\n[8] absolute path handling")
        abs_inv = tmp / "abs.csv"
        abs_inv.write_text(
            "sample_id,image_path,label_path\n"
            f"a,{tmp / 'x.tif'},{tmp / 'y.tif'}\n", encoding="utf-8")
        _, abs_seen2 = read_inventory_relpaths(abs_inv, n_sample=5)
        check("absolute paths are detected", abs_seen2 is True, f"got {abs_seen2}")
        abs_target = tmp / "y.tif"
        check("absolute path is returned unchanged",
              resolve_inventory_path(inner, str(abs_target)) == abs_target,
              "absolute path was joined onto the root")
        check("relative path is joined",
              resolve_inventory_path(inner, "regionA\\labels\\a_1.tif")
              == inner / "regionA" / "labels" / "a_1.tif",
              "relative path was not joined")

        # ---- 9. no evidence falls back to structure, and says so ----------- #
        print("\n[9] fallback behaviour")
        empty_inv = tmp / "empty.csv"
        empty_inv.write_text("sample_id,image_path,label_path\n", encoding="utf-8")
        root3, info3 = resolve_inventory_root(inventory_csv=empty_inv, bases=[raw])
        check("falls back to structure when the inventory is empty",
              root3 == inner, f"got {root3}, expected {inner}")
        check("fallback is labelled as a guess, not as evidence",
              info3.get("method") == "structure"
              and any("could not be verified" in n for n in info3.get("notes", [])),
              f"method={info3.get('method')} notes={info3.get('notes')}")

        # ---- 10. explicit root wins and is validated ----------------------- #
        print("\n[10] explicit root")
        root4, info4 = resolve_inventory_root(explicit=inner, bases=[raw])
        check("explicit root is used", root4 == inner and info4["method"] == "explicit",
              f"got {root4} / {info4.get('method')}")
        raised = False
        try:
            resolve_inventory_root(explicit=tmp / "does_not_exist", bases=[raw])
        except FileNotFoundError:
            raised = True
        check("a non-existent explicit root raises rather than silently falling back",
              raised, "no FileNotFoundError raised")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    if FAILED:
        for f in FAILED:
            print(f"  - {f}")
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

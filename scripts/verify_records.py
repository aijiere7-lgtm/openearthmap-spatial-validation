"""Verify archived run integrity and reproduce final mIoU from class IoUs."""
import csv
import hashlib
import json
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def main():
    freeze = json.loads((ROOT / "data/metadata/external_regions.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in freeze.items() if key != "content_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    assert digest == freeze["content_sha256"]
    ref = set(freeze["selection"]["external_regions"])
    assert len(ref) == 12
    composition = read_csv("data/metadata/reference_region_composition.csv")
    assert {r["region"] for r in composition} == ref
    assert sum(int(r["n_tiles"]) for r in composition) == 470

    manifest = read_csv("experiments/record_provenance.csv")
    archived = read_csv("experiments/run_metrics.csv")
    reported = {r["run_id"]: r for r in archived}
    files = {p.stem:p for p in (ROOT / "experiments/records").glob("*.json")}
    assert len(manifest) == len(archived) == len(files) == 75
    assert {r["run_id"] for r in manifest} == set(files) == set(reported)
    counts = Counter()
    retained_not_maximum = 0
    largest = 0
    for entry in manifest:
        key = entry["run_id"]
        data = files[key].read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["archived_sha256"]
        record = json.loads(data)
        row, fit = record["row"], record["fit"]
        expected = reported[key]
        assert row["run_id"] == key
        assert row["external_freeze_sha256"] == digest
        assert row["started_at_utc"] > freeze["frozen_at_utc"]
        assert set(record["ext_metrics_by_region"]) == ref
        assert row["fold"] == 0 and row["epochs_run"] == 20
        assert row["strategy"] == expected["protocol"] and row["model_id"] == expected["model"]
        assert (row["seed"],row["n_train"],row["n_val"]) == tuple(int(expected[k]) for k in ("seed","n_train","n_validation"))
        counts[row["strategy"]] += 1

        history = fit["history"]
        assert [e["epoch"] for e in history] == list(range(1, 21))
        best = history[0]
        for epoch in history[1:]:
            if epoch["val_miou"] > best["val_miou"] + 0.0001:
                best = epoch
        assert fit["best_epoch"] == best["epoch"]
        assert fit["best_val_metrics"]["miou"] == best["val_miou"]
        retained_not_maximum += best != max(history, key=lambda e:e["val_miou"])
        largest = max(largest, abs(best["val_miou"] - row["val_miou"]))

        for set_key,row_key,field in (("val_metrics","val_miou","validation_miou"),
                                      ("ext_metrics","ext_miou","reference_miou")):
            metrics = record[set_key]
            ious = [v for k,v in metrics.items() if k.startswith("iou_") and v is not None]
            assert len(ious) == 8
            assert isclose(mean(ious), metrics["miou"], rel_tol=0, abs_tol=1e-12)
            assert metrics["miou"] == row[row_key] == float(expected[field])

    design = read_csv("supplementary_materials/Table_S1_Executed_design.csv")
    assert counts == {r["protocol"]:int(r["runs"]) for r in design}
    assert len(counts) == 9 and retained_not_maximum == 2
    assert isclose(largest, 0.0030153474802477875, rel_tol=0, abs_tol=1e-12)
    print("Verified 75 JSON records, 1,500 epoch entries, 150 pooled scores and the 12-region freeze.")

if __name__ == "__main__":
    main()

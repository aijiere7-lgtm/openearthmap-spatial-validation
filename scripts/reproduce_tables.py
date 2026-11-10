"""Export the main manuscript tables and supporting tables as CSV and HTML."""
import argparse
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "supplementary_materials"
MAIN = ("S0_random_patch", "S2_spatial_block_1r", "S3_buffered_1r", "S4_leave_one_region_out")

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))

def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/tables")
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SUP / "summary.json").read_text(encoding="utf-8"))
    raw_design = read(SUP / "Table_S1_Executed_design.csv")
    design = {r[0]:dict(zip(raw_design[0],r)) for r in raw_design[1:]}
    t1 = [["Protocol", "Training tiles", "Validation tiles", "Buffer-excluded tiles", "Seeds per model", "Runs"]]
    t2 = [["Protocol", "Validation score", "Reference score", "Signed gap", "Absolute gap", "Shared-seed absolute gap"]]
    for i, (protocol, name) in enumerate(zip(MAIN, summary["protocols"])):
        d = design[protocol]
        t1.append([name, d["training"], d["validation"], d["excluded"], len(d["seeds"].split()), d["runs"]])
        a, b = summary["all_runs"][i], summary["shared_runs"][i]
        t2.append([name, f"{a['validation']:.4f}", f"{a['reference']:.4f}", f"{a['gap']:+.4f}",
                   f"{a['absolute']:.4f}", f"{b['absolute']:.4f}"])
    write(out / "Table_1.csv", t1)
    write(out / "Table_2.csv", t2)

    sources = [
        ("Table S1. Executed fold-0 design", SUP / "Table_S1_Executed_design.csv"),
        ("Table S2. Reported run scores", ROOT / "experiments/run_metrics.csv"),
        ("Table S3. Seed-level dispersion", SUP / "Table_S3_Seed_dispersion.csv"),
        ("Table S4. Dispersion across observed reference regions", SUP / "Table_S4_Region_dispersion.csv"),
        ("Table S5. Mean class IoU", SUP / "Table_S5_Class_scores.csv"),
        ("Reference region composition", ROOT / "data/metadata/reference_region_composition.csv"),
    ]
    with (out / "Supplementary_Tables.html").open("w", encoding="utf-8") as handle:
        handle.write('<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenEarthMap supplementary tables</title>')
        handle.write('<style>body{font:15px Arial,sans-serif;margin:2rem;color:#222}table{border-collapse:collapse;margin-bottom:2rem}th,td{border:1px solid #ccc;padding:5px}th{background:#e7edf2}</style>')
        handle.write('<h1>Supplementary tables</h1><p>Metrics are descriptive. Region-level scores and pooled scores have different denominators. All runs use fold 0.</p>')
        for label, path in sources:
            handle.write(f"<h2>{label}</h2><table>")
            for i, row in enumerate(read(path)):
                tag = "td" if i else "th"
                handle.write("<tr>" + "".join(f"<{tag}>{html.escape(v)}</{tag}>" for v in row) + "</tr>")
            handle.write("</table>")
        handle.write("</html>")
    print(f"Wrote two main tables and supplementary HTML to {out}")

if __name__ == "__main__":
    main()

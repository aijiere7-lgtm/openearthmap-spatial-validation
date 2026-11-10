# Workflow guide

## Archived result verification

Run scripts/verify_records.py to compare the tracked JSON hashes with the provenance manifest and compare final validation and reference mIoU with experiments/run_metrics.csv. It checks all 75 retained fits, 1,500 epoch entries, the freeze digest and the reported metrics. The original SHA-256 column identifies the workspace source files. The sanitized digest identifies the bytes tracked here. The only field change in each record is replacing the absolute best-checkpoint path with its basename. JSON was reserialized. Nine unsuccessful attempt files and all binary checkpoints were left out.

Run scripts/reproduce_summary.py with --distances for descriptive recomputation from the original score exports and memberships. The executed-design CSV is the inventory of what ran. Although membership files include other folds, only fold 0 was trained for this manuscript. Exact coordinates and sample IDs are archive data and are included in the pending redistribution review.

## Manuscript outputs

Verify scores and summaries first, then use scripts/reproduce_figures.py and scripts/reproduce_tables.py. Each takes an optional --output-dir. The figure script reuses the original submission plotting code and the verified statistics. The table script generates the two main table CSVs and the submission's supplementary HTML.

## Original processing, fitting and reporting code

The original source is preserved to document how the results were produced. Its operational outputs remain separate from the tracked archival inputs. Work in a disposable checkout for any future workflow that writes to data/ or experiments/.

| Stage | Entry points | Required inputs and notes |
|---|---|---|
| Dataset retrieval and inventory | fetch_openearthmap.py, build_inventory.py, audit_official_manifests.py, audit_image_label_pairs.py, region_availability.py | A separately obtained source distribution and its original terms. Keep the imagery/label paths and inventory relative to the corpus root. This archive excludes the source rasters. |
| Coordinates and class composition | build_region_metadata.py, extract_sample_coordinates.py, extract_class_composition.py, spatial_metadata_audit.py | Audited image/label inventory and raster metadata. The tracked reference composition is a derived table. |
| Geographic reference and spatial scale | freeze_external_regions.py, spatial_dependence.py | Region metadata and per-tile class composition. The frozen reference and the archived scale fits are already tracked. Do not overwrite them when verifying the paper. |
| Partition construction and original audit | build_splits.py, verify_split_validity.py | Reference, metadata, scale fits and labels. The published comparison uses the nine original membership files, not freshly sampled folds. |
| Model environment and fitting | fetch_pretrained_weights.py, probe_pretrained.py, run_smoke_test.py, run_experiments.py | Compatible CUDA/PyTorch stack, source data, generated inventory, partition index, pretrained weights and the smoke gate. A smoke test also trains, so it was excluded from the release checks. |
| Original statistics and visualization | analyze_results.py, make_per_class_table.py, make_figures.py | Original run registry, region metrics and result tables. These scripts cover the broader project pipeline. They do not produce the final paper's descriptive summaries. make_per_class_table.py reads historical experiments/runs/ when used in that workflow. |

The historical requirements record Python 3.13 and a CUDA 12.8 PyTorch build. Install the matching CUDA PyTorch wheel using the [official PyTorch distribution guidance](https://pytorch.org/get-started/locally/) when reconstructing that separate environment. Package availability and end-to-end training on another machine have not been validated. The observed hyperparameters are in each record's hyperparameters object. The original configs/default.yaml includes planned designs, seeds, evaluation batch settings and broader statistical options. Actual executed settings follow the run records and scripts, including reporting batch size one and the approximate-size control with one subsample.

The original entry points and workflow documentation make the archival training recipe inspectable. The 75 runs cannot be claimed as independently retrained from this repository. GPU operations, acquisition of raw data and pretrained weights, and inference were outside the release validation.

Historical context is retained in [the original research protocol](ORIGINAL_PROTOCOL.md) and [the original decision log](ORIGINAL_DECISIONS.md). These documents include planned or superseded settings. Use the tracked run records and the main README for the executed manuscript design.

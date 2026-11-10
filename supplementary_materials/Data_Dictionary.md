# Data dictionary

| File | Unit and interpretation |
|---|---|
| experiments/run_metrics.csv | One row per completed run. Protocol is the exact archived identifier. M1 is U-Net/ResNet-34, M2 is DeepLabV3+/ResNet-50 and M3 is SegFormer-B0. The two mIoU values are pixel-pooled within each set then averaged over class IoUs. optimism_bias is the signed difference and absolute_estimation_error is its magnitude. |
| experiments/training_budget_and_checkpoint_audit.csv | Same 75 runs with start time, scheduled steps, selected epoch and batch-eight selection mIoU, batch-one final validation mIoU, difference (final minus selection) and raw maximum epoch. Pixel counters are batch tensor slots, including ignore positions. They are not the valid-pixel class-support denominator. No skipped-AMP-step count was archived. |
| data/splits/*.csv | Tile ID, region, longitude/latitude in geographic degrees and fold assignments. For expanded buffered records, filter eval_fold = 0, then read train, val or excluded_buffer. In compact records fold = 0 is validation, other assigned folds are training and excluded rows are outside both. The files describe assignments beyond the single evaluated fold. |
| Table_S1_Executed_design.csv | One row per executed setting with exact training, validation and excluded tile counts, scheduled step attempts, seed labels and completed runs. |
| per_region_metrics.csv | One archived region summary per run, evaluation set and region. n_tiles counts tiles in that run's set. 900 geographic reference records span all 12 regions in all 75 runs. |
| per_class_metrics.csv | One run, evaluation set and class. support_pixels counts labelled pixels in the class. IoU, precision, recall and F1 are archived metric proportions. |
| Table_S3_Seed_dispersion.csv | Across-seed descriptive statistics for validation, reference and signed gap. all_models first averages M1–M3 for each seed. sample_sd uses the n−1 denominator. |
| Table_S4_Region_dispersion.csv, region_mean_scores.csv | First average each region's reference mIoU across the protocol's runs. Report each region and then the range, inclusive quartiles, median and IQR across the 12 region means. These unweighted regional summaries are distinct from the pixel-pooled reference mIoU. |
| Table_S5_Class_scores.csv | Mean per-run class IoU by main protocol and evaluation set with validation minus reference difference. |
| reference_region_composition.csv | Reference region, continent, nominal latitude, usable tile count, labelled pixel count, dominant class and shares. Indices 0–7 denote bareland, rangeland, developed space, road, tree, water, agriculture and building. |
| data/metadata/external_regions.json | Reference and development region definitions, UTC freeze, selection method, and canonical SHA-256 digest. Digest excludes the top-level content_sha256 field. |
| fold0_nearest_distances.csv | One row per validation tile and setting, giving its nearest training centroid distance in km using a spherical Earth with radius 6,371.0088 km. |
| fold0_geographic_diagnostics.json | Archived core distance and classwise Moran summaries. |
| footprint_overlap.csv, footprint_overlap_pairs.csv | Derived fold-0 counts and cross-set pairs for positive-area ground footprint intersections. A positive intersection does not establish pixel identity. |
| spatial_scale_fit.json, spatial_dependence_curve_within_region.csv, spatial_autocorrelation.csv | Archived composition range fits, lag-bin estimates and spatial diagnostics. Fitted composition scales do not measure spatial correlation in model errors. Archived tests in the original autocorrelation output are not used as inferential tests of protocol differences. |
| summary.json | Full-precision protocol means, signed and absolute gaps, mean model scores and ranks, first-choice reference shortfall, shared-seed calculations, seed-level ranks and scale-scan results. |

Runs are identified by run_id. For the regional and class tables, join run_id to experiments/run_metrics.csv to obtain the protocol. Some field names in archived files use “external” for the geographic reference. Timestamps are UTC.

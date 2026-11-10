# OpenEarthMap spatial validation

## Project overview

This repository accompanies the research manuscript “Spatial validation changes accuracy estimates and model selection on OpenEarthMap”. It preserves the archived experiments and implements a lightweight workflow for checking their results and regenerating the paper's figures and tables.

## Research objectives

We compare how validation partitions affect estimates of land-cover segmentation accuracy and the ranking of competing models on a common geographic reference. We report the signed validation-minus-reference mIoU difference, its absolute magnitude and agreement on model rankings.

## Dataset description

[OpenEarthMap](https://doi.org/10.1109/WACV56688.2023.00619) describes 5,000 high-resolution images across 97 regions and 44 countries, with eight land-cover classes. The study's retrieved distribution contained 3,838 images and 3,500 label rasters, with 2,687 usable pairs. The analysis used 2,214 development tiles from 62 regions and a region-disjoint reference of 470 tiles from 12 regions. Three Kinshasa tiles fell below the region-size threshold. The reference was fixed before training using class coverage and continent-stratified geographic separation. It is specific to this study.

## Experimental design

The archive contains 75 completed runs of 20 epochs on fold 0. The main protocols account for 54 runs, the scale endpoints for 12 and the approximate-size random control for nine. The four main comparisons share seed labels 42, 1337 and 2024. Each main protocol uses one realized partition.

| Protocol | Validation rule | Training / validation / excluded | Completed runs |
|---|---|---|---:|
| S0 | Random tile assignment | 1,771 / 443 / 0 | 15 |
| S2 | Spatial blocks at 1r | 1,701 / 513 / 0 | 15 |
| S3 | Spatial blocks with centroid buffer at 1r | 1,310 / 513 / 391 | 9 |
| S4 | Aachen region holdout | 2,165 / 49 / 0 | 15 |

The empirical composition scale r is 12.137 km. The scale scan compares 0.5r, 1r and 2r at seed 42. Under S3 the nominal block width and great-circle buffer threshold vary together. The control used one subsample of 1,324 random training tiles and the S0 validation set.

The three model implementations are U-Net with ResNet-34 (M1), DeepLabV3+ with ResNet-50 (M2) and SegFormer-B0 (M3). Checkpoint selection used validation batches of eight and a minimum improvement threshold of 0.0001. Reporting used batches of one, with class IoU calculated from pooled pixel confusion counts. The largest observed difference between the two validation paths was about 0.003015 mIoU.

## Repository structure

| Path | Contents |
|---|---|
| src/ | Original dataset, model, training, metrics, spatial and statistical modules |
| scripts/ | Original pipeline entry points and standalone verification and figure/table regeneration commands |
| configs/ | Original default config and archived package inventory |
| experiments/ | 75 sanitized JSON run records, analysis score table, checkpoint audit and SHA-256 provenance |
| data/metadata/ | Frozen reference definition, region composition and spatial fit inputs |
| data/splits/ | Membership files for nine executed settings and their index |
| figures/ | The three manuscript figures in vector PDF and 300 DPI TIFF |
| docs/ | Workflow boundaries, version notes, quality report and publication status |
| tests/ | Original scientific unit checks and the self-contained evaluation smoke test |
| supplementary_materials/ | Analysis summaries, descriptive data tables and the data dictionary |

## Installation and dependencies

The archive checks and table exporter use the Python standard library. Figure regeneration and the portable metric test use the small environment in requirements.txt. Python 3.12 was used for verification. From the repository root, create and activate a Python environment, then install the dependencies.

~~~sh
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python -m pip check
~~~

The optional environment.yml describes the analysis environment. requirements-training.txt and configs/archived-pip-freeze.txt record historical training dependencies. See [the workflow guide](docs/WORKFLOWS.md) for the data and CUDA requirements. Historical pins were not re-resolved or exercised as a training environment in this release.

## Data preparation

The archive checks, tables and figures use only tracked files. A separate training reproduction would require the study's source imagery/labels, compatible pretrained weights and a matching training environment. Retrieve the original distribution from [OpenEarthMap](https://www.open-earth-map.org/) under the source-specific terms. Preserve its region/image/label layout under data/raw/ in a disposable checkout. The development sample was constructed from the audited distribution without adding xBD RGB images. See the [workflow guide](docs/WORKFLOWS.md) for inventory, geospatial metadata and partition provenance.

## Verify existing results and reproduce descriptive analyses

~~~sh
python scripts/verify_records.py
python scripts/reproduce_summary.py --distances
python -m unittest discover -s tests -p test_portable.py
~~~

The first command checks record digests, epoch histories, frozen reference and reported class-score arithmetic. The second recalculates protocol and shared-seed scores and rankings, scale points, seed dispersion, regional and class summaries, counts and centroid distances from the archived records. These commands do not invoke a training or inference entry point.

## Reproduce figures and tables

~~~sh
python scripts/reproduce_figures.py
python scripts/reproduce_tables.py
~~~

Outputs go to build/figures/ and build/tables/. The rendering scripts use the summary and supporting tables verified above. The source PDF/TIFF files in figures/ represent the manuscript versions. Historical scripts such as analyze_results.py and make_figures.py belong to the original workflow. The scripts named reproduce_* are the manuscript reproduction path.

## Experimental limitations

One evaluated fold per protocol and one Aachen holdout limit geographic inference. Seed labels describe run coverage, with model construction preceding fit-time seeding. Equal epoch counts give different scheduled step counts across training sizes. The archive lacks per-step AMP skip counts and a complete set of compatible regional confusion matrices. The empirical spatial scale describes land-cover composition. The 12 reference regions provide a defined comparison and were selected without probability sampling. The scale endpoints use one seed, and the training-size control uses one subsample.

## Citation

See [CITATION.cff](CITATION.cff) for the local v1.0.0 software citation metadata. Fengjun Jiang is the software creator. Cite OpenEarthMap separately as the source dataset. The GitHub link, Zenodo software version and concept DOIs, and any separate data DOI will be added after actual publication and verification.

## License and data availability

The original software and its associated documentation are available under the [MIT License](LICENSE). The MIT grant covers the software and documentation. The accompanying derived research records are shared for reproducibility and remain subject to any applicable source-data terms. The tracked records contain metrics, memberships, tile coordinates and derived statistics. Raw imagery and label masks, model weights, prediction rasters and original workstation paths are excluded. The dataset providers describe the relevant source licences in the [official project repository](https://github.com/bao18/open_earth_map#license). Repository and DOI publication are pending.

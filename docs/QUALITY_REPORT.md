# Repository quality report

Scope is the local v1.0.0 candidate. Validation used Python 3.12 on Windows. No training, inference, data acquisition or experimental rerun was performed.

| Check | Result | Evidence |
|---|---|---|
| Syntax | Pass | python -m compileall -q src scripts tests compiled every included Python file. |
| Lightweight dependencies | Pass in the current runtime | The four requirements.txt versions match installed distributions and pip check found no broken requirements. An empty venv was created. Installing from scratch using network indexes and the optional Conda route were not exercised. |
| Archived data reading and arithmetic | Pass | python scripts/verify_records.py checked 75 JSON hashes, 1,500 epochs, checkpoint selection, 150 score values and the frozen 12-region, 470-tile definition. |
| Descriptive recomputation and membership | Pass | python scripts/reproduce_summary.py --distances checked nine executed designs, exact protocol means and rankings, shared seed comparisons, the scale scan, seed/region/class summaries and nearest distances. |
| Portable tests | Pass | python -m unittest discover -s tests -p test_portable.py ran two tests. One exercises the original confusion-matrix implementation against hand-calculated pooled mIoU and the other checks the run archive. |
| Relative corpus paths | Pass | python tests/test_root_resolution.py passed 26 checks using a writable temporary directory. Includes nested layouts, invalid roots and relative paths. |
| Figure and table regeneration | Pass | Six generated PDF/TIFF files were byte-identical to the RAET figure files. Both main CSV tables matched every corresponding Word table cell. The supplementary HTML matched the RAET delivery byte for byte. |
| Archive contents | Pass | The staged tree and committed version are scanned for known token and key patterns, absolute workstation paths, author contact fields, raw-data locations, binaries other than the six output figures and other prohibited files. Exact derived tile locations are documented for release review. |
| Citation metadata | Basic fields checked | CITATION.cff includes software type, title, version, MIT licence and the sole confirmed software creator. Zenodo ingested the citation metadata and registered the version DOI with Fengjun Jiang as sole creator. |
| Historical dependency mapping | Declared | External imports were inventoried against requirements-training.txt, with the figure-only PDF dependencies in requirements.txt. Resolution of the archived training stack was outside the lightweight environment test. |

## Historical checks with incomplete runtime coverage

- tests/test_metrics.py passed the hand-checked confusion matrix, score arithmetic and early bootstrap checks before requiring SciPy for BCa.
- tests/test_split_logic.py stopped when the legacy diagnostics imported ImageHash.
- tests/test_spatial_dependence.py requires libpysal and related spatial statistics packages.
- tests/test_statistics.py stopped at the GeoPandas dependency check.
- tests/test_label_encoding.py imports the training dataset and requires PyTorch.
- Installing the missing historical packages in this environment did not resolve the requested wheels. Historical processing and plotting entry points depend on those packages and original/generated inputs. The historical training scripts were syntax checked only. The separate lightweight reproduction workflow completed.

See [publication status](PUBLICATION_STATUS.md) for GitHub and Zenodo verification steps.

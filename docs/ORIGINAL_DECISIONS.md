# DECISIONS

Methodological decision history. Append-only. Every entry states the decision, the evidence behind it, the alternatives rejected, and what it affects.

**Rule**: any change to `research_protocol.md` after its freeze (2026-09-17) is recorded here with its date and the analyses it invalidates. Changes made after results were observed are labelled **POST-HOC**.

---

## D-001 — Use OpenEarthMap as the primary dataset

**Date**: 2026-09-17
**Status**: adopted

**Decision**: OpenEarthMap is the primary and initially the only dataset.

**Evidence** (verified from primary sources; see `data/metadata/dataset_source_verification.md`):
- 5,000 images at 1024×1024 px, 8 classes, 97 nominal regions across 44 countries and 6 continents — the geographic metadata is what makes a genuine external geographic hold-out constructible.
- The official split is **random and within-region** (6:1:3), which makes the dataset directly relevant to the question rather than a contrived testbed.
- Sub-metre GSD (0.25–0.5 m), manually annotated, with published class statistics.

**Alternatives rejected**:
- **LoveDA** — deferred to P2. Its two-domain (urban/rural) design imposes a coarse, pre-defined geographic contrast that would confound the block-size sensitivity analysis (RQ2), which needs a continuous separation scale. It remains valuable as an independent replication.
- **DeepGlobe** — larger tiles (2448×2448) and a different class taxonomy; no advantage for the geographic-metadata requirement.

**Affects**: §4 of the protocol; the entire experimental design.

---

## D-002 — The official 1500-image test split is not used as the external geographic test

**Date**: 2026-09-17
**Status**: adopted

**Decision**: The official test split is treated as an in-distribution hold-out only. The external geographic test is constructed independently from whole held-out regions.

**Evidence**: The paper states the images "from each region were randomly divided into training, validation, and test sets with a ratio of 6:1:3". A within-region random split places training and test images from the same region — and potentially the same acquisition block — on both sides. It is therefore not geographically independent.

**Consequence**: Using the official test split as the external benchmark would have manufactured the study's conclusion. It is used only as (a) a benchmark-reproduction reference and (b) the basis for the S0 random-split baseline.

**Affects**: §4.1, §6 of the protocol; every result labelled "external".

---

## D-003 — Accept a scope reduction: 28 xBD regions excluded

**Date**: 2026-09-17
**Status**: adopted, with disclosed limitation

**Decision**: Proceed without the xBD-sourced regions rather than block the project.

**Evidence**: The xBD RGB imagery is not redistributed inside OpenEarthMap; the official instruction is to download it from `xview2.org`. That host is (a) network-blocked from this environment (measured) and (b) registration-gated upstream. Neither condition is satisfiable autonomously.

**Alternatives rejected**:
- Substituting another source for the xBD RGB imagery — rejected outright: it would not be the same imagery, and the labels would no longer correspond to their source. Fabricating a substitute is not an option.
- Blocking on user action — deferred, not rejected. If the user can supply xBD RGB imagery, the region pool expands and the analysis can be re-run; the code is written to take the region pool empirically so this requires no redesign.

**Consequence**: the region pool is at most 69, and the true number is determined empirically at QC. Geographic coverage is unevenly reduced (much of Latin America, South/Southeast Asia and North America). This is a pre-registered limitation (protocol §18.2).

**Affects**: §4.2, §6.2 of the protocol; the external-region selection pool.

---

## D-004 — Obtain the dataset from a mirror, with mandatory hash verification

**Date**: 2026-09-17
**Status**: adopted, with disclosed limitation

**Decision**: Retrieve `OpenEarthMap.tar.gz` from the ModelScope mirror, verifying SHA-256 against the mirror's published digest `947bdb12...88f3cf`.

**Evidence**: The canonical Zenodo record is unreachable — the proxy returns `502 Bad Gateway` on CONNECT. ModelScope publishes the same upstream dataset, the same licence string, the same upstream project and paper references, and a SHA-256.

**Residual risk, disclosed**: hash verification establishes transport integrity of the mirror artefact. It does **not** independently establish byte-equivalence with the Zenodo `.zip` — different hash algorithm, different container format. A file-level comparison against a Zenodo download remains outstanding and is recorded as such in the Data Availability section.

**Rejected alternative**: treating the mirror as equivalent without verification. The QC phase's structural audit (image/mask counts, dimensions, correspondence, label range) provides a second, independent check on the data actually being OpenEarthMap.

**Affects**: §6.3 of the verification document; Data Availability statement.

---

## D-005 — Device-agnostic pipeline; GPU availability left unverified rather than assumed

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Write all training code device-agnostic, resolving the device from config with an explicit override. Do not assume the GPU is usable; do not assume it is unusable either.

**Evidence**: The RTX 4060 is visible to WMI but `nvidia-smi` fails with `Failed to initialize NVML: Unknown Error`, and `nvcc` is absent. This pattern indicates the driver device nodes are not exposed to the process, which is characteristic of a restricted execution context. It does **not** by itself prove PyTorch cannot use CUDA.

**Action**: a CUDA-enabled PyTorch wheel is being downloaded and `torch.cuda.is_available()` will be tested explicitly. The result is recorded in `environment_report.md` and in D-006 below.

**Consequence**: no result may be described as GPU-computed until that test passes. The experimental grid is sized for GPU compute, and the P0/P1/P2 priority ladder (protocol §15) ensures that a compute-truncated execution still yields a coherent study rather than a partial grid.

**Affects**: §2, §6 of the environment report; the execution order of the experiment phases.

---

## D-006 — RAM constraint drives a streaming data pipeline

**Date**: 2026-09-17
**Status**: adopted

**Decision**: The data pipeline streams from disk. No phase may assume the dataset fits in memory.

**Evidence**: 15.82 GB system RAM against 5,000 images at 1024×1024×3 (≈ 15.7 GB as uint8 for imagery alone, before masks or augmentation, before any model or optimiser state). Holding the corpus in memory is not feasible.

**Consequence**: QC aggregates are computed incrementally; features used for the spatial-dependence analysis are derived and cached as compact vectors (e.g. per-image moments and perceptual hashes) rather than retained as imagery; training uses on-the-fly crop sampling from JPEG/PNG on disk.

**Affects**: the design of `src/data/` and `src/spatial/`.

---

## D-007 — Replace the shell curl download loop with a Python downloader

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Perform the dataset transfer in Python (`scripts/fetch_openearthmap.py`) rather than a bash `curl -C -` retry loop.

**Evidence**: The shell version failed reproducibly with `curl: (23) client returned ERROR on write of 15116 bytes` at byte 0 on every attempt, while the identical curl invocation run directly (without the `| tail -2 | tee` pipeline) succeeded. Diagnosing Git Bash pipe semantics was judged lower-value than removing the pipeline. The Python version additionally gives explicit control over resume semantics — in particular the case where a server ignores a `Range` header, which would otherwise silently append a full body onto a partial file and corrupt the archive.

**Affects**: `scripts/fetch_openearthmap.py`; the reproducibility instructions in the README.

---

## D-008 — Two phases are explicitly CPU-feasible and therefore not blocked by the GPU question

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Proceed with data QC, spatial-dependence analysis, external-region freezing, split construction and leakage diagnostics while GPU availability is unresolved.

**Evidence**: These phases are IO-bound and statistic-bound, not compute-bound. `libpysal` and `esda` are already installed, so Moran's I and spatial-weights computation are available immediately. None of these phases touches model training, so none can be invalidated by the GPU outcome.

**Affects**: execution order; the sequencing in `PROGRESS.md`.

---

## D-009 — Coordinate tier will be determined empirically, not assumed

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Determine the coordinate tier (T1 per-image / T2 region-level / T3 grouping-only) by inspecting the retrieved archive, and record the determination with its evidence.

**Evidence**: The paper states the source imagery "have geocoordinate information" and that GSD was standardised, but it does **not** document whether per-image coordinates are shipped in the distributed archive. Assuming T1 would risk fabricating a distance-resolution capability the data may not have.

**Binding consequence**: if only T3 is achievable, the study will not report a numeric dependence range `r`, will not claim a distance–similarity relationship, and will not assert that any block size is "most appropriate" (protocol §5).

**Affects**: §5, §7 of the protocol; the block-size definitions in §8.

---

## D-010 — The mirror archive is doubly nested; extraction is a two-stage step

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Treat archive extraction as two sequential stages and verify each, rather than assuming a single `tar -xzf` yields the imagery.

**Evidence**: Listing the partially downloaded archive directly (`tar -tzf` on the `.part` file, which lists members sequentially until truncation) returned exactly two entries:
```
OpenEarthMap/
OpenEarthMap/OpenEarthMap.zip
```
So `OpenEarthMap.tar.gz` wraps a directory containing a single `OpenEarthMap.zip`. The tar layer provides no compression benefit (a zip inside gzip is incompressible), which is why the `.tar.gz` and the published `.zip` are both ~9.1 GB. A single extraction would leave a zip on disk and no imagery, and the failure would look like a layout-discovery bug rather than a packaging detail.

**Affects**: `scripts/fetch_openearthmap.py`; the data-acquisition procedure; disk-space planning (two concurrent ~9 GB copies during extraction).

---

## D-011 — Pretraining parity is restored via a mirror, not abandoned

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Keep the all-pretrained (ImageNet-1k) initialisation policy for all three models, obtaining M3's weights through the mirror `hf-mirror.com`, rather than downgrading to an all-scratch policy.

**Evidence**: `scripts/probe_pretrained.py` found `pretraining_uniform = false`: M1 (ResNet-34) and M2 (ResNet-50) obtained ImageNet weights from `download.pytorch.org`, while M3 (SegFormer-B0, `nvidia/mit-b0`) failed because `huggingface.co` is unreachable from this host. Connectivity testing then showed `hf-mirror.com` returns HTTP 200 while `huggingface.co` returns nothing, and the mirror serves `nvidia/mit-b0` with `config.json` (70,043 bytes) and `pytorch_model.bin` (14,380,029 bytes).

The downloaded config was verified to be genuinely SegFormer-B0 by architecture rather than by trusting the URL: `model_type == 'segformer'`, `hidden_sizes == [32, 64, 160, 256]`, `depths == [2, 2, 2, 2]`. The 70 KB size is explained by an embedded 1000-entry ImageNet `id2label` map.

**Alternatives rejected**:
- *All-scratch*: consistent, but discards ImageNet initialisation for all three models, which on a 5,000-image corpus would degrade absolute accuracy and make convergence within the compute budget unlikely. The parity problem is solvable, so accepting the cost was not justified.
- *Mixed*: rejected outright. Giving M3 random initialisation while M1/M2 receive ImageNet weights would convert a pretraining artefact into an apparent architecture effect, and RQ3 would be uninterpretable.

**Disclosure required in the manuscript**: the three encoders share ImageNet-1k pretraining *in kind*, but not in provenance — M1/M2 use torchvision ResNet weights and M3 uses `nvidia/mit-b0`. Furthermore, for all three the pretrained component is the *encoder* while the task head is randomly initialised. This is inherent to comparing heterogeneous architectures and is recorded as a limitation rather than presented as a controlled variable.

**Affects**: §10 of the protocol; the RQ3 architecture comparison; `scripts/fetch_pretrained_weights.py`; `src/models/builders.py`.

---

## D-012 — All initialisations are pinned locally and verified offline

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Pin every pretrained weight file to `checkpoints/pretrained/`, mirror the ResNet encoders into a project-local `TORCH_HOME`, and require training to run with no network access.

**Evidence**: Integrity is checked rather than assumed. The torchvision weight filenames embed the first 8 hex digits of their own SHA-256, and both verified exactly:
- `resnet34-333f7ec4.pth` — 87,306,240 bytes, `sha256 333f7ec4c6338da2…` ✓
- `resnet50-19c8e357.pth` — 102,502,400 bytes, `sha256 19c8e3572231adff…` ✓

Byte counts were deliberately **not** hard-coded as pass/fail thresholds: the two sizes I had estimated from progress-bar output were both wrong (off by 13,206 and 59 bytes respectively). Asserting an unverified size would have produced a false failure loop, so the embedded SHA-256 prefix is the authoritative gate instead.

A separate check confirms the weights were genuinely applied rather than silently ignored: for each model, the encoder's parameter signature is compared against a freshly initialised model of the same architecture, and all three differ (`differs_from_scratch=True`).

Finally, `scripts/verify_model_init.py` was run with all proxy variables pointed at an unreachable address (`127.0.0.1:9`) so that any network attempt fails immediately. All three models still built with pretrained weights, proving the load path is genuinely local.

**Note on a benign warning**: `segmentation_models_pytorch` emits *"Error loading resnet50 `imagenet` weights from Hugging Face Hub, trying loading from original url..."*. This is expected — it probes HF first, HF is blocked, and the fallback resolves to the pinned `TORCH_HOME` copy. The offline run confirms no network dependency remains.

**Affects**: `scripts/fetch_pretrained_weights.py`, `scripts/verify_model_init.py`, `src/models/builders.py`; the reproducibility guarantee in protocol §17.

---

## D-013 — GPU compute IS available; D-005 is resolved

**Date**: 2026-09-17
**Status**: adopted — **supersedes D-005**

**Decision**: Treat GPU compute as available, and enable it in the main environment by upgrading to CUDA builds of the *same* versions.

**Evidence**: D-005 recorded `nvidia-smi` failing with `Failed to initialize NVML: Unknown Error` and therefore left GPU availability unverified. That failure was a tool-invocation/context problem, **not** a hardware or driver problem.

The decisive test was constructed by avoiding the two ways this could have been answered wrongly:

1. *A CPU-only torch build can never answer the question* — `torch.cuda.is_available()` is hardwired False, so the main environment could not settle it.
2. *Installing CUDA torch over the working environment to test would risk destroying it.*

So a **separate throwaway venv** (`envs/cuda_probe`) was created holding only the CUDA build. Result:

```
torch_version:      2.11.0+cu128
compiled_with_cuda: 12.8
cuda_available:     True
device_count:       1
device_name:        NVIDIA GeForce RTX 4060
capability:         (8, 9)              -> Ada Lovelace, sm_89
GPU_MATMUL_RESULT:  29347.80859375      -> a real GPU matmul succeeded
VERDICT: GPU_COMPUTE_AVAILABLE
```

Driver capability was then queried directly through the driver API rather than inferred: `cuDriverGetVersion` returns `13040` (CUDA 13.4), so CUDA 13.0 builds are supported.

**Version-match plan (chosen to avoid any cascade)**: the main environment has torch 2.14.0 and torchvision 0.29.0. Exact CUDA counterparts exist:
- `torch-2.14.0+cu130-cp313-cp313-win_amd64.whl` — 1,990,601,610 bytes
- `torchvision-0.29.0+cu130-cp313-cp313-win_amd64.whl` — 6,406,089 bytes

Installing these changes only the build suffix (`+cpu` → `+cu130`), so no version moves and nothing else in the stack (transformers 5.17.0, smp 0.5.0, timm 1.0.29, rasterio, libpysal, esda) is disturbed. The alternative — the already-downloaded `torch 2.11.0+cu128` — was **rejected**: it would downgrade torch by three minor versions and force a matching torchvision downgrade, cascading into smp and transformers.

**Scheduling decision**: the 2.0 GB wheel download is deliberately **deferred until the dataset transfer completes**. Both share the same limited link (measured ~1.6 MB/s), the dataset is on the critical path for all subsequent analysis, and the torch wheel is not needed until training — roughly 1.5 hours of QC work after the dataset lands. Downloading now would delay the critical path for no benefit.

**Binding consequence**: no result may be described as GPU-computed until the **main environment** is upgraded and a GPU probe passes **with the project's actual stack** (smp, transformers, rasterio all still importing and running). A pass in the isolated probe environment establishes that the hardware and driver work; it does not establish that the project environment is GPU-capable.

**Affects**: D-005 (superseded); the feasibility of the full grid (3 models × 5 seeds × 6 designs ≈ 90 runs), which was previously judged infeasible on CPU; the Compute section of the manuscript.

---

## D-014 — The experiment registry names the strategy column `strategy`, not `design`
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the column that identifies the validation strategy (S0, S2, S3_b1.0r, …) is named `strategy` in `experiments_registry.csv`, `per_region_metrics.csv`, and everywhere downstream. The CLI flag is `--strategies` with `--designs` retained as an alias.

**Evidence**: the pre-registered mixed-effects model is
`Metric ~ Strategy + Model + Strategy x Model + (1|Region)`.
An end-to-end validation run against synthetic data (`validation/analysis_pipeline/`) failed with
`PatsyError: Error evaluating factor: NameError: name 'strategy' is not defined`
because the registry called the column `design`. The failure was silent in the sense that mattered: `fit_mixed_effects` caught it, reported `fallback_recommended: True`, and the pipeline completed with `[STATUS] PASS` while the pre-registered model had not actually been fitted.

**Alternatives rejected**: adding a `strategy` alias column alongside `design` — two names for one concept is what produced the mismatch in the first place, and an alias leaves the ambiguity in place for the next reader. Renaming was free because the registry was still empty.

**Affects**: `scripts/run_experiments.py`, `src/statistics/reporting.py`, `scripts/analyze_results.py`, `scripts/make_synthetic_registry.py`, `scripts/test_statistics.py`. No migration was required (0 rows existed).

---

## D-015 — A region-level paired optimism bias is not identifiable, and this is a design guarantee
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the pre-specified primary contrast (S0 vs EXT on mIoU) is tested **at run level**, paired within (model, seed). No region-level paired version is reported as a result. `primary_contrast_s0_vs_ext` attempts one and, when the region sets are disjoint, returns `region_level_unavailable_reason` explaining why. Likewise `mixed_effects.json` records `ob_region_fit.fitted = false` with the reason, rather than omitting the analysis silently.

**Evidence**: the optimism bias for a run is `validation − external`. The validation estimate is computed on **development** regions; the external estimate on **external** regions. Those sets are **disjoint by construction** — the freeze excludes external regions from every partition (protocol §6.3). No region therefore carries both quantities, so a region random intercept for the bias cannot be estimated. The synthetic validation run, once rebuilt to reproduce the real disjoint structure (24 development regions, 10 external regions, 0 in common), confirmed this: the pivot produced zero paired rows, and the earlier attempt to fit an OB random intercept failed with `LinAlgError: Singular matrix` — the correct symptom, since the region effect cancels identically in the difference.

**Why this is recorded as a decision and not a bug**: the earlier implementation would have produced a numeric region-level contrast on any dataset where some region happened to appear in both sets. That number would have looked like a result and would have meant something other than its name. Making the absence explicit is the substantive change.

**Alternatives rejected**: reporting the region-level version as "not yet computed" — it is not computable in principle here, and the distinction matters for how the manuscript states its limitations. Fitting the OB model on a partial region overlap — that would answer a different question from the one the model's name asserts.

**Affects**: `research_protocol.md` §13.2/§13.4 interpretation; the manuscript's Statistics section and its limitations; `src/statistics/reporting.py`, `scripts/analyze_results.py`.

---

## D-016 — The mixed-effects model is fitted once per response, never on a pooled frame
**Date**: 2026-09-17
**Status**: adopted
**Decision**: `Metric ~ Strategy + Model + Strategy x Model + (1|Region)` is fitted separately for (a) the **external** metric, on external regions — the primary fit, since the external test is the truth being estimated — and (b) the **validation** metric, on development regions, reported for contrast. The two fits are never pooled.

**Evidence**: `per_region_metrics.csv` contains a validation row and an external row for every (strategy, model, seed, region). Pooling them into one response fits a mixture of two different quantities; the strategy coefficients then describe neither, and the sign of a coefficient becomes uninterpretable. Detected during end-to-end validation: the first implementation fitted on the pooled frame and returned coefficients that were a weighted average of the two effects.

**Verification**: on synthetic data with a planted optimism of S0 = +0.090, S2 = +0.020, S4 = +0.050 and a planted region intercept, the corrected fits recovered the signs and magnitudes — external metric: S2 = +0.0287, S4 = +0.0111 (planted +0.035, +0.025 relative to S0); validation metric: S2 = −0.0289, S4 = −0.0284; ICC = 0.767 and 0.591 against a planted ICC of ~0.83.

**Affects**: `scripts/analyze_results.py`; the manuscript's Results §4.4 and §4.5.

---

## D-017 — The statistics package is `src/statanalysis/`, because `src/statistics/` shadowed the standard library
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the statistical analysis package lives at `src/statanalysis/`. No module or package placed directly under `src/` may share a name with a standard-library module or with a commonly top-level-imported third-party package. `scripts/test_statistics.py` now enforces this against 302 names, including the concrete `geopandas` import.

**Evidence**: `src/` is prepended to `sys.path` by every script. A package at `src/statistics/` therefore shadowed the standard-library `statistics` module for the entire process. `geopandas.explore` begins with `from statistics import mean`; it received the local package instead, raising
`ImportError: cannot import name 'mean' from 'statistics' (…\src\statistics\__init__.py)`.
That broke `geopandas`, and through it `libpysal`, and through that the whole spatial-dependence analysis. `scripts/test_spatial_dependence.py`, which had passed earlier in the session, began failing with a traceback pointing at `libpysal` — far from the actual cause.

**Why this is worth a decision record**: the defect was introduced by adding a new package, was invisible in the new package's own tests (they passed), and surfaced only through an unrelated third-party import chain. Nothing about the name `statistics` looks wrong in isolation. The only reliable defence is an automated check, which is now in place.

**Alternatives rejected**: `src/stats/` — no stdlib collision, but a short common name is a larger target for third-party collisions, and `statanalysis` states what the module does. Adding `sys.path` entries in a different order — the shadowing would recur the moment any script imported the package before a victim library.

**Affects**: `src/statanalysis/`; `scripts/analyze_results.py`; `scripts/test_statistics.py`; and any future module added under `src/`.

---

## D-018 — Two coordinate standards, encoded separately, because the tolerance depends on the scale
**Date**: 2026-09-17
**Status**: adopted
**Decision**: coordinate provenance is judged against **two different standards**, defined in `common.py` and never derived from one another:

- `COORD_SOURCES_OK_FOR_REGION_DISPERSION` — for selecting geographically spread regions for the external test. Accepts gazetteer centroids.
- `COORD_SOURCES_OK_FOR_RANGE` — for estimating the dependence range `r`. Accepts only coordinates derived from the imagery's own georeferencing.

Separately, `select_external_regions` no longer drops coordinate-less regions silently. The counts form a chain that must balance (`after_class = absent + unacceptable_source + usable`), the invariant is asserted rather than assumed, the excluded regions are listed by name, and an all-coordinate-less input produces an error that names the **coordinate** problem rather than blaming the class-presence filter.

**Evidence for the two standards.** The quantities are two orders of magnitude apart. Region selection operates at inter-region distances of 10^3–10^4 km, where a gazetteer centroid's positional error of 10^1–10^2 km cannot change which region is furthest from which. The dependence range is 1–10 km, where the same error is the same size as, or larger than, the quantity being estimated — a variogram built on those coordinates would measure the positional error. Applying the stricter standard to region selection would block a legitimate analysis; applying the looser one to `r` would license a meaningless number. Both failures are real, and they fail in opposite directions, so one set cannot be derived from the other.

**Evidence for the explicitness fix.** The previous implementation called `dropna(subset=["lon", "lat"])`. When every region lacked coordinates it returned `"no eligible regions after class-presence filter"` — naming a cause that was not the cause. More seriously, when only *some* regions lacked coordinates it excluded them from the candidate pool with nothing recorded, so the selection would have been made over a reduced pool that neither the freeze record nor the manuscript disclosed. Verified across four cases (all present / partial / none / class-filter genuinely exhausted): the four now produce distinguishable, correct outcomes, and 14 assertions in `test_split_logic.py` guard them.

**Affects**: `src/common.py`; `src/splits/strategies.py`; `scripts/freeze_external_regions.py`; the freeze record and `external_regions.md`; the manuscript's §3.4 and its limitations.

---

## D-019 — The fetcher must be idempotent, and a script must never be edited while it is running
**Date**: 2026-09-17
**Status**: adopted
**Decision**: `fetch_openearthmap.py` verifies an existing archive (size **and** SHA-256) before deciding to transfer, and skips the transfer when it already holds a complete, verified file. `--force-download` overrides this; `--deep-verify` opts into the slow full-member read of the nested archive, which is now **off** by default.

**Evidence — two distinct failures, one of which was mine.**

1. **The script was not idempotent.** `download()` was called unconditionally unless `--verify-only` was passed, so a plain re-run re-acquired the archive. The README described the transfer as "resumable and idempotent"; only the first was true. On a 9 GB file that distinction costs a full re-transfer.

2. **The running process was executing stale code.** The transfer had been started *before* the two-stage extraction fix was written. Python had already loaded the old module, so the completed run performed the old single-stage extraction — leaving a 9.1 GB nested zip and no imagery — while printing `extraction complete` and `=== DONE ===`. The mismatch was detected by grepping the current source for the exact strings the log contained: `"extraction complete"` and `"extracting to"` both returned **0** occurrences, which proves the log came from a different version of the file than the one on disk.

**Consequence and recovery.** A re-run was started to perform stage 2. Because of failure (1) it began re-downloading, and because a stray `nohup` invocation from an earlier attempt had also started a process, **two concurrent downloaders** were running. Both were stopped immediately. The verified archive was **unharmed** — the fetcher writes to `.part` and only replaces the destination after verification — and was re-verified independently: 9,095,072,903 bytes, SHA-256 `947bdb12…88f3cf`, matching the published digest. The 465 MB stale `.part` was removed only after asserting that the authoritative archive was present at the exact expected size.

**Why `--deep-verify` is now off by default.** `ZipFile.testzip()` decompresses every member, doubling the cost of a 9.1 GB extraction. It is redundant here: the outer tar.gz's SHA-256 has been verified against the published digest, and that digest covers the nested zip's bytes, so a matching container implies a byte-identical inner archive. `extractall()` checks each member's CRC as it writes, so corruption raises rather than passing silently.

**Also fixed**: `tarfile.extractall(..., filter="data")`, which rejects absolute paths and traversal outside the destination. Python 3.14 makes this the default; setting it explicitly keeps behaviour identical on 3.12+ and silences the deprecation warning, with a `TypeError` fallback for older interpreters.

**Process rule adopted**: a long-running script is not edited while it is executing. If a fix is needed mid-run, either wait for the process to exit, or stop it deliberately and restart with the new code. Verifying which code a log came from is done by grepping the current source for the log's distinctive strings — a zero count proves the log predates the current file.

**Affects**: `scripts/fetch_openearthmap.py`; `README.md` (the idempotency claim is now true); the reliability of any long-running step in this project.

---

## D-020 — The OpenEarthMap release ships 3,838 images and 3,500 labels, not 5,000 of each
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the analysed corpus is defined as the **intersection of image and label availability**, which for this release is **2,687 tiles across 74 eligible regions**, not the 5,000 tiles across 97 regions that the dataset description implies. This figure — not 5,000 — governs every downstream quantity: sample sizes, the matched-sample-size control, the region pool, the external-test design, and all power statements.

**Evidence.** Measured on this host, not taken from documentation.

- `OpenEarthMap.zip` md5 = `64155d1dc9d3b68536063f79878e1a67`, matching the md5 published on the canonical Zenodo record 7223446 for its 9.1 GB `OpenEarthMap.zip`. **Cross-archive equivalence is therefore proven**, closing the open item recorded in `dataset_source_verification.md` §7 issue 3. The corpus is the official release; the anomaly below is a property of the release, not of the acquisition path.
- The ZIP central directory contains 7,635 entries (7,338 `.tif`, 4 `.txt`, 1 `.csv`), identical to the extracted tree — so the extraction is faithful, not truncated.
- Images on disk: **3,838**. Labels on disk: **3,500**. Both with: **2,687**.

Two independent structural causes, each traced to a verbatim statement in the primary source:

1. **xBD RGB imagery is not distributed.** `readme.txt`, the GitHub README and the Zenodo record all state: *"The RGB images of xBD dataset are not included in the OpenEarthMap dataset."* `xbd_files.csv` has 1,162 rows, and `5000 − 1162 = 3838` exactly. The count of images that are xBD targets is **0**, and the count of xBD tiles that lie in the 22 entirely-xBD regions is **1,162 of 1,162**.
2. **Test labels are withheld by design.** No test tile has a label on disk: **0 of 1,500**. Every one of the 3,500 labels belongs to a train or val tile (`3000 + 500 = 3500`, 0 outside, 0 train/val tiles lacking a label). The Zenodo record and the README both state that test-set performance is evaluated on a CodaLab leaderboard, which is what makes withholding the labels necessary.

All counts close with no residual: images `2303 + 384 + 1151 = 3838`; labels `3000 + 500 + 0 = 3500`; usable `2303 + 384 = 2687`; xBD `697 + 116 + 349 = 1162`.

**Alternatives rejected.**
- *Treat the shortfall as a corrupt or truncated extraction.* Rejected by the central-directory comparison: the ZIP's own manifest lists exactly the same 7,338 `.tif` members that are on disk.
- *Treat it as a mirror artefact.* Rejected by the md5 match against the publisher's digest.
- *Recover the test labels.* Not possible from any public release; they exist behind a submission-based evaluation service. Submitting to it would introduce an irreproducible external dependency and is out of scope.
- *Obtain the xBD RGB imagery from xView2.* Requires registration and acceptance of a separate licence; recorded as an open item for the user, not attempted.

**Affects**: `research_protocol.md` (corpus definition, sample sizes, region pool); `data/metadata/dataset_release_audit.md` (new); `scripts/region_availability.py` (new); the class-composition verification, which can no longer target the published shares without a stated cause; the manuscript's §3.1 and §3.3; the pre-registered limitations.

---

## D-021 — The official test split is unusable as a reference, so the external test must be region-held-out from the usable corpus
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the study's external geographic test is constructed by **holding out entire regions from the 2,687 usable tiles**. The official 1,500-tile test split is not used as a reference external test at any point, because it carries no labels in the release.

**Evidence.** The official split remains central to the study as *evidence of practice*: its design is random and within-region at 6:1:3, verified verbatim from the paper and independently corroborated by the manifests, which contain exactly 3,000 / 500 / 1,500 entries with no tile appearing in more than one manifest and no tile appearing in none. The study audits that practice. Auditing it does not require evaluating on it.

The protocol already specified a region-held-out external test. That specification was written as a methodological preference; it is now a **necessity**, because the alternative no longer exists. The distinction matters for the manuscript: the external test is not "the official test set" and must never be described as such.

**Alternatives rejected.**
- *Use the official test split and obtain labels via CodaLab.* Introduces a non-reproducible external service into the critical path of a study whose entire claim is reproducibility, and would make the frozen-external-test guarantee unenforceable, since the freeze is enforced in code and the submission is not.
- *Report both the official test split and a region-held-out test.* The official split cannot yield a metric without labels; reporting it would mean reporting nothing, or reporting a number not produced by an executed run.
- *Subsample the official test split's non-xBD tiles (1,151 of them have imagery).* They have imagery but no ground truth, so no accuracy can be computed.

**Affects**: `research_protocol.md` §6 (external-test selection, freeze); `scripts/freeze_external_regions.py`; the manuscript's §3.4 and §3.11; the wording of every claim about the official split, which must distinguish its *design* (audited) from its *labels* (unavailable).

---

## D-022 — The label integer is not the class index: raster value = class index + 1
**Date**: 2026-09-17
**Status**: adopted
**Decision**: label rasters are converted to class indices by the explicit mapping

```
raster 0 = "unknown" (residual, -> ignore index)   raster 1 = Bareland
raster 2 = Rangeland   raster 3 = Developed space  raster 4 = Road
raster 5 = Tree        raster 6 = Water            raster 7 = Agriculture land
raster 8 = Building
```

so paper class *k* is stored as integer *k+1*. The study trains and reports on the **eight** land-cover classes, sending raster value 0 to the ignore index. The nine-class variant the official baseline uses is reported as a sensitivity analysis.

**Evidence.** Two independent lines of evidence, both primary.

1. **The authors' own code.** `open_earth_map/utils.py` defines
   ```python
   class_grey_oem = {"unknown": 0, "Bareland": 1, "Grass": 2, "Pavement": 3,
                     "Road": 4, "Tree": 5, "Water": 6, "Cropland": 7,
                     "buildings": 8}
   ```
   and `open_earth_map/dataset.py` sets `n_classes: int = 9` as its **default**, with `ToTensor(classes=np.arange(n_classes))` one-hot encoding all nine integers with no shift and no ignore index. `demo_training.py` and `Demo.ipynb` both pass `N_CLASSES = 9`. So the official baseline trains `unknown` as a ninth class.
   The RGB values in the same dict (`800000`, `00FF24`, `949494`, `FFFFFF`, `226126`, `0045FF`, `4BB549`, `DE1F07`) match the paper's Table 2 hex codes **one-for-one and in the same order**, which is what pins the mapping to the paper's class names rather than merely to an ordering of integers.

2. **The measured pixel shares.** Applying the `k+1` mapping to all 2,687 usable tiles reproduces the paper's published per-class shares to within 2.8 pp for all eight classes: Road 6.79 vs 6.70, Water 3.11 vs 3.30, Bareland 1.76 vs 1.50, Agriculture 12.61 vs 13.70, Tree 18.52 vs 20.20, Building 17.29 vs 15.60, Developed 18.88 vs 16.10, Rangeland 21.04 vs 22.90. Under the naive `integer == class index` reading, "Bareland" would receive the residual category's 0.72% and Rangeland would receive 1.76% against a published 22.9% — off by more than a factor of ten.

**What this prevented.** The code as written had three distinct silent failures, none of which raises an exception:
- `remap_labels` tested `0 <= v < NUM_CLASSES`, so raster value 8 (Building, 17.3% of pixels) was mapped to the ignore index and **deleted from every metric**, while raster value 0 (the residual category) was mapped to class index 0 and **reported as Bareland**.
- `IGNORE_INDEX` was defined locally as 255 in `data/dataset.py` while a second definition elsewhere equated it to the raster's residual value; had the latter been used, **every Bareland pixel** would have become ignore.
- `extract_class_composition.py` used the same `[0, NUM_CLASSES)` test, so the region- and sample-level class-composition tables that drive `spatial_block_folds`' stratification would have been computed from a **mis-assigned** class histogram.

The failure mode is the dangerous kind: the numbers would have been plausible, internally consistent, and wrong. Nothing would have flagged it except the comparison against the paper's published shares, which is why that comparison is a QC output and not a footnote.

**Alternatives rejected.**
- *Follow the official baseline and train 9 classes.* Rejected as the primary configuration: `unknown` is not a land-cover class, the paper's taxonomy and published shares are defined over eight, and including a residual category makes mIoU partly a measure of how well the model predicts "not any of the above". Reported as a sensitivity analysis instead.
- *Keep the naive mapping and document it.* Rejected: documentation does not fix a wrong metric, and the mis-assignment is not recoverable after the fact from partition-level metrics.

**Guards.** `scripts/test_label_encoding.py` (32 assertions) pins the mapping, asserts `IGNORE_INDEX` lies outside `[0, NUM_CLASSES)` and differs from the raster's residual value, checks that raster 8 reaches the Building metric and raster 1 the Bareland metric end-to-end, and checks that ignored pixels never enter the confusion matrix. `build_inventory.py` gates on the label-value set matching the audited encoding rather than `[0, 8)`.

**Affects**: `src/common.py` (encoding constants, `IGNORE_INDEX`); `src/data/dataset.py` (`remap_labels`, `count_unmapped_values`, LUT); `src/data/inventory.py`; `scripts/build_inventory.py` (class statistics); `scripts/extract_class_composition.py`; `scripts/run_smoke_test.py`; `scripts/test_label_encoding.py` (new); the class-composition tables that feed stratification; every per-class metric in the study.

---

## D-023 — The corpus is T1 (per-image georeferencing) and is NOT uniformly sized
**Date**: 2026-09-17
**Status**: adopted
**Decision**: the study operates at coordinate tier **T1**. Per-tile geographic footprints are taken from each raster's own CRS and affine transform, transformed to a common frame before any distance is computed, and the spatial-dependence range `r` is estimated in **metres** from those footprints. Tile size is treated as heterogeneous and is handled by cropping/padding rather than by assuming a uniform grid.

**Evidence — coordinate tier.** 60 images sampled at random across the corpus (seed 0): **60 of 60** carry both a CRS and a non-identity affine transform; **0** lack georeferencing. The transforms are metric and geographic in different tiles, and the CRS varies widely — `EPSG:25832`, `EPSG:4326`, `EPSG:3857`, `EPSG:32737`, `EPSG:32717`, `EPSG:32646`, `EPSG:32740`, `EPSG:32718`, `EPSG:26916`, `EPSG:31256`, `EPSG:32653`, `EPSG:32630`, plus a custom Transverse Mercator definition. Ground sampling distances measured from the transforms are 0.25, 0.3 and 0.5 m, matching the documented 0.25–0.5 m range.

Two consequences that are easy to get wrong:
- Distances must be computed **after** reprojecting to a common frame. Subtracting eastings from two tiles in different UTM zones, or from one metric and one geographic tile, produces a number that is not a distance.
- Tiles in `EPSG:4326` store their pixel size in **degrees**, not metres (measured ~4.5e-6°, i.e. ≈0.5 m at that latitude). Any code that reads `transform.a` as metres and uses it as a GSD is wrong for those tiles by a factor of ~10^5. The GSD must be derived from the pixel size converted to metres.

This is the best of the three tiers: `r` is estimable and the block sizes can be set to 0.5r, 1r and 2r as pre-registered, rather than the study being reduced to grouping or to a qualitative statement.

**Evidence — size heterogeneity.** The measured size census over all 3,838 images is:

| Size | Tiles |
|---|---|
| 1024×1024 | 2,747 |
| 1000×1000 | 710 |
| 650×650 | 260 |
| 900×900 | 50 |
| 439×406 | 35 |
| 438×406 | 34 |
| 958×1024 | 1 |
| 439×407 | 1 |

**1,091 of 3,838 tiles (28.4%) are not 1024×1024**, and the smallest is 438×406. Image/label dimension mismatches within a pair: **0** — every tile's label matches its image exactly. The paper states 1024×1024, and the official `dataset.py` explains the discrepancy: it resizes any non-power-of-two tile up to the next power of two (BICUBIC for the image, NEAREST for the mask), so 1000×1000 is resampled to 1024×1024 at load time rather than being a property of the stored data.

**Why cropping/padding rather than resampling.** Resampling changes the effective GSD differently for each tile — a 439×406 tile upsampled to 512×512 would have its pixel size multiplied by ~0.86 while a 1024×1024 tile cropped to 512 would keep its own. Since the study's central quantity is a *spatial* one, a per-tile change in ground sampling that is correlated with region would be a confounder. Training therefore uses random 512×512 crops, and evaluation reads the **full tile** so the reported accuracy describes the whole image rather than a central sub-window. Tiles smaller than the crop in one dimension are padded (reflect for the image, ignore index for the mask). Evaluation batches mix sizes and pad to the largest tile present; padded mask positions carry the ignore index and are excluded from every metric.

**Consequence for the protocol.** The protocol's T3 branch — which barred the study from reporting a numeric `r` or asserting an optimal block size — does not apply. The T1 branch does, with its full obligations: report `r` with a stability assessment across variogram models, and set block sizes from the estimate.

**Affects**: `research_protocol.md` §5 and §7; `src/data/dataset.py` (crop/pad policy, documented); `src/spatial/dependence.py` (CRS handling, metres); `scripts/spatial_metadata_audit.py`; the manuscript's §3.2 and §3.5; the pre-registered limitation about heterogeneous tile size and GSD.

---

## D-024 — Corpus-root resolution is evidence-driven, and it was wrong in six places

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Every component that needs the extracted-corpus root must call the single shared resolver in `src/data/inventory.py`. Root choice is decided by **evidence** (do the inventory's own relative paths resolve under this candidate?) with a structural rule as a labelled fallback. A root chosen by structure alone is recorded as a guess, never as a verified fact.

**Evidence**: The corpus is nested as `data/raw/OpenEarthMap/OpenEarthMap_wo_xBD/<region>/{images,labels}/*.tif`. `discover_layout` is **recursive**, so the wrapper directory `data/raw/OpenEarthMap` and the real corpus root both report exactly **3,838 images / 3,500 labels** — an exact tie. Any rule that scores candidates by file count and keeps the first match therefore picks the wrapper, and every relative path in the inventory (`aachen\labels\aachen_1.tif`) then fails to resolve.

This was not hypothetical. It produced the session's worst failure: `extract_class_composition.py` read 2,687 readable labels and wrote nothing, reporting `no samples produced pixel counts`, with no exception raised. The same first-match-by-count logic was independently reimplemented in **six** places:

| Component | Failure mode |
|---|---|
| `extract_class_composition.py` | silent zero output (the observed failure) |
| `build_inventory.py` | wrong root recorded in the inventory |
| `build_region_metadata.py` | wrong root for region metadata |
| `audit_image_label_pairs.py` | wrong root for the pair audit |
| `run_smoke_test.py` | required `image_dir` to be truthy; it is `None` at **both** levels, so the real-data smoke test could not run at all |
| `run_experiments.py` | same, so the grid refused to start |

The resolver was validated on real data: the wrapper resolves **0/24** sampled paths, the true root resolves **24/24**. `scripts/test_root_resolution.py` locks this down with 26 assertions, including a fixture that reproduces the exact nesting.

**Alternatives rejected**: (a) Hardcoding the path — the corpus is re-extractable and the nesting is a property of the upstream archive, not of this project. (b) Choosing the deepest candidate unconditionally — correct here, but it is a guess that would be wrong for a corpus stored flat. (c) Scoring by file count — the option that caused the defect.

**Affects**: `src/data/inventory.py` (`resolve_inventory_root`, `detect_corpus_root_by_structure`, `resolve_inventory_path`); all six call sites; `scripts/test_root_resolution.py`; the manuscript's §3.1 data-provenance paragraph.

---

## D-025 — All distances are geodesic; the equirectangular approximation is barred beyond a 1° latitude span

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Distance statistics use exact great-circle (haversine) distances, or a unit-sphere KD-tree for nearest-neighbour queries. The planar equirectangular approximation is permitted only over a latitude span ≤ 1°, and raises `ValueError` beyond that unless explicitly overridden with a recorded reason.

**Evidence**: The corpus spans 98° of latitude across six continents. A single-reference-latitude equirectangular projection has an east–west error factor of `cos(lat0)/cos(lat)`. Measured inside a dataset whose mean latitude is 30°N, one degree of longitude at 60°N evaluates to **96.41 km** planar versus **55.60 km** geodesic — an overstatement of **73%** (and 81% against a 60°N reference).

Two inconsistent implementations coexisted: `strategies.haversine_km` (correct) and `dependence.to_local_km` (planar). The planar one fed the distance axis of the similarity curve and therefore propagated into `r`, the study's central quantity. Separately, `nearest_distance_km` built its KD-tree in the distorted plane, so it could select the **wrong** neighbour — and that distance sets the buffer width and the external-region selection.

`src/spatial/geometry.py` is now the single authority. Nearest neighbours are found on the unit sphere: chord length `2·sin(θ/2)` is strictly monotone in great-circle angle, so chord-space nearest neighbours are exact.

**Alternatives rejected**: Projecting to a single equal-area CRS — a global equal-area projection is equidistant nowhere, so it trades one distortion for another while appearing more rigorous. Local UTM per region — correct within a zone but introduces zone-boundary artefacts and cannot measure cross-region distances at all.

**Affects**: `src/spatial/geometry.py` (new); `src/spatial/dependence.py`; `src/splits/strategies.py`; `scripts/test_spatial_dependence.py` §5b–5f (36→52 assertions).

---

## D-026 — The dependence range is fitted on WITHIN-REGION pairs, in a window set by pair support alone

**Date**: 2026-09-17
**Status**: adopted

**Decision**: `r` is estimated from a distance–similarity curve built **only from pairs of tiles in the same region**, and the fit window is the set of lag bins supported by at least 500 pairs. The monotone-rise walk is disabled by default.

**Evidence**: Two independent problems, both measured.

*1. Bin scale.* Over all pairs the 99th-percentile separation is 18,382 km, so 20 linear bins are **459.5 km** wide — wider than the entire phenomenon, which sits inside the first bin. Restricting to within-region pairs drops the 99th percentile to 56.1 km and the bin width to **1.403 km**, and the pair count to **49,381**, few enough to enumerate exhaustively with no subsampling. The all-pairs curve is retained as context only, explicitly flagged `used_for_range: false`.

*2. Window.* The empirical semivariance over n pairs has relative standard error ≈ `sqrt(2/n)`: ~6% at n=500, ~9% at n=250, ~14% at n=100. Requiring 500 pairs therefore bounds per-bin noise at a stated level rather than at a visually chosen distance. On this curve it retains 13 of 20 bins up to 17.57 km, which keeps the plateau (γ ≈ 6.0–6.7 from 10.5 to 17.6 km) and excludes the inflated tail (γ ≈ 8.4 beyond 19 km, where support collapses below 500).

The monotone walk was swept over tolerance ∈ {0, 0.02, …, 0.20} × smoothing ∈ {off, 3-bin} and is **degenerate** on this curve:

| tolerance | 0.000 | 0.050 | 0.069 | 0.070 | 0.120 | ≥ 0.120 |
|---|---|---|---|---|---|---|
| bins kept | 8 | 10 | 10 | 12 | 13 | 13 |
| upper km | 10.54 | 13.35 | 13.35 | 16.17 | 17.57 | 17.57 |

For every tolerance ≥ 0.12 it returns **exactly** the support window, so it contributes no information; below that it truncates *inside* a genuine plateau, because a plateau whose bin-to-bin scatter is the ~6% predicted by `sqrt(2/n)` necessarily contains dips of that size. A rule with no discriminating power must not set the window. An earlier 3-bin moving average was worse still: it smoothed away real structure and let the inflated tail back in (`r = 26.22 km` against a 27.41 km window).

The fit is **weighted by pair count** (`sigma = 1/sqrt(n_pairs)`), because bins near the origin hold ~9,500 pairs while the tail holds a few hundred; an unweighted fit is effectively fitting tail noise.

**Alternatives rejected**: (a) An unweighted fit — dominated by the noisiest part of the profile. (b) A hand-picked maximum lag — the option that produced `r = 32.99 km` from a 28.12 km window, i.e. a range longer than the data. (c) Choosing the window by where the profile "looks flat" — not reproducible and not defensible in review.

**Affects**: `src/spatial/dependence.py` (`distance_similarity_curve_within_groups`, `detect_variogram_window`, `_fit_variogram`); `scripts/spatial_dependence.py`; `results/tables/spatial_dependence_curve_within_region.csv`; `scripts/test_spatial_dependence.py` §5g–5h; the manuscript's §3.5 and §4.2.

---

## D-027 — Identifiability is assessed per variogram model, and an unidentifiable fit is excluded from the reported interval

**Date**: 2026-09-17
**Status**: adopted

**Decision**: A fitted range counts as a **measurement** only if the model attains its sill *inside* the fitted window (`r < last fitted lag`). Models failing that test are reported as extrapolations and excluded from the range interval. The reported `r` is the median of the **identified** fits.

**Evidence**: All three models use the same *effective-range* parameterisation — spherical reaches its sill at `a`; exponential uses `1 − exp(−3h/r)` and gaussian `1 − exp(−3(h/r)²)`, both reaching 95% of the sill at `h = r` — so their ranges are directly comparable and a disagreement is genuine model-form disagreement, not a parameterisation artefact.

Fitted on the 17.57 km window: **spherical 13.501 km, gaussian 10.774 km, exponential 23.377 km**. The exponential fit exceeds the window edge, meaning it is still climbing at the last fitted lag and never attained its sill — it is an extrapolation. Checking identifiability only for the *selected* model (the previous behaviour) admitted that value and reported a spread of **0.93 of the median**, wrongly declaring the range unstable. Restricted to identified fits the interval is **10.77–13.50 km**, spread **0.20**, and the range is stable.

**Result**: `r = 12.137 km` (median of identified fits; spherical is the best-fitting model at R² = 0.962), interval **10.77–13.50 km**, `range_identified = true`. Block sizes 0.5r/1r/2r = **6.07 / 12.14 / 24.27 km**.

**Binding consequence**: because the interval spans a factor of 1.25, the block-size ladder is a **sensitivity range, not a fixed set**. The manuscript must not assert a single optimal block size. The sill is reached empirically by ~10.5 km under every window and model tested, which is the one statement about the dependence scale that is robust.

**Affects**: `src/spatial/dependence.py` (`estimate_range`); `data/metadata/spatial_dependence.json` (`range_per_model_km`, `range_identified`); `scripts/build_splits.py` (block-size derivation); the manuscript's §4.2 and the limitations section.

---

## D-028 — A class that appears but is never matched has F1 = 0, not NaN

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Per-class F1 is `NaN` only for a class appearing **nowhere** — neither in the ground truth nor in the prediction. A class present in the ground truth or predicted at least once, but never correctly matched, scores a defined **0**.

**Evidence**: Under the `absent='skip'` convention that mIoU uses, a class present in the ground truth but never predicted contributes a finite IoU of 0. If F1 dropped that same class as `NaN`, the two headline metrics would be averaged over **different class sets** and could not be reported side by side. It also made macro-F1 `NaN` for an entire run: the real-data smoke test produced `val_mIoU = 0.0000, val_macroF1 = NaN` for M2.

A second, independent defect surfaced while diagnosing it: `engine.py` logged `val.get("macro_f1") or float("nan")`, and `0.0 or nan` evaluates to `nan` — so a legitimate macro-F1 of exactly 0.0 was displayed as a missing value. A total failure was indistinguishable from absent data.

**Alternatives rejected**: Reporting macro-F1 over all 8 classes with absent ones counted as 0 — that would contradict the `absent='skip'` convention already stated for mIoU, and would silently lower macro-F1 by however many classes happen to be missing from a fold.

**Affects**: `src/evaluation/metrics.py` (`f1_per_class`); `src/training/engine.py` (epoch logging); `scripts/test_metrics.py` (new present-but-unpredicted section); every metric in the results tables.

---

## D-029 — The S3 disjointness QC must group by `eval_fold`, not by `fold`

**Date**: 2026-09-17
**Status**: adopted

**Decision**: Train/validation disjointness is checked per **evaluation fold**. For the expanded representation, `eval_fold` is the grouping key.

**Evidence**: S3 is stored expanded — one row per (sample, `eval_fold`) — because buffering excludes a sample from the training set of **one** fold only, which a single `split` column cannot express. Each sample therefore carries the same `fold` value in all five of its rows while its `split` changes with `eval_fold`. Grouping by `fold` puts every sample into both the train and the val set of its own group and reported **15 failures with up to 513 overlapping samples**. Grouping by `eval_fold` gives an overlap of **exactly 0**, verified directly on the built design.

The failure was in the QC, not the design — but it would have blocked the pipeline on a false alarm, and the tempting "fix" would have been to weaken the check.

**Affects**: `scripts/build_splits.py` (disjointness QC); `data/metadata/split_qc.json`.

---

## D-030 — Block sizes are derived from the recorded range artifact, and an unjustified fallback is refused

**Date**: 2026-09-17
**Status**: adopted

**Decision**: `build_splits.py` reads `r` from `data/metadata/spatial_dependence.json` (key `range.r_km`, nested) and records full provenance. If no range is available **and** no fallback is explicitly configured, the stage **fails** rather than inventing block sizes from the study extent.

**Evidence**: The script looked for `spatial_dependence_range.json`, a file that is **never written**. `r` would therefore always read as `None`, S2/S3 would silently fall back to `span/40`, `span/20`, `span/10`, and the registry would have recorded block sizes that were not derived from the dependence range — while the manuscript described them as though they were. This would have defeated RQ2, the study's core question.

Additionally, the range interval must be computed from **identified** models only (D-027). Admitting the extrapolated exponential fit inflated the reported interval to 10.77–23.38 km for the wrong reason.

**Alternatives rejected**: Falling back to study-extent scales by default. A block size of `span/20` looks plausible and is not derived from any measurement of spatial dependence, so it cannot support the study's claims.

**Affects**: `scripts/build_splits.py` (`load_dependence_range`); `data/metadata/split_qc.json` (`block_size_provenance`); the manuscript's §3.4.

---

## D-031 — The main environment runs torch 2.11.0+cu128, deviating from the version-matched plan in D-013

**Date**: 2026-09-17
**Status**: adopted — **partially supersedes D-013** (the version-match plan only; the GPU finding stands)

**Decision**: Upgrade the **main** environment to the already-downloaded `torch-2.11.0+cu128` with `torchvision 0.26.0+cu128`, rather than downloading the version-matched `torch 2.14.0+cu130` / `torchvision 0.29.0+cu130` pair. The upgrade is accepted **only if** a probe with the project's real stack passes; otherwise it is reverted.

**Evidence**: D-013 established that the GPU works (`RTX 4060`, sm_89, real matmul) and set a binding rule that no result may be called GPU-computed until the **main** environment is upgraded and probed **with the project's actual stack**. It then chose the version-matched wheels to avoid a cascade, and deferred the 2.0 GB download until the dataset transfer finished. That condition is now satisfied.

The deviation is justified by measurement, not convenience. A single epoch of M1 on 1,771 training tiles at 512² crops **did not finish in 4 m 47 s** on CPU. At ~60 epochs that is over 5 hours per run; across 261 planned runs it exceeds 1,300 hours. The grid is infeasible on CPU and the CUDA wheel is already on disk, so the choice is between a GPU-capable environment now and a 2.0 GB download at a measured ~1.6 MB/s with a history of truncation.

D-013's cascade concern is treated as a hypothesis to be **tested**, not assumed: the probe must import and run `segmentation_models_pytorch`, `transformers` and `rasterio`, and complete a forward pass for all three architectures, not merely report `torch.cuda.is_available()`.

**Binding consequence**: the recorded environment now differs in torch version from D-013's plan. `logs/pip_freeze_before_cuda.txt` preserves the pre-upgrade state for rollback, and the manuscript's Compute section must state torch 2.11.0+cu128 (not 2.14.0) unless the version-matched pair is later installed. Any GPU-computed result additionally requires the probe to have passed **and** the run's own `device` field to read `cuda`; a CPU run must never be described as GPU-computed.

**Affects**: D-013 (version-match plan superseded, GPU finding retained); `data/metadata/environment_check.json`; `environment.yml`; `requirements.txt`; the manuscript's Compute section.

---

## D-032 — `_has_imagery` must not test `layout.image_dir`; it silently re-extracted 9.1 GB per run

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: `scripts/fetch_openearthmap.py::_has_imagery` now returns
`lay.n_image_files > 0 and lay.n_label_files > 0` instead of
`bool(lay.image_dir and lay.n_label_files > 0)`.

**Evidence**: For this corpus `discover_layout(data/raw/OpenEarthMap).image_dir`
is **`None`** — by design. `Layout` deliberately leaves `image_dir`/`label_dir`
unset for a multi-region corpus (`<region>/images` + `<region>/labels`) and
carries the per-region directories in `region_dirs`, because collapsing them to
a single directory would silently truncate the corpus to one region. The
truthiness test therefore returned False for a **fully extracted** corpus, the
stage-2 skip never fired, and the inner 9,099,481,727-byte zip was re-extracted
on every pipeline run.

Observed on the 2026-09-18 run: `stage 2: extracting OpenEarthMap.zip
(9,099,481,727 bytes)` after `stage 1: already present, skipping`. All **7,343**
corpus files were rewritten in 80 s (mtimes 01:28:47Z → 01:30:07Z). This is not
merely wasted time: an extraction interrupted mid-file leaves truncated imagery,
and every downstream QC artefact keyed on mtime is invalidated.

This is the **eighth** occurrence of the same root cause recorded in D-024/D-026
— a truthiness test on a field that is legitimately `None` for this corpus.
`run_experiments.py` carried the identical bug (`required image_dir to be
truthy`), and the standing rule is that root/imagery resolution must be
evidence-driven through the shared resolver, never a truthiness check.

**Alternatives rejected**: Adding `or bool(lay.region_dirs)` — narrower, and it
would still be wrong for a single-directory corpus that happens to have no
region dirs. Counting files is layout-shape independent and covers both.

**Affects**: `scripts/fetch_openearthmap.py` (`_has_imagery`); the Data QC gate's
runtime and the integrity of `data/raw/OpenEarthMap/**` mtimes.

---

## D-033 — `fetch_openearthmap.py` emitted no `[STATUS]` line, so its gate could never PASS

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: `fetch_openearthmap.py` now prints `[PHASE]`, `[STATUS]`,
`[COMPLETED]`, `[FILES]` and, where applicable, `[ISSUES]`, and exits non-zero
when extraction produced no imagery.

**Evidence**: `grep -c STATUS scripts/fetch_openearthmap.py` returned **0**. The
protocol (section 16) requires every QC gate to emit PASS / PARTIAL / FAIL *as
code*, and `run_pipeline.py` parses exactly those lines. With none present the
orchestrator fell back to `"the script printed no [STATUS] line (exit code 0);
status inferred as PARTIAL"` — so the **Data QC** gate was permanently PARTIAL
and the pipeline's aggregate `overall` could never reach PASS, even when the
archive hash and both extraction layers were fully verified. A gate that cannot
report success is not a gate.

The step also returned 0 unconditionally, so a genuinely missing imagery tree
would have exited successfully and let `build_inventory` run against nothing.

**Alternatives rejected**: Treating the missing line as cosmetic. The whole point
of the orchestrator is that "a partial run cannot be mistaken for a complete
one"; a gate stuck at PARTIAL destroys exactly that property.

**Affects**: `scripts/fetch_openearthmap.py` (gate output, exit code);
`results/pipeline/pipeline_summary.json` and `pipeline_report.md`.

---

## D-034 — the grid's seed list must come from the config, not from a hardcoded argparse default

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: `run_experiments.py` now defaults `--seeds` to
`training.seeds` from `configs/default.yaml` (5 seeds: 42, 1337, 2024, 7, 17)
rather than the literal `"0,1,2"`. The grid is therefore **29 strategies x 3
models x 5 seeds = 435 runs**, not 261.

**Evidence**: The protocol requires "Multiple seeds (≥3, target 5)" (P0 table)
and its checklist asks for "at least 3 seeds". `configs/default.yaml` declares
`training.seeds: [42, 1337, 2024, 7, 17]`, and the config header states that
changing a value also fixed in the protocol is a protocol change requiring a
DECISIONS entry. The runner nevertheless shipped `default="0,1,2"`, so the grid
would have silently used a seed set the protocol does not specify — and one that
does not even appear in the config. `PROGRESS.md`'s "261-run grid" figure is the
product of that default, so it was a planning number, not a protocol mandate.

The project's standing convention is config-driven, no hardcoding; here the code
was authoritative and the config was decoration. That is the same class of
divergence as D-024 (config vs code) and it is fixed in the same direction.

**Alternatives rejected**: Keeping 3 seeds because it satisfies the ≥3 floor.
The config is the protocol's operationalisation and says 5; the floor is a
minimum, not a target. Also rejected: editing the config down to 3 seeds — that
would be choosing the analysis to fit the implementation.

**Affects**: `scripts/run_experiments.py` (`--seeds` default); the planned run
count (435); the runtime budget; `PROGRESS.md`'s grid-size statement.

---

## D-035 — the runner read a `train` config section that does not exist; six protocol-fixed hyperparameters silently fell back to literals

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: `run_experiments.py` now reads the `training:` section through a
loud accessor (`training_cfg()` / `cfg_required()`) that raises when a
protocol-fixed value is absent. Every hyperparameter the protocol pins down is
read from the config: `seeds`, `batch_size`, `crop_size`, `epochs`, `lr`,
`weight_decay`, `warmup_epochs`, `early_stopping.patience`, `num_workers`,
`loss.ce_weight`, `loss.dice_weight`. `--seeds`, `--epochs` and `--batch-size`
default to `None` so the config wins unless explicitly overridden.

`fit()` additionally gained `warmup_epochs`, `ce_weight` and `dice_weight`, and
the schedule became a `LambdaLR` implementing **linear warmup then cosine
decay** in place of the bare `CosineAnnealingLR`. The resolved values are logged
at grid start and persisted per run under `hyperparameters` in the run JSON.

**Evidence**: The config defines `training:`; the runner asked for
`cfg.get("train", {})`. Because `.get` cannot raise, every lookup fell through to
a literal in the source file. Comparing what the protocol and config specify
against what would actually have run:

| Item | `research_protocol.md` §11 / config | Would have run | Verdict |
|---|---|---|---|
| seeds | `42, 1337, 2024, 7, 17` | `0, 1, 2` | **divergent** |
| lr | `1e-4` | `6e-5` | **divergent** |
| early-stopping patience | `10` | `8` | **divergent** |
| batch size | `8` | `4` | **divergent** |
| loss weights | `ce 1.0 / dice 1.0` | `0.5 / 0.5` | **divergent** |
| schedule | "cosine decay with **linear warmup**" | cosine, **no warmup** | **divergent** |
| num_workers | `4` | `0` | divergent (throughput) |
| epochs | `60` | `60` | matched by coincidence |
| weight decay | `1e-4` | `1e-4` | matched by coincidence |

Six of nine diverged, and the two "matches" were coincidences of a literal
happening to equal the config — the code was authoritative and the config was
decoration. This is the same class as D-024 and D-034; it is fixed in the same
direction. The defect was caught **before** any run: `experiments_registry.csv`
did not exist and `experiments/runs/` held only one stale `.error.json` from
2026-09-17, so no result is invalidated.

`warmup_epochs` deserves separate emphasis. It was declared in the frozen config
and required by the protocol, but the string `warmup` did not occur anywhere in
`src/` or `scripts/` — a schedule asserted only in a document cannot be audited.
Warmup is not cosmetic here: all three encoders start from ImageNet weights and
would have been driven at full learning rate from the first step, and a
transformer (M3) tolerates that less gracefully than a CNN. Left as it was, the
schedule itself would have differed between architecture families, confounding
RQ3.

**Alternatives rejected**: Renaming the config section to `train` so the
existing `.get("train", ...)` calls resolve. That would edit the protocol-frozen
artifact to match a bug in the consumer, and would leave the silent-default
pattern in place for the next parameter. Also rejected: keeping the literals and
recording them as the de-facto protocol — that would be choosing the analysis to
fit the implementation.

**Affects**: `scripts/run_experiments.py` (config access, CLI defaults, `fit`
call, run provenance); `src/training/engine.py` (`fit` signature, loss weights,
schedule, `FitResult`); every hyperparameter the grid will run with; the
reproducibility of the pre-registered model.

---

## D-036 — the pipeline's `[STATUS]` regex required the verb alone, so two correct scripts read as UNKNOWN

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: `STATUS_RE` in `run_pipeline.py` now accepts and captures trailing
context after the verb:
`^\[STATUS\]\s+(PASS|PARTIAL|FAIL)\b[ \t]*(.*?)[ \t]*$`. The captured note is
recorded per step as `status_note`, logged, and included in the summary.

**Evidence**: The pattern was `^\[STATUS\]\s+(PASS|PARTIAL|FAIL)\s*$`, i.e. the
line had to be *nothing but* the verb. Two scripts legitimately append context:

- `spatial_metadata_audit.py` prints `[STATUS] PASS (tier T1)` → parsed
  UNKNOWN → the pipeline inferred **PARTIAL** for a step that had succeeded.
- `freeze_external_regions.py` prints `[STATUS] FAIL — <reason>` → parsed
  UNKNOWN → **FAIL** inferred from the exit code, with the reason never shown.

Both appeared in the 2026-09-18 pipeline run as
"the script printed no [STATUS] line", which was false: the lines were present
and correct. The defect was therefore doubly harmful — it mis-reported status
*and* discarded the diagnostic text that explained it. Relaxing the anchor fixes
both directions and removes the pressure to delete context from status lines.

**Alternatives rejected**: Editing the two scripts to print a bare verb. That
treats the symptom, deletes useful information (`tier T1` is the study's
coordinate tier), and leaves the trap for the next script.

**Affects**: `scripts/run_pipeline.py` (status parsing, per-step reporting,
summary); the reading of every gate result.

---

## D-037 — an existing freeze must be verified and pass, not fail; and its content hash must be checked

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: When `external_regions.json` exists and `--force` was not given,
`freeze_external_regions.py` now (a) recomputes `content_sha256`, (b) emits
`[STATUS] PASS` with an explicit note and exits 0 if it matches, and (c) emits
`[STATUS] FAIL` and exits 1 if it does not. Nothing is written on either path.

**Evidence**: The previous guard logged "already exists and is FROZEN", printed
`[STATUS] FAIL`, and returned 1. The External QC gate (protocol §16) asks whether
the freeze **exists and is unchanged** — not whether it was written during this
particular invocation. A routine pipeline re-run therefore reported
`External QC=FAIL`, and because the message was indistinguishable from a real
breakage, the failure had to be explained by hand in
`results/pipeline/freeze_verification.md`. Measured: stored `content_sha256`
`d7a3319b…` equals the recomputed value, 12 external regions, frozen
2026-09-17T08:50:55Z — the freeze is intact.

The guard was also **missing the check that matters**. It tested existence, never
integrity. A truncated or hand-edited `external_regions.json` would have been
accepted as "frozen", while `build_splits.py` derived every partition from
whatever the file then contained. Verifying the hash converts a guard against
*overwriting* into a guard against *drifting*.

**Alternatives rejected**: Leaving exit 1 and documenting it as expected. That
leaves a permanent FAIL in every pipeline summary, which trains the reader to
ignore the one gate whose whole purpose is to protect the study's central
comparison. Also rejected: passing `--force` to silence it — that rewrites the
freeze and invalidates every partition already derived from it.

**Affects**: `scripts/freeze_external_regions.py`; the External QC gate's
meaning; `results/pipeline/freeze_verification.md` (now a record, not an
explanation of an apparent failure).

---

## D-038 — partition files carry ASSIGNMENTS only; the runner joins imagery paths from the inventory

**Date**: 2026-09-18
**Status**: adopted — defect fixed

**Decision**: A partition file records *which sample is train/val in which fold*
and nothing else. `run_experiments.py` resolves `image_path` / `label_path` by
merging the partition against `data/metadata/dataset_inventory.csv` on
`sample_id` (`path_lookup` + `with_paths()`). The merge raises if any partition
`sample_id` is absent from the inventory, rather than silently producing a frame
with unresolved paths.

**Evidence**: An abandoned run from 2026-09-17 is preserved as
`experiments/runs/S0_random_patch__M1__seed0__fold0.error.json`:

```
KeyError: "['image_path', 'label_path'] not in index"
  run_experiments.py line 325: train_df = train_df[meta_cols].reset_index(drop=True)
```

`data/splits/S0_random_patch.csv` has columns
`sample_id, region, country, lon, lat, split, fold, strategy, block_id,
distance_to_training` — no path columns. The runner assumed they were present.
Left unfixed, **every one of the 435 runs** would have died at the same line
before a single epoch, and the failure would have looked like a data problem
rather than a schema mismatch.

**Alternatives rejected**: (a) *Writing paths into the partition files.* Rejected
because a partition is an assignment and a path is an inventory fact; if the
inventory is ever rebuilt (a second extraction, a relocated corpus) the embedded
paths become stale copies that no QC gate compares against anything. Keeping the
join at run time makes a stale inventory impossible by construction — there is
only one copy. (b) *A left-merge that tolerates missing ids.* Rejected: it would
train on whatever rows happened to resolve, and a partition that disagrees with
the inventory is exactly the condition that must stop the grid, not degrade it.

**Affects**: `scripts/run_experiments.py` (`path_lookup`, `with_paths`);
`scripts/build_splits.py` (unchanged — its narrow column set is now the
documented contract); the stale `error.json`, which the retry will overwrite.

---

## D-039 — code edits must be read back before a long run; a same-file second edit can silently drop the first

**Date**: 2026-09-18
**Status**: adopted — verification made mandatory

**Decision**: After editing any file that a long run depends on, the change is
**read back from disk** and, where the change is behavioural, exercised. Two
edits sent to the *same file* in one batch may leave only the second applied. A
project script `scripts/verify_repairs.py` performs this read-back for every
protocol-critical repair and exits non-zero if any is missing.

**Evidence**: Three repairs were reported as applied and were not:

| Repair | Symptom when lost | How it surfaced |
|---|---|---|
| `freeze_content_sha256` added to the imports of `freeze_external_regions.py` | `NameError` inside the External QC gate → gate FAIL | pipeline summary, run 3 |
| `STATUS_RE` widened in `run_pipeline.py` | two correct scripts read as UNKNOWN | pipeline summary, run 2 |
| `crop_size=crop_size` in the runner | training would use 512 while the log claimed the config value | `verify_repairs.py` |

The failure mode is quiet by construction: the editor returns success, the file
on disk is syntactically valid, and the defect appears far from its cause — as a
`NameError` one call deep, or as a gate status, or as a hyperparameter that is
merely *wrong* rather than absent. Two of the three were only caught because a
downstream gate happened to print the consequence.

**Alternatives rejected**: (a) *Trusting the editor's success message.* That is
what produced the three losses. (b) *One edit per turn.* Safer but not
sufficient — the loss is silent, so the only reliable check is reading the file
back. (c) *Relying on the pipeline to catch it.* It caught two of three; the
`crop_size` loss was behaviourally invisible because the literal happened to
match.

**Affects**: every script under `scripts/`; `scripts/verify_repairs.py`;
the read-back step before launching the 435-run grid. Recorded as a skill
(`verify-file-edits-persisted`) so the check is not re-derived each session.

---

## D-040 — the epoch budget is 20, not 60: the first costed decision this project has made about its own grid

**Date**: 2026-09-19
**Status**: adopted — post-measurement, pre-result

**Decision**: `configs/default.yaml` `training.epochs` is **20**. Every other
training setting fixed by `research_protocol.md` §11 is unchanged. The value is
read from the config by the runner (D-035), recorded per run in the run JSON's
`hyperparameters` block, and hashed into `config_sha256`.

**Evidence**: `scripts/benchmark_throughput.py` was run against the real fold-0
training frame (1,771 samples), real imagery, batch 8, 512 px crops, AMP enabled
(`engine.train_one_epoch` uses `torch.autocast` + `GradScaler` on CUDA):

| Model | s/step | s/epoch | 60-epoch run | 20-epoch run |
|---|---|---|---|---|
| M1 U-Net resnet34 | 0.283 | 62.5 | 1.04 h | 0.35 h |
| M2 DeepLabV3+ resnet50 | 0.355 | 78.4 | 1.31 h | 0.44 h |
| M3 SegFormer-B0 | 0.246 | 54.3 | 0.90 h | 0.30 h |

The protocol's full grid — 29 partitions × 3 models × 5 seeds = **435 runs** —
therefore costs a measured **471.9 h ≈ 19.7 days** at 60 epochs and ≈ 6.6 days
at 20 epochs on this machine's single RTX 4060. Neither fits the resources
available to this project; 20 epochs is the largest budget at which the
protocol's own P0 ladder (§15) can be executed in a coherent, reportable form.

**The grid is GPU-compute-bound, not I/O-bound** — established rather than
assumed, because the opposite conclusion would have changed the fix. The same
`SegDataset` + `DataLoader` delivers an epoch in 16.1 s with 4 workers and
10.9 s with 8, against 54–78 s of compute per epoch. Two earlier measurements
were misleading and are recorded so they are not re-derived: a loader-only run
of 25 batches measured 0.35 s/step (worker-startup cost, no warm-up), and the
per-sample decode cost is 43.9 ms of which 17.0 ms is the RGB TIFF, 9.8 ms the
label TIFF and 4.6 ms the label remap.

**Consequences, disclosed rather than absorbed**:
1. The models are trained for one third of the originally planned budget and are
   **undertrained relative to a converged solution**. Absolute mIoU values are
   not comparable with published OpenEarthMap baselines, and the manuscript
   states this wherever a level (as opposed to a difference) is reported.
2. The comparison itself is unaffected in its treatment: the epoch budget, the
   cosine schedule (built from this same value, so it is self-consistent), the
   optimiser, the loss, the augmentation policy and the early-stopping patience
   are identical for every strategy, model and seed. What changes is the point
   on the training curve at which the estimator is evaluated — which is a
   limitation on generalization to converged models, stated as such.
3. Early stopping (patience 10) is now unable to fire before the budget ends
   except in pathological cases, so `epochs_run` will cluster at 20. This is
   reported rather than hidden, because §11 makes `epochs_run` part of the
   treatment.

**Alternatives rejected**: (a) *Run 60 epochs and report whatever completes.*
Rejected: at 1.07 h per run the P0 core alone (45 runs) takes 48 h, and a
truncation at an arbitrary point is precisely the outcome §15 exists to prevent.
(b) *Reduce the corpus instead.* Rejected: it changes the spatial structure
under study, which is the object of measurement. (c) *Reduce the crop to 256 px.*
Rejected as the larger scientific change: it alters the receptive-field context
at which land-cover segmentation is learned, whereas the epoch budget alters
only how far along the same curve training has travelled. (d) *Claim the
protocol was followed unchanged.* Prohibited — §22 requires post-freeze
modifications to be appended here with their rationale.

**Affects**: `configs/default.yaml`; every run JSON and the registry's
`epochs_run` column; manuscript §3.9 (training protocol) and the limitations
section, which now carry the reduced budget explicitly.

---

## D-041 — the throughput benchmark had never been executed; it was broken in two ways and the first costed number in this project came from repairing it

**Date**: 2026-09-19
**Status**: adopted — defect fixed

**Decision**: `scripts/benchmark_throughput.py` is repaired and is the standing
instrument for costing a grid before launching it.

**Evidence**: Two defects, both of which would have made the script fail or
mislead rather than merely be inconvenient:

1. `from training.engine import CombinedLoss, build_model` — `build_model` lives
   in `models.builders`; the import raised `ImportError` immediately. The script
   had therefore never run, which is why the project's cost figures up to
   2026-09-18 were **extrapolations from a CPU measurement** ("the 261-run grid
   would exceed 1,300 h") rather than measurements of the GPU the grid uses.
2. `train = part[part["split"] == "train"]` — the obsolete convention that
   D-014's repair removed from `strategies.py` and that
   `splits.diagnostics.fold_frames` exists to replace. Every compact partition
   labels each sample `split="val"`, so this frame was **empty**; with an empty
   frame the script would have reported `steps_per_epoch = 0` and a 0.0 h grid.
   Membership now comes from `fold_frames()`.

Two further mismatches surfaced on the repaired path: `build_model` requires
`num_classes`, and `SegDataset.__getitem__` returns a dict (`{"image","mask",
"sample_id"}`), not a tuple. Both are now handled.

**Alternatives rejected**: (a) *Timing a throwaway call to
`run_experiments.py`.* Rejected: the runner appends to
`experiments/experiments_registry.csv`, the append-only evidence trail, so a
timing row would remain in it forever and the registry would stop meaning "runs
that produced results". (b) *Estimating from FLOPs.* Rejected: the project's
entire claim of methodological care rests on measured, not modelled, quantities.

**Affects**: `scripts/benchmark_throughput.py`; D-040, which rests on its
output; `logs/benchmark_throughput.log`.

---

## D-042 — the grid is launched as an ordered plan of stages, not as one flat 435-run call

**Date**: 2026-09-19
**Status**: adopted

**Decision**: `scripts/run_grid_plan.py` runs the grid as nine ordered stages,
each an explicit `run_experiments.py` invocation: (1) the P0 strategies at seed
42; (2) the buffered design at seed 42; (3–4) the remaining P0 seeds;
(5) the mandatory matched-sample-size control; (6) the buffered design at the
remaining seeds; (7) P0 seeds 7 and 17; (8–9) the P1 block-size and buffer
sweeps. Progress is mirrored to `results/pipeline/grid_plan_status.json`.

**Evidence**: A flat invocation walks `all_splits_index.csv` in file order and
would, on this machine, spend its first 24 hours on `S0` × three models × five
seeds — 15 runs of a single strategy. If it were then interrupted, the completed
set would contain no spatial-block and no leave-region-out result at all: the
study would have a random-split baseline and nothing to compare it against.
§15's rule is that "the completed subset must remain a coherent, reportable
study if execution is truncated", and file order does not satisfy it.

**Alternatives rejected**: (a) *A single 435-run call with a time limit.*
Rejected: the truncation point is then determined by file order rather than by
scientific priority. (b) *Ordering by strategy only.* Rejected: it delays the
model comparison (RQ3), which needs all three architectures at one seed before
it needs a second seed of one architecture.

**Affects**: `scripts/run_grid_plan.py`; `results/pipeline/grid_plan_status.json`;
the order in which `experiments/experiments_registry.csv` fills.

---

## D-043 — the grid was killed by host RAM, not by a training defect: `num_workers` 4 → 2

**Date**: 2026-09-19
**Status**: adopted — environment defect fixed

**Decision**: `training.num_workers` is **2**. `num_workers` is a throughput
knob; it changes neither the data the model sees nor the order of optimisation
steps, so no result depends on it.

**Evidence**: The first six runs of the plan all failed, each after ~75–80 s —
that is, during the first epoch's validation — with

```
MemoryError: Unable to allocate 8.00 MiB for an array with shape (8388608,)
and data type bool
  engine.py:363  _, val = predict_and_score(model, val_loader, device)
  metrics.py:56  oob = (pred_v < 0) | (pred_v >= self.k)
```

The failing allocation is **8 MB**. The host was at its commit limit, not short
of a particular buffer: 15.82 GB physical RAM with 2.16 GB free, against a
30.82 GB commit limit with 1.60 GB free. Four DataLoader workers held ~1.6 GB
private each (~6.4 GB total) — torch, rasterio and albumentations baseline plus
pinned prefetch buffers for 1024 × 1024 tiles. `Memory Compression` was active.

The identification is by measurement, not inference: stopping the grid and its
workers moved free physical memory 2.16 → 7.50 GB and free commit
1.60 → 15.16 GB.

**Why this mattered more than a retry would suggest**: the runner records a
failed run as `<run_id>.error.json` and **continues**, so the grid was burning
~80 s per run and producing a directory of failures while the registry stayed
empty. A resumed grid would have repeated exactly this. The failure was also
misleading on its face — an 8 MB allocation failure inside a confusion-matrix
update looks like a bug in `metrics.py`, and it is not: `metrics.py` is
correct, and `test_metrics.py` passes 40 assertions against it.

**Alternatives rejected**: (a) *Retry and hope.* Rejected: the cause was
systematic and reproducible across all six runs and all three architectures.
(b) *Keep 4 workers and reduce the validation batch size.* Rejected as the
larger intervention — it changes the evaluation path, which is the object of
measurement, to protect a throughput setting that is not. (c) *Free memory by
closing the user's applications.* Not this project's decision to make.
(d) *Swap the corpus to a smaller format.* Rejected: the grid is
GPU-compute-bound (D-040), so format work buys no run time.

**Affects**: `configs/default.yaml`; all runs after this point. The six failed
runs left no registry rows, so nothing is invalidated and nothing must be
discarded — `.error.json` files are retried and overwritten on the next pass.

---

## D-044 — `collate` could not pad a mixed-size evaluation batch, so every run died in epoch-1 validation

**Date**: 2026-09-19
**Status**: adopted — defect fixed, regression test added

**Decision**: `src/data/dataset.py :: collate` selects its padding mode per
tensor: `reflect` while the padding is strictly smaller than the dimension being
padded, `constant` (0, i.e. the ImageNet mean **after** normalisation) beyond
that. Mask padding is unchanged — it always carries `IGNORE_INDEX`.

**Evidence**: The second launch of the grid failed every run at ~80 s with

```
RuntimeError: Padding size should be less than the corresponding input
dimension, but got: padding (0, 586) at dimension 2 of input [3, 406, 438]
  engine.py:199  for batch in loader:            (predict_and_score)
  dataset.py:305 x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
```

The corpus is not uniformly sized — the config records 1,091 of 3,838 tiles off
1024×1024, with 439×406 and 958×1024 present — and `collate` pads every member of
a batch up to the batch maximum. Pairing a 406-row tile with a 1024-row tile in
an evaluation batch asks for 586 rows of reflection, which is undefined.

**Why it survived every test built so far.** Training crops every tile to
512 px, so training never pads. Only **evaluation** reads full tiles, and both
the synthetic smoke test and `test_metrics.py` use uniformly sized fixtures. The
defect requires *two different tile sizes in one batch*, which no earlier
fixture produced — a reminder that a uniform fixture cannot test an assumption
about non-uniformity, however many assertions it carries. The real-data smoke
test passed because it reads a single small subset.

**Impact had it shipped**: nothing would have been trained at all. The runner
records a failure and continues, so the visible symptom would have been an
ever-growing pile of `.error.json` files and an empty registry.

**Verification**: `scripts/test_dataloader.py` — 11 assertions, all passing:
the exact 406×438 + 1024×1024 collision, mask padding carrying `IGNORE_INDEX`,
real pixels surviving intact, reflection still used for a 24 px pad (so the fix
did not silently downgrade every batch to constant padding), and two **real
tiles of different size** collated straight from the corpus. The full pipeline
path (1 epoch of `fit` with its internal batch-8 validation, then grouped
validation and external evaluation) was additionally run end-to-end on real
subsets before relaunching.

**Alternatives rejected**: (a) *Zero-pad unconditionally.* Rejected: reflection
is defined for the common case (1000×1000 → 1024×1024) and avoids injecting a
constant edge into a tile's receptive field; the fallback applies only where
reflection is impossible. (b) *Evaluate at batch size 1 everywhere.* It would
side-step the pad entirely, but it multiplies evaluation cost across 20 epochs
per run on a compute-bound host (D-040) to avoid a fix that is exact. The
reported metrics are already padding-free: the runner evaluates validation and
the external set at batch 1, so only the early-stopping metric ever sees a
padded batch, and padded positions cannot enter it.
(c) *Drop the non-1024 tiles.* Rejected outright: it would delete real regions'
data and change the spatial structure under study.

**Affects**: `src/data/dataset.py`; `scripts/test_dataloader.py` (new);
every run of the grid.

---

## D-045 — two definitions of "the freeze hash": the External QC gate could never pass, and said the opposite of the truth

**Date**: 2026-09-19
**Status**: adopted — defect fixed

**Decision**: `scripts/analyze_results.py` computes the current external-test
freeze hash with `common.freeze_content_sha256()` — the same function the freeze
script writes with and the runner verifies with. One hash, one definition.

**Evidence**: On the first analysed run the statistics gate reported

```
FAIL  external freeze consistent across runs: sha=5b2aa51ec9965893
ISSUES  the freeze recorded in the registry (d7a3319b7b1f9622) does not match
        the current freeze (5b2aa51ec9965893); the external test set has been
        modified since these runs
```

Nothing had been modified. `analyze_results.py` did two different things: it
read `freeze.get("freeze_sha256")` — a key that has **never existed** in the
record, whose field is `content_sha256` — and then fell back to
`sha256_json_canonical(freeze)` over the **whole** record. That fallback is
necessarily a different hash from the one the runner stores, because
`freeze_content_sha256` excludes the `content_sha256` field (a digest cannot
contain itself). The mismatch is structural, not accidental: those two numbers
can never agree. Verified directly: stored `content_sha256` = recomputed
`freeze_content_sha256` = `d7a3319b…`, and the 12 external regions are intact.

**Why a false alarm here is the worst possible failure mode.** The external
geographic test is the arbiter of the entire study: every OB, AEE, GG and RI
value is measured against it. A gate that cries tampering on a pristine freeze
has two bad outcomes and no good one — either a *real* later modification is
dismissed as the same known noise, or valid results are discarded. It also
suppressed the gate's PASS for reasons unrelated to the analysis.

**Verification**: after the fix the same gate reads
`PASS external freeze consistent across runs: sha=d7a3319b7b1f9622`, matching
the value in the registry row and the value stored in `external_regions.json`.

**Alternatives rejected**: (a) *Accept either hash.* Rejected: it would make the
gate unfalsifiable and let a genuine mid-grid freeze change pass unnoticed.
(b) *Store the whole-record hash in the registry instead.* Rejected: the freeze
script's own integrity check (D-037) uses the content hash, so the registry
would then disagree with the freeze document it is supposed to attest to.

**Affects**: `scripts/analyze_results.py`; the statistics gate's External-QC
check; `results/statistics/statistics_qc.json`.

---

## D-046 — inference on full-size tiles must be padded to the encoder stride; every DeepLabV3+ run died at the first validation

**Date**: 2026-09-20
**Status**: adopted — defect fixed, verified on real tiles of five sizes

**Decision**: `training/engine.py :: predict_and_score` pads the input up to a
multiple of `ENCODER_PAD_MULTIPLE = 32` before the forward pass and crops the
logits back to the tile's own size afterwards. The padded region never reaches a
confusion matrix, so it cannot influence any metric.

**Evidence**: Every M2 run failed at the first epoch's validation with

```
RuntimeError: Wrong input shape height=1000, width=1000. Expected image height
and width divisible by 16. Consider pad your images to shape (1008, 1008).
  run_experiments.py  [2/9] S0_random_patch__M2__seed42__fold0
```

DeepLabV3+ with an ImageNet ResNet backbone requires spatial dimensions divisible
by its output stride. Training never hit this because training crops every tile
to 512; **evaluation reads full tiles**, and 710 of the corpus's tiles are
1000 × 1000, 260 are 650 × 650, and further sizes are 900, 958 × 1024, 439 × 406.
M1 (U-Net) and M3 (SegFormer-B0) tolerate arbitrary sizes, so the defect looked
model-specific — but it is not: it is a property of evaluating full-size tiles
with any stride-constrained encoder, and it would have removed one of the three
architectures from RQ3 entirely while the other two ran normally.

32 is used for all three models rather than 16 for the affected one, so the
evaluation path is identical across architectures; a per-model pad constant
would make the evaluation procedure itself a model-dependent factor, which §10
prohibits.

**Verification**: the real evaluation path was run on a subset containing four
1000 × 1000 tiles, two 1024 × 1024 tiles and two 650 × 650 tiles. Before the fix
M2 raised; after it, all three models complete and produce finite metrics over
the same 6,942,152 pixels
(`M1 mIoU = 0.0129`, `M2 mIoU = 0.0262`, `M3 mIoU = 0.0441` **on an untrained
checkpoint**, so those values are a smoke check and carry no scientific meaning).

**Alternatives rejected**: (a) *Resize every tile to a common size.* Rejected:
it resamples the imagery and therefore the ground sample distance, changing the
data to fit the code. (b) *Evaluate at 512 px crops, as in training.* Rejected:
it would report accuracy over a central sub-window rather than the whole tile,
which changes the quantity being measured — the reported accuracy would no
longer describe the tile. (c) *Drop M2.* Rejected: it removes a third of the
architecture comparison to avoid a nine-line fix.

**Affects**: `src/training/engine.py`; every M2 run, and any future
stride-constrained encoder. **Note on the running grid**: the stage-1 process had
already imported the pre-fix module, so its M2 runs fail (cheaply, at the first
validation, ~80 s each) and the launcher's fixed stage list will not revisit
them. The plan is therefore re-run once after it completes; because
`run_experiments.py` skips any `run_id` already in the registry, the second pass
fills exactly the missing M2 runs and recomputes nothing.

---

## D-047 — the pre-registered primary contrast matched nothing, and the statistics gate passed anyway

**Date**: 2026-09-20
**Status**: adopted — defect fixed, and the gate that missed it closed

**Decision**: (1) `statanalysis.reporting` maps a partition label to its protocol
strategy ID in one place (`strategy_id`, `select_strategy`), so `S0` selects
`S0_random_patch`, `S2` selects every `S2_*` partition. (2) The statistics gate
gains a check that the primary contrast was **actually computed**, not merely
declared pre-specified.

**Evidence**: With two runs on record the primary contrast returned

```json
{"name": "PRIMARY: S0 vs EXT on mIoU", "pre_specified": true,
 "run_level": null, "note": "no runs for strategy 'S0'"}
```

`primary_contrast_s0_vs_ext` filtered the registry with
`df["strategy"] == "S0"`, but partitions are named by design
(`S0_random_patch`) while `research_protocol.md` §13.4 and the pre-registered
contrast name strategies by ID. The two never matched. Meanwhile the gate line
"primary contrast pre-specified and unadjusted" reported **PASS**, because it
checked a boolean flag rather than a result: the study's one pre-registered
analysis could have been absent from the manuscript while the QC record
certified compliance.

This is the same failure shape as D-014 (`design` vs `Strategy`) and D-035
(`train` vs `training`): two names for one concept, a `.get`-style lookup that
cannot raise, and a downstream artefact that looks valid. The difference here is
that the artefact is the headline test.

**Verification**: after the fix the same two-run registry yields a computed
contrast — `mean_a = 0.6324` (validation), `mean_b = 0.5144` (external),
`mean_diff = +0.1180`, resampling unit "model x seed", `n_pairs = 1`, and a
Wilcoxon result that is correctly **refused** with
`"too few pairs for a signed-rank test; not computed"` rather than reported at a
p-value from one pair. The new gate line reads
`PASS primary contrast actually computed`.

**Alternatives rejected**: (a) *Rename the partitions to `S0`, `S1`, ….*
Rejected: the labels carry the design parameters (`S2_spatial_block_1r`,
`S3_buffered_2r`) that make the registry readable, and renaming them would
invalidate 29 existing partition files and every run recorded against them.
(b) *Loosen the match to `startswith("S0")`.* Rejected in favour of an explicit
ID mapping: a prefix test would also match a future `S0B_*` partition and
silently pool it into the primary contrast. (c) *Leave the gate as it was.*
Rejected: a gate that passes on an uncomputed pre-registered analysis is worse
than no gate.

**Affects**: `src/statanalysis/reporting.py`; `scripts/analyze_results.py`;
`results/tables/primary_contrast.json`; `results/statistics/statistics_qc.json`.

---

## D-048 — an interval that did not contain its own point estimate: two estimands in one row

**Date**: 2026-09-20
**Status**: adopted — defect fixed, invariant added

**Decision**: `strategy_level_summary` reports the run-level estimate with a
run-level interval (`mean` / `ci_low` / `ci_high`, resampling unit
"model x seed") **and**, separately, the region-level estimand with its own mean
and interval (`region_mean` / `region_ci_low` / `region_ci_high`, unit
"region"). No row mixes a mean computed on one unit with an interval computed on
another. `scripts/analyze_results.py` gains a QC invariant: every reported
interval must contain its own point estimate.

**Evidence**: `results/tables/strategy_summary.csv` contained

```
strategy              metric    n  mean    ci_low  ci_high  resampling_unit
S0_random_patch       ext_miou  3  0.5242  0.4428  0.5129   region
```

The interval excludes the mean it belongs to. The cause was two estimands in one
row: the mean averaged over **runs** (three models), while the interval was a
bootstrap over **regions** (per-region means, equal weight per region across 12
external regions). The region-weighted quantity is ≈0.4776, not 0.5242, so the
interval was correct for a number that was not printed next to it.

**Why this matters beyond the arithmetic.** Protocol §13.1 asks for the region as
the resampling unit for geographic designs, and that requirement is about a
region-level estimand. Satisfying it by attaching a region bootstrap to a
run-level mean satisfies the letter of the requirement while misreporting the
quantity — and an interval that excludes its own point estimate is the visible
symptom of a mistake that is otherwise silent, because both numbers look
plausible in isolation.

**Verification**: after the fix the same three rows read `mean = 0.5242` with
`ci = [0.5144, 0.5390]` (unit "model x seed") and `region_mean = 0.4776` with its
own interval over `n_regions = 12`. The new gate line reads
`PASS every interval contains its own point estimate`.

**Alternatives rejected**: (a) *Keep the region bootstrap and relabel the mean.*
Rejected: the run-level mean is the quantity the study's primary contrast uses,
so it must stay, and the region-level quantity is genuinely informative and
belongs beside it. (b) *Drop the region interval.* Rejected: §13.1 requires it
for geographic designs.

**Affects**: `src/statanalysis/reporting.py`; `scripts/analyze_results.py`;
`results/tables/strategy_summary.csv`; the statistics gate.

---

## D-049 — F4 plotted the curve the analysis declines to use, and dropped the one it uses

**Date**: 2026-09-20
**Status**: adopted — defect fixed

**Decision**: figure F4 is rebuilt with three panels: (a) the **within-region**
distance–similarity curve, which is what `r` is estimated from, annotated with
the fitted range; (b) the all-pairs curve, labelled as context only; (c) Moran's
*I* per class. Axis tick labels for strategy names use short two-line forms.

**Evidence**: `make_figures.py` read `spatial_dependence_curve.csv` (all pairs)
and looked for a distance column among `lag_km / distance_km / lag / bin_km`.
The artefact's column is `lag_centre_km`, so the figure was reported
**unavailable** — F4 is one of the manuscript's core figures and was silently
absent from the figure set. Had the column name matched, the figure would have
shown the all-pairs curve, whose bins are ~460 km wide and which
`spatial_dependence.json` explicitly records as "used_for_range: false", while
omitting the within-region curve from which `r = 12.137 km` is actually fitted.
The second panel was also structurally impossible: `spatial_autocorrelation.csv`
has one row **per class** (`feature`, `morans_i`, `p_sim`), not a distance
series, so it cannot be plotted against distance — it is now a per-class bar
chart.

Separately, F5 and F6 used full partition names as tick labels, which collide
into illegible overlap at three or more strategies ("S0_random_patch" printed on
top of "S2_spatial_block_1r"), and F6's annotation collided with the axis labels.

**Alternatives rejected**: (a) *Add `lag_centre_km` to the candidate list and
stop there.* Rejected: that fixes the crash and keeps the wrong curve. (b)
*Plot both curves on one axis.* Rejected: they differ by two orders of magnitude
in bin width and would imply a precision the all-pairs curve does not have.

**Affects**: `scripts/make_figures.py`; `results/figures/F4_spatial_dependence.png`,
`F5_strategy_accuracy.png`, `F6_estimation_error.png`.

---

## D-050 — `adjacent_patch_prop` was measured at a threshold 10x smaller than a tile

**Date**: 2026-09-20
**Status**: adopted — defect identified; sweep added, original field retained

**Decision**: the original `adjacent_patch_prop` (threshold 0.05 km) is left
exactly as recorded in `results/tables/spatial_independence_diagnostics.csv`,
because it is not wrong, only uninformative. A separate artefact,
`results/tables/spatial_adjacency_sweep.csv`
(`scripts/spatial_adjacency_sweep.py`), reports the share of validation tiles
whose nearest training tile is within 0.05, 0.1, 0.25, 0.5 and 1.0 km, for every
design. It reads the partitions read-only, so it cannot perturb a running grid.

**Evidence**: protocol §9 defines the adjacent-patch proportion as the share of
validation patches with a training patch *sharing a border*. This corpus's tiles
are 1024 px (or 1000, 900, 650) at 0.25–0.5 m GSD, i.e. 0.20–0.51 km across, so
two tiles that share a border have centres ~0.2–0.5 km apart — while the
diagnostic used 0.05 km. Every design therefore reported `0.0`, including the
random split, whose *minimum* train–validation distance is 0.099 km.

The sweep gives the measurement the definition asks for: **56.0 %** of
random-split validation tiles have a training tile within 0.5 km, against 5.3 %
(0.5 r), 2.5 % (1 r) and 0.8 % (2 r) for unbuffered blocks, and **0 %** for every
buffered design and for region-level hold-out. This is the evidence behind
§4.1.3 and §5.2 of the manuscript.

**Alternatives rejected**: (a) *Change the threshold in `diagnostics.py`.*
Rejected while the grid runs: the diagnostics table is regenerated by
`build_splits.py`, which also rewrites the partition files that a running grid is
reading. (b) *Report only the minimum distance.* Rejected: the minimum is a
single pair and is sensitive to one sample, whereas the proportion is a
distributional statement.

**Affects**: `scripts/spatial_adjacency_sweep.py` (new);
`results/tables/spatial_adjacency_sweep.csv`; manuscript §4.1.3.

---

## D-051 — the grid is launched as a Windows scheduled task, because a detached process was killed by a session-close event

**Date**: 2026-09-20
**Status**: adopted — execution method changed

**Decision**: `scripts/run_grid_plan.py` is launched through a Windows scheduled
task (`OEMGridPlan`), not as a `Start-Process` child or a WMI-created process.

**Evidence**: Two launches were lost to the host, not to the code.

1. A `Start-Process` child was killed together with the agent session's process
   tree; the launcher survived long enough to try the remaining stages, whose
   subprocesses then failed to start (rc = 0xC000026B) and produced eight stages
   of "0.0 min" no-ops in `grid_plan.log`.
2. A WMI-created process (`Invoke-CimMethod Win32_Process Create`) survived an
   agent-session abort, but the **training process** was still terminated
   mid-epoch after 1 h 17 min:

```
forrtl: error (200): program aborting due to window-CLOSE event
KERNELBASE.dll  ...  KERNEL32.DLL  ...  ntdll.dll
```

   That is the Intel Fortran/MKL runtime inside the interpreter aborting because
   its console received a close event. The launcher died with it, leaving the
   grid stopped and the run unrecorded (the registry still held 8 rows, so the
   run is simply retrained rather than lost from the record).

The project had already reached this conclusion once: `logs/run_grid.sh` notes
that "a background shell dies with the agent session; a scheduled task is not".
The lesson was recorded in prose but not enforced in the launch method used
later, which is why it recurred twice.

**Consequences.** Resumption is unaffected: `run_experiments.py` skipped the 8
completed runs and restarted the interrupted one immediately
(`registry: 8 completed runs`, free commit 18.46 GB). The cost of the incident is
the 1 h 17 min of the interrupted run, not any loss of record.

**Alternatives rejected**: (a) *Retry the WMI method.* Rejected: it has now
failed once for the same reason that the project's own notes predicted.
(b) *A polling supervisor that restarts the launcher.* Rejected for now: two
launchers alive at once would both skip completed runs but could double-train an
in-flight `run_id`, and a duplicate `run_id` in an append-only registry is a
corruption the statistics gate treats as an error. A supervisor needs a lock
file to be safe, which is a larger change than the problem warrants while a
single machine is running one grid.

**Affects**: how the grid is started; `logs/grid_plan.log`;
`results/pipeline/grid_plan_status.json`. The scheduled task must be deleted
(`schtasks /delete /tn OEMGridPlan /f`) once the grid is complete, so that it
cannot start a new grid in a later session.

---

## D-052 — the grid task is made self-healing, the "second launcher" is identified as the interpreter's shim, and log timestamps are corrected to the UTC they claim

**Date**: 2026-09-20
**Status**: adopted — execution method changed; one measurement defect fixed

**Context.** The grid has now lost three partial runs to the host rather than to
the code:

1. `forrtl: error (200): program aborting due to window-CLOSE event` from the
   Intel Fortran/MKL runtime inside the interpreter, when the console that owned
   the training process went away (14:25 local, and again at 16:56 local, taking
   the launcher with it).
2. The task registered under D-051 could not recover by itself: its trigger was
   a single `StartBoundary` at 23:59 with no repetition, `StartWhenAvailable`
   false, and `ExecutionTimeLimit` PT72H — a plan of 67 remaining runs at ~35 min
   each is ~39 h and would have been killed at 72 h if it had got that far.
3. `LogonType S4U` ("run whether the user is logged on or not"), which would have
   removed the console entirely, **cannot be registered from this account**:
   `Register-ScheduledTask ... Access is denied` (registration is not elevated).

**Decision.**

1. **One trigger, repeating.** `scripts/register_grid_task.ps1` registers
   `OEMGridPlan` with a single time trigger repeating every 15 minutes,
   `StartWhenAvailable`, `MultipleInstances IgnoreNew`, and no execution time
   limit. A launcher that dies is restarted within the interval; a launcher that
   finishes writes `logs/grid_plan.stop` and every later repetition exits in
   under a second.
2. **The training child has no console.** The launcher spawns
   `run_experiments.py` with `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` and
   `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1`. A process with no console cannot be sent
   the `CTRL_CLOSE_EVENT` that aborted three runs, and a detached child outlives
   its launcher. The launcher records the child's pid in
   `logs/grid_plan_child.json` and, if it is restarted while that child is still
   alive, waits for it instead of starting a second trainer.
3. **One trainer per `run_id`, enforced at the moment of training.**
   `scripts/runlock.py` provides pid-based locks; `run_experiments.py` takes
   `logs/runlocks/<run_id>.lock` immediately before it trains a run and releases
   it in a `finally`. The registry is append-only and keys a run by `run_id`, so
   a duplicate row is a corruption the statistics gate treats as an error.
4. **Log timestamps are UTC, as their trailing `Z` claims.** `get_logger` used
   `logging`'s default `%(asctime)s`, which is **local** time, and appended a
   literal `Z`. Every log line written before this change is Asia/Shanghai local
   time mislabelled UTC, i.e. 8 h ahead of the time it claims.

**Evidence.**

*(a) The "two launchers" were one launcher.* At 19:33:31 and 19:33:33 (local)
two processes with an identical launcher command line appeared two seconds
apart, which reads as the repeating task having started two grids. The process
tree says otherwise:

| pid | parent | command line |
|---|---|---|
| 2024 | — | `svchost.exe` (Task Scheduler service host) |
| 6156 | 2024 | `envs\default\Scripts\python.exe -u "...\run_grid_plan.py"` |
| 2760 | 6156 | `envs\default\Scripts\python.exe -u "...\run_grid_plan.py"` |
| 16256 | 2760 | `envs\default\Scripts\python.exe -u scripts/run_experiments.py ...` |
| 2108 | 16256 | same as 16256 |
| 15648, 16968 | 2108 | `versions\3.13.12\python.exe -c "from multiprocessing.spawn ..."` |

The interpreter in this environment is a **two-stage shim**:
`envs/default/Scripts/python.exe` (241,152 bytes) starts the real interpreter at
`versions/3.13.12/python.exe` with the same command line. Every Python command
here is therefore a parent/child pair with identical `CommandLine` text. The
workers (15648/16968) name the real interpreter path and carry
`multiprocessing.spawn`, which is what identifies the pair as a shim rather than
a second grid. **Only one training process existed**; the run was stopped by
hand before it wrote a row, so no duplicate was recorded and nothing was lost.

This mattered, because the first version of the guard added in this decision
made the launcher scan for other launchers and refuse if it found one — and it
found its own shim. At 19:36:53 two instances refused each other and the grid
was left with no process running at all. `other_launcher_pids` now excludes the
parent pid and the launcher's own children, and the remaining tie is broken by
pid so that exactly one instance proceeds.

*(b) The timestamps.* The launcher writes true UTC from `time.gmtime()`; the
runner wrote local time through `logging`. At one and the same moment on
2026-09-20 the log reads

```
2026-09-20T07:42:19Z   -> S2_spatial_block_1r__M3__seed42 rc=0 in 0.0 min; ...
2026-09-20T15:42:20Z INFO    run_experiments | [1/1] S4_leave_one_region_out__M3__seed42__fold0
```

Eight hours apart, for one event. The registry's `started_at_utc` /
`finished_at_utc` columns, every `train_seconds` value and every artefact
timestamp use `common.utcnow()` and were always correct; **this is a defect in
log readability, not in any recorded result**. Lines already written are left as
they are and the 8 h offset is stated here rather than silently rewritten.

**Alternatives rejected.**
* *Retry S4U / request elevation.* Refused by the host, and elevation is not
  something this task may assume.
* *Run the grid as a plain detached process from the agent session.* Already
  failed twice (D-051).
* *Leave the single 23:59 trigger and restart the task by hand after each
  incident.* That is what produced the third lost run: recovery depended on an
  agent session being alive to notice.
* *Hold the launcher's lock and refuse whenever any other launcher-looking
  process exists, without ancestor exclusion.* It stopped the grid (19:36:53).

**Affects**: `scripts/run_grid_plan.py`, `scripts/runlock.py`,
`scripts/register_grid_task.ps1`, `scripts/run_experiments.py`,
`src/common.py`, `logs/grid_plan.log`, `logs/runlocks/`,
`results/pipeline/grid_plan_status.json`. Regression tests:
`scripts/test_grid_plan_lock.py` (12 checks) and
`scripts/test_run_lock_duplicate.py` (refuses a run held by another live
process; registry unchanged). The task must be deleted
(`schtasks /delete /tn OEMGridPlan /f`) when the grid finishes, so that it
cannot start a new grid in a later session.

---

## D-053 — the partitions pass an independent audit; the dependence-range interval was two different quantities; per-class optimism is not uniform

**Date**: 2026-09-20
**Status**: adopted — one manuscript error corrected, one ambiguity named

**Decision.** Three things are fixed by this entry.

1. **`scripts/verify_split_validity.py` is the standing independent audit of the
   partitions** and is run before any manuscript number that describes the
   splits is frozen. It re-derives corpus counts from the inventory, recomputes
   every train–validation distance with `BallTree(metric="haversine")` rather
   than with the chunked haversine the splits were built from, and writes
   `results/pipeline/split_validity_report.md`,
   `results/tables/split_validity_per_fold.csv`,
   `results/tables/split_validity_summary.csv` and
   `results/tables/split_class_composition_val_vs_ext.csv`.
2. **The dependence range *r* carries two uncertainty statements, and they are
   named.** `spatial_dependence.json` reports `r_ci_km = [11.67, 15.54]`, the
   bootstrap 2.5th–97.5th percentile of the *best-fitting* model's range under
   resampling of the variogram's lag points. `split_qc.json` reports
   `r_interval_km = [10.774, 13.501]`, the spread of the fitted range **across
   the two identified models** (Gaussian 10.774, spherical 13.501). Both are
   legitimate; they are not the same object, and neither artefact is wrong. The
   manuscript had attached the bootstrap interval to the across-model
   comparison — see the correction below.
3. **Per-class metrics are tabulated from the stored artefacts**
   (`scripts/make_per_class_table.py` → `results/tables/per_class_metrics.csv`,
   `results/tables/per_class_bias.csv`). They were already computed and saved
   inside every `experiments/runs/<run_id>.json`; the checklist's row 13 was
   PARTIAL only because nothing had read them out. Nothing was re-evaluated.

**Evidence.**

*Independence of the partitions.* Over 202 folds in 29 partitions: 0 samples in
both training and validation, 0 of the 470 external tiles in any training or
validation set, 0 duplicated rows within any fold. The corpus closes exactly:
2,214 development + 470 external + 3 tiles in `kinshasa` (the region dropped for
having fewer than `min_tiles_per_region` = 4 tiles) = 2,687 usable tiles. The
0-byte `duplicates_exact.csv` / `duplicates_near.csv` are corroborated rather
than trusted: the audit recomputed the image SHA-256 set (0 repeats) and the
perceptual-hash nearest-neighbour distances (closest pair 46 bits apart at a
threshold of 6), so there is no near-duplicate pair anywhere in the corpus and
therefore none split across a training/validation boundary.

*The designs measure what they claim to measure.* Declared versus measured
minimum validation-to-training distance: 0.5 r buffer 6.0687 declared against
6.0707 measured; 1 r 12.1374 against 12.154; 2 r 24.2748 against 24.3331. The
quantity that must track the design parameter does so at all three settings.
The unbuffered block designs, by contrast, cannot isolate at all: their minimum
distances are 0.167, 0.167 and 0.255 km, with 5.28 %, 2.45 % and 0.80 % of
validation tiles having a training tile within 0.5 km. Leave-one-region-out is
not a guarantee either — across all 62 folds its minimum is 7.060 km, because
regions can share a border even when they are separate administrative units.

*The matched control is a class-stratified Random-Matched-N2 control, which is
what protocol §8.2 asks for.* Every `MATCHED_*` partition holds the random
split's fold structure — the same class-stratified folds, so same-region
contamination is 1.000 and ≈50 % of validation tiles have a training tile within
0.5 km — and only reduces `n_train` to the target by a seeded class-stratified
draw (`random_folds(..., n_train_target=N2)`). It therefore isolates **sample
size at random spatial structure**; it is not an isolation-matched design. The
paired comparison that separates the two effects is matched-random versus
buffered at the same `n_train`.

*The buffered design is the block design with the buffer removed from training.*
For each block size, `S2_spatial_block_Xr` and `S3_buffered_Xr` have identical
fold-0 validation sets (495 tiles at 0.5 r, 513 at 1 r, 401 at 2 r, with
identical class shares). The S2/S3 contrast is therefore exactly the effect of
removing the buffer, with the evaluation set held fixed.

*What the registry claims is what the partitions contain.* Every registry row's
`n_train` and `n_val` equal the fold sizes recomputed from its partition file,
every row used the same external freeze hash, and all eight rows share one
`config_sha256`.

*The manuscript error.* §4.1.2 read: "r = 12.137 km, with an interval of
11.67–15.54 km across the two variogram models that attained their sill". The
interval across those two models is 10.774–13.501 km; 11.67–15.54 km is the
bootstrap interval for one of them. The sentence has been rewritten to state the
point estimate, the across-model spread and the bootstrap interval as three
distinct quantities, and to state that the point estimate lies inside both.

*Per-class optimism is not uniform, and part of it is not a partitioning
effect.* Mean validation-minus-external IoU by class and design:

| class | S0 random | S2 block 1 r | S4 region hold-out |
|---|---:|---:|---:|
| Bareland | +0.311 | +0.024 | −0.249 |
| Agriculture land | +0.220 | +0.180 | +0.256 |
| Tree | +0.095 | +0.051 | −0.024 |
| Water | +0.083 | +0.111 | −0.330 |
| Developed space | +0.056 | +0.017 | −0.056 |
| Building | +0.033 | +0.005 | −0.014 |
| Rangeland | +0.020 | +0.014 | −0.055 |
| Road | +0.020 | −0.043 | −0.029 |

Agriculture land is over-estimated by a similar large margin under *every*
design including the geographically isolated one, so for that class the
difference between the development pool and the external regions
(+7.40 percentage points of pixel share, `external_regions.json`) is a better
explanation than the partition. The per-class table is what makes that
separable; the aggregate mIoU difference alone would have attributed it to the
design.

*A naming defect that is recorded rather than fixed.* `S1_group_scene` groups by
`region`, because the release carries no scene identifier, so its measured
isolation is region hold-out (minimum distance 29.383 km, same-region
contamination 0.000). The name is accurate about the implementation
(`group_col: region`) and misleading about the concept ("scene"). S1 is not part
of the 75-run grid, and re-running the partitions to rename it would replace
files that completed runs were trained on, so the name stands and the content is
documented here.

**Alternatives rejected.** *Treat the two *r* intervals as an error and pick
one.* They answer different questions and both contain the point estimate; the
defect was in the prose, not in the measurement. *Re-run `build_splits.py` to
rename the key.* It would rewrite the partition files that eight completed runs
were trained on. The key stays; the meaning is documented here and in §4.1.2.

**Affects**: `manuscript/manuscript_v1.md` §4.1.2;
`scripts/verify_split_validity.py`; `scripts/make_per_class_table.py`;
`results/pipeline/split_validity_report.md`; `results/tables/per_class_*.csv`.

---

## D-054 — the grid runs under `pythonw.exe`: a controlled probe shows the three lost runs were console-close events, not memory, not the scheduler

**Date**: 2026-09-20
**Status**: adopted — execution method changed

**Context.** Four runs have now been lost to the host, each with the same
signature and none with a Python traceback:

```
forrtl: error (200): program aborting due to window-CLOSE event
KERNELBASE.dll ... KERNEL32.DLL ... ntdll.dll
```

and, for the scheduled task that was carrying them,
`LastTaskResult = 3221225786 = 0xC000013A` (`STATUS_CONTROL_C_EXIT`). The fourth
loss happened 38 s after launch (19:37:48 → 19:38:25), while the machine was
otherwise idle with 24 GB of commit free, so the two standing explanations --
memory pressure, and the agent session closing -- do not cover it.

**Evidence: a controlled probe rather than another guess.**
`logs/_survival_probe.py` was run by a scheduled task built exactly like the
grid's, with one difference: the task's action was `pythonw.exe` instead of
`python.exe`. The probe reproduces the launcher's structure (a parent that
spawns a `DETACHED_PROCESS` child) and records what the process can actually
observe:

| measured, inside the task's process | value |
|---|---|
| `GetConsoleWindow()` | **0 — no console** |
| `IsProcessInJob` | **false — no job object** |
| `sys.stdout` | `None` (GUI subsystem, nothing redirected) |
| parent heartbeats | every 5 s, continuous for > 6 min |
| detached child heartbeats | every 5 s, continuous for > 6 min |

Both processes kept writing across a dozen agent shell sessions opening and
closing. The identical structure under `python.exe` had been killed within
20–90 s, twice, in the same period. Two mechanisms were therefore tested and
**rejected**: a job object with kill-on-close (`in_job = false`) and the task
scheduler's console policy (it stops mattering once no console exists).

**Conclusion.** A console-subsystem interpreter owns a console; a console-close
or ctrl event is delivered to it, the Intel Fortran runtime inside torch reports
it as a `window-CLOSE` abort, and the process exits with `0xC000013A` — taking
its children with it because they share the console, `DETACHED_PROCESS` or not.
A GUI-subsystem interpreter never attaches to a console, so there is nothing to
close and no such event to receive. The Fortran message was a *symptom*: it was
the runtime reporting the event, not a Fortran defect.

**Decision.**

1. `scripts/register_grid_task.ps1` registers `OEMGridPlan` with
   `pythonw.exe` as the action.
2. The launcher and the runner rebind a `None` `sys.stdout`/`sys.stderr` to a
   log file (`logs/grid_plan_console.log`,
   `logs/run_experiments_console.log`). Without this, Python discards
   tracebacks silently under a GUI interpreter, and a crash would again leave no
   evidence on disk. `logging.StreamHandler(None)` would also raise on the first
   record.
3. Everything else from D-052 stands: the repeating trigger, the launcher lock,
   the per-`run_id` lock, and the detached training child.

**Alternatives rejected.** *Raise the memory guard or reduce workers.* The losses
happened with 24 GB of commit free; memory was never the binding constraint in
the fourth case. *Register the task with `LogonType S4U` so it runs in session
0.* Refused by the host: `Access is denied` without elevation (D-052).
*Accept retraining after every kill.* The kills arrived within 20–90 s, so no
run could finish.

**Affects**: `scripts/register_grid_task.ps1`, `scripts/run_grid_plan.py`,
`scripts/run_experiments.py`, `logs/_survival_probe.py`,
`logs/grid_plan_console.log`, `logs/run_experiments_console.log`. The probe task
`OEMProbe` was unregistered after the measurement.

---

## D-055 — the figures are reviewed as images, and three defects are fixed (one of them in what F8 measured)

**Date**: 2026-09-20
**Status**: adopted — figure definitions changed

**Context.** The protocol's figure gate had been satisfied by the generator
exiting 0. Opening the PNGs is a different test, and it found three things that
running the script could not.

**1. F4's legend was drawn on top of its own data.** The legend sat at
`loc="lower left"` with `framealpha=0.9`. The within-region similarity curve
descends monotonically from −2.2 to −3.75, so the lower-left corner is exactly
where its last four points are; at 0.9 alpha they were faintly visible *through*
the legend box. Moved to `loc="upper right"` with `framealpha=1.0`, which is the
empty corner for a descending curve.

**2. Colour meant two things in one figure.** In F5 and F6 the panels were
coloured from `PALETTE` by position, so `#0072B2` meant "validation estimate" in
panel (a) and "M1" in panel (b) of the same figure, and the F6 boxes were
coloured by strategy while the panel beside them coloured by model. A reader
resolves that silently and wrongly. Colour now encodes a model wherever a model
is what varies (`MODEL_COLOURS`), and anywhere colour would encode nothing — the
estimate/truth bars (F5a), the signed-bias boxes (F6a), the concordance bars
(F8d) — a neutral grey is used instead. F7 already encoded one model per colour
with linestyle for estimate-versus-truth, and is unchanged.

**3. F8 compared rankings computed on different model sets.** The figure drew,
for each design, the models' rank under that design's validation estimate
against their rank under the external test. Two defects followed from one
cause: the three designs' slopegraphs were overlaid in a single panel, where
they collapsed onto three coincident paths that a reader cannot attribute to a
design; and, more seriously, a design with an incomplete set of runs was ranked
against the full model set. With `S4_leave_one_region_out__M3__seed42` still
training, the S4 panel ranked **two** models (M1, M2) while the reference ranked
**three** (M3, M2, M1), and printed "2 reversals" from that comparison. Ranks
are now computed on the intersection of a design's models with the reference's,
the panel states how many models it compared ("2 of 2 models re-ranked"), and
each design gets its own panel in a 2 × 2 layout.

The figure's per-panel count is now labelled **"models re-ranked"** and is
deliberately *not* the `n_reversals` column of `results/tables/ranking_concordance.csv`,
which counts discordant *pairs*. Both are correct for their own question and
neither word is used for the other's number. (`analyze_results.py` was already
right about this: it records `n_models` per strategy and restricts ρ to the
common models; only the figure was comparing across unequal sets.)

**Evidence for 3.** `results/tables/ranking_concordance.csv` at the time of
review: S4 with `n_models = 2`, ρ = 1.000, `n_reversals = 0`; S0 with
`n_models = 3`, ρ = −0.500, `n_reversals = 2`; S2 with `n_models = 3`,
ρ = 1.000, `n_reversals = 0`. The 3-model S0 panel genuinely re-ranks all three
models; the 2-model S4 panel cannot support a statement about three.

**Alternatives rejected.** *Wait until every run is complete and check the
figures once at the end.* The rank defect was a definition error, and a
definition error found at the end invalidates any text already written around
it; finding it now keeps the Results sections honest while they are being
written. *Leave the overlay and explain it in the caption.* A caption that has
to say "the three paths cannot be told apart" is describing the defect.

**Affects**: `scripts/make_figures.py` (F4, F5, F6, F8),
`results/figures/*.png` and `*.pdf`. F7 is unaffected and is still withheld
pending the block-size and buffer sweeps (plan stages 8–9).

---

## D-056 — epoch time is now measured per phase, and a loader change is rejected on measurement

**Date**: 2026-09-20
**Status**: adopted — instrumentation added; one proposed change measured and rejected

**The problem.** `S4_leave_one_region_out__M3__seed42` was completing an epoch in
**467–470 s**, while the same architecture on the two other designs had completed
epochs in **105 s** (S2) and **120 s** (S0) earlier the same day. Two candidate
causes were proposed and both were wrong, so the next step was measurement
rather than a third guess.

**What was measured, and what it ruled out.**

| measurement | value | what it rules out |
|---|---|---|
| bytes read per epoch, S4 fold 0 | 6.29 GB image + 2.10 GB label = **8.39 GB** | — |
| bytes read per epoch, S0 fold 0 | 6.88 GB + 1.71 GB = **8.59 GB** | **I/O volume: the slow run reads slightly LESS than the fast one** |
| loader throughput, 2 workers | 13.2 samples/s → 164 s/epoch | — |
| loader throughput, 4 workers | 12.6 samples/s → 172 s/epoch | **worker starvation: adding workers buys nothing, so the proposed change is rejected** |
| GPU utilisation, 20 samples over 40 s | mean **0.2 %** | GPU-bound |
| training-process CPU over the same 40 s | **2.6 s** | CPU-bound |
| VRAM | 7,810 of 8,188 MiB used, **147 MiB free** | — |
| other GPU residents | explorer, browsers, Wallpaper Engine, the desktop shell | 84 MiB recovered by stopping Wallpaper Engine |

The pattern — a process blocked in the driver with almost no CPU, a GPU that is
idle but full, and workers that keep reading — is the signature of **VRAM
oversubscription under WDDM**, where the driver migrates allocations into system
memory and every kernel launch waits on the transfer. It is stated here as the
leading hypothesis, not as an established cause: the measurement that would
settle it is per-phase timing inside the run, which did not exist when the
question was asked.

**Decision.**

1. `src/training/engine.py` now records, in every epoch's history row,
   `train_seconds`, `val_seconds`, `val_n_pixels`, `val_inference_seconds`,
   `cuda_reserved_after_train_mb`, `cuda_reserved_after_val_mb` and
   `cuda_peak_alloc_mb`, and prints the two phase times on the epoch log line.
   The first of these answers "which phase is slow"; the second answers "is the
   card full". Both are log-only: no value, decision or ordering in training
   changes.
2. `num_workers` **stays at 2**. Raising it to 4 was measured at 12.6 samples/s
   against 13.2, i.e. no gain, and it would cost ~3.2 GB of commit that D-043
   showed this host cannot always spare. The config carries the measurement.
3. The loader term is now a known quantity rather than an assumption: an epoch
   of this fold **cannot** take less than ~165 s, because 8.39 GB must come off
   the disk at the measured ~50 MB/s. Any future claim that a run is "slow" is
   made against that floor.

**Affects**: `src/training/engine.py`, `configs/default.yaml`, the `history`
array of every future `experiments/runs/<run_id>.json`, and
`logs/grid_plan.log`. The single in-flight S4/M3 run was restarted so that it
carries the instrumentation; it is retrained from epoch 1 and no registry row
was written by the abandoned attempt.

---

## D-057 — F7 is rebuilt before its data exist: the two sweeps must not be averaged together

**Date**: 2026-09-20
**Status**: adopted — figure definition changed (preventive; no result was affected)

**Context.** F7 (sensitivity to spatial separation) is withheld until the
block-size and buffer sweeps run in plan stages 8–9. Reading the code before
those runs, rather than after, found that the figure would have aggregated them
incorrectly.

**The defect.** F7 extracted one "separation parameter" from the strategy name
(`(\d+(?:\.\d+)?)r`) and grouped by it. For `r = 1` that key is identical for
`S2_spatial_block_1r` and `S3_buffered_1r`, so the figure would have plotted the
**mean of two different designs** at every sweep point: the block design, whose
training folds are almost the same size as the random split (1,783 vs 1,771) and
which still leaves 2.5 % of validation tiles within 0.5 km of training, and the
buffered design, which removes 2,259 training assignments and leaves 0 %. Their
nominal separation is the same; their isolation, their training-set size and
therefore their bias are not. A curve through those means would have looked like a
sensitivity result and would have described no design in the study.

**Decision.** F7 is rebuilt as three panels:

1. **(a) Block-size sweep** — S2 at 0.5 r, 1 r, 2 r; colour = model,
   solid = external, dashed = validation estimate.
2. **(b) Buffer-width sweep** — S3 at the same three settings, identical
   encoding.
3. **(c) Error versus *measured* separation** — absolute estimation error against
   the across-fold median train–validation distance taken from
   `results/tables/split_validity_summary.csv`, not against the nominal multiple
   of *r*. The nominal value is a design parameter; the distance actually
   achieved is the measurement, and §4.1.3 already insists that claims rest on
   the second.

The figure also takes its own width (11.4 in) instead of the shared 7.2 in: at the
shared width the three panel titles collided and the x-axis labels overlapped.

**Verification.** The code path was exercised with a temporary fabricated
registry (18 sweep rows) written to `logs/`, rendered, inspected as an image, and
then **deleted along with its output**. No fabricated value entered `results/`,
`experiments/` or the manuscript, and `results/figures/` still contains only the
four figures whose data exist.

**Affects**: `scripts/make_figures.py` (`figure_f7`). F7 remains withheld until
plan stages 8–9 complete; the gate that would have produced the mixed curve no
longer exists.

---

## D-058 — the grid runs in daily windows: one click starts it, 23:30 shuts the machine down, power-on resumes it

**Date**: 2026-09-22
**Status**: adopted — execution schedule changed at the user's request

**Decision.** The plan is now run in day-long windows rather than continuously.

| Piece | What it does |
|---|---|
| `RunGrid_Nightly.cmd` (project root) | **Double-click.** Registers/refreshes the grid task, enables and starts it, and installs the nightly shutdown. Options: `-ShutdownAt "22:00"`, `-GraceMinutes 40`, `-NoShutdownTask` |
| `scripts/night_run_until_2330.ps1` | The work behind the click |
| `scripts/night_shutdown.ps1` | Runs at 23:30: decides, then shuts down (1-minute warning; `shutdown /a` cancels) |
| `OEMGridPlan` task | **AtLogOn (delay 1 min) + 15-minute repetition** — the next power-up resumes the plan within a minute, and a killed launcher is restarted by itself |
| `OEMNightShutdown` task | Daily at 23:30 |

**Why the shutdown is not a bare `shutdown /s`.** A run registers only when it
finishes, so a shutdown that lands mid-run discards everything it has done — at
the measured 2–3.5 h per run, that is up to three hours of compute repeated the
next day. `night_shutdown.ps1` therefore waits **at most 25 minutes** when, and
only when, a run is already at epoch ≥ 18 of 20, so a run one or two epochs from
the end lands its registry row instead of being thrown away. In every other case
it shuts down immediately. It always ends by requesting the shutdown: a bug in
the script must not leave the machine running all night.

**What this costs, stated plainly.** A run interrupted by the nightly shutdown
writes no registry row, so it is retrained from epoch 1 the next day. The cost is
bounded by the run length and is *not* hidden: `logs/night_shutdown.log` records
for every night whether it shut down immediately, waited, or waited and timed
out, and the registry keeps the count of completed runs.

**Why duplicate starts are still harmless.** Both grid triggers can fire
together (logon plus a repetition boundary). The launcher holds a single-instance
lock, it refuses to start while another launcher is alive, and
`run_experiments.py` takes a per-`run_id` lock, so a second start exits without
touching the registry (`scripts/test_grid_plan_lock.py`,
`scripts/test_run_lock_duplicate.py`). This is what makes an AtLogOn trigger safe
here after it was removed in D-052 for the opposite reason — there, one trigger
was simpler to reason about before those guards existed.

**Affects**: `RunGrid_Nightly.cmd`, `scripts/night_run_until_2330.ps1`,
`scripts/night_shutdown.ps1`, `scripts/register_grid_task.ps1`,
`logs/night_shutdown.log`, the scheduled tasks `OEMGridPlan` and
`OEMNightShutdown`. To stop the nightly shutdown:
`Unregister-ScheduledTask OEMNightShutdown -Confirm:$false`.

---

## D-059 — a one-click pause, and the nightly window's first full cycle in the record

**Date**: 2026-09-23
**Status**: adopted — operational addition

**Decision.** `PauseGrid.cmd` (project root, double-click) calls
`scripts/pause_grid.ps1` and stops everything the grid is doing, in the order
that matters:

1. **the task is disabled first**, so the 15-minute repetition trigger cannot
   start the grid again a minute later;
2. **the launcher before the trainer**, because `run_one` retries a failed run
   once — killing the trainer first would simply start that run again;
3. the trainer, then any **orphaned DataLoader worker** (a worker's command line
   is `... -c "from multiprocessing.spawn import spawn_main; ..."` and does not
   match the launcher/trainer filter, so it is matched separately and killed
   **only when its parent pid is already gone** — that narrow rule cannot touch
   an unrelated multiprocessing process);
4. verification: the remaining grid processes, the registry rows before and
   after, and GPU state are printed and appended to `logs/pause_grid.log`.

**What it deliberately does not do.** It deletes nothing: the registry is
append-only and is not touched, and the stale lock files stay as the evidence of
where the pause happened (the launcher reclaims them by itself). It does not
unregister the nightly shutdown unless asked (`-AlsoRemoveShutdown`), and it says
so on screen — the machine still shutting down at 23:30 while the grid is paused
is not a surprise the user should discover later. `-DryRun` lists exactly what
would be stopped and stops nothing; it was used to verify the tool
(6 processes: 2 launcher, 2 trainer, 2 DataLoader workers; registry 23).

**The nightly window's first complete cycle, on record.** The 23:30 job on
2026-09-22 found the run at epoch 18/20, waited **6 minutes** rather than
discarding it, the run registered (rows 18 → 19) and the machine then shut down.
The next morning the machine was switched on at 09:12 and the grid resumed by
itself through the AtLogOn trigger. By 21:02 on 2026-09-23 the registry held
**23 rows**, STAGE 3 was complete and STAGE 4 was running — i.e. the daily
window costs less than one run per night when the grace rule engages.

**Affects**: `PauseGrid.cmd`, `scripts/pause_grid.ps1`, `logs/pause_grid.log`.
Resuming is unchanged: `RunGrid_Nightly.cmd`.

---

## D-060 — the pre-registered analysis scope is frozen in code, and three silent scope defects are fixed before the 75-run refresh

**Date**: 2026-10-13
**Status**: adopted — preventive; no published result was affected

**Context.** The training grid finished at 75/75 on 2026-10-12 and the analysis
layer was one refresh behind (63-run snapshot, 2026-10-08). Reading the analysis
code BEFORE re-running it — the same timing that produced D-057 before the
sweeps existed — found three defects, each of which would have been silent at
nine registered strategies (the five pre-registered designs plus the four P1
sweep strategies):

1. **The FDR family would have inflated from ten contrasts to thirty-six.**
   `analyze_results.py` called `paired_contrasts()` without an explicit pair
   list, and that function pairs every strategy present in the frame. Protocol
   §13.4 requires the family to be defined in advance; the family the
   manuscript reports is the ten contrasts among the five designs. The old
   statistics-QC check passed whenever the `family` column was non-null — which
   it would have remained while every adjusted p-value changed.
2. **F8 would have silently dropped two of five designs.** `make_figures.py`
   took `shown = strategies[:6]`; once nine strategies were registered, the
   alphabetical slice kept MATCHED, S0, S2-0.5r, S2-1r, S2-2r, S3-0.5r and
   dropped `S3_buffered_1r` and `S4_leave_one_region_out` — the two designs
   §4.5–§4.6 lean on most. The comment above that line already described the
   same defect class: `[:3]` had dropped the fourth design once.
3. **F7 would have merged MATCHED into the buffer sweep.** The sweep regex used
   `re.search`, and `MATCHED_S3_buffered_1r_N1324_rep0` contains the substring
   `S3_buffered_1r`; nine matched-control runs would have entered the buffer
   sweep's 1 r cell — the mixed-population mean D-057 was written to prevent,
   entering through a different door.

**Decision.**

- `common.PRE_REGISTERED_DESIGNS` names the five-design scope once, and
  `common.parse_sweep_strategy()` is the single ANCHORED parser for P1 sweep
  names. Both are imported, never re-implemented.
- Every inferential analysis — primary contrast, secondary family (explicit
  `pairs`), mixed models, ranking concordance — runs on the pre-registered
  designs only. The family order stays `combinations(sorted(...))` so the ten
  contrasts keep the exact order, direction and sign they had in the 63-run
  snapshot, which is what makes K2 of ROUND_PLAN_2026-10-13 a value-by-value
  verification against `results/_snapshot_63run_2026-10-08/`.
- P1 sweeps are summarised in a new `results/tables/sweep_summary.csv` with two
  scopes (`seed42_paired`, `all_available`) and within-scope deltas against
  1 r: a delta against a 1 r mean that includes seeds the endpoints do not have
  would change what the difference measures.
- The statistics gate now asserts the frozen family: FAIL on any contrast
  outside the set or on missing pre-registered contrasts; PARTIAL when the
  analysed registry is not the study registry (e.g. the synthetic validation
  frame, whose strategy names deliberately do not match).
- Figures F5, F6 and F8 are drawn on the pre-registered designs only; F7 is
  drawn on the sweeps only, and refuses to render if a MATCHED name ever passes
  its filter.
- `check_manuscript_numbers.py` no longer pools numbers from
  underscore-prefixed archive directories, so the 63-run snapshot cannot let a
  stale manuscript number pass the gate.

**Alternatives rejected**: (a) letting the family follow "all strategies
present" and re-labelling it — assembling a family after seeing which
contrasts exist is what protocol §13.4 forbids; (b) filtering only the
manuscript tables after computing everything — the contamination would remain
in the artefacts those tables are checked against; (c) renaming MATCHED to
dodge the regex collision — the collision is a signal that name parsing was
unanchored, and the next name would collide again.

**Affects**: `src/common.py`; `scripts/analyze_results.py`;
`scripts/make_figures.py`; `scripts/check_manuscript_numbers.py`; the 75-run
refresh (S3–S7 of `ROUND_PLAN_2026-10-13.md`).

---

## D-061 — figure review and manuscript writing follow the K-Dense scientific skills (GitHub), under this project's gates

**Date**: 2026-10-13
**Status**: adopted — process decision

**Decision.** At the user's instruction, the figure-review and
manuscript-writing steps of the 75-run consolidation use the
`scientific-visualization` and `scientific-writing` skills from *Scientific
Agent Skills* (K-Dense Inc., `github.com/K-Dense-AI/scientific-agent-skills`,
MIT, v2.72.0), installed under `~/.workbuddy/skills/`. Where the skills and
this project's conventions differ, the project's conventions win — in
particular the number-traceability gate, the figure-code link, and the rule
that no claim exceeds its evidence range. The skills contribute their
checklists: honest encodings, uncertainty and missing-data display, and
inspect-the-rendered-file discipline (visualization); evidence binding,
methods–results reconciliation and uncertainty preservation (writing).

**Disclosure.** The skill pack requests acknowledgement of material use;
a citation of Kassis et al. (2026, arXiv:2609.00065) is carried in the
project's software/documentation record at packaging time, with the question of
whether it also enters the manuscript's reference list left to the user at
submission preparation (an external-input item in §10 of the round plan).

**Affects**: figure review (S4), §4.3 drafting and the manuscript sync
(S5–S6).

---

## D-062 — figure legibility pass on the 75-run render: measured, not eyeballed

**Date**: 2026-10-13
**Status**: adopted — presentation fixes; no data changed

**Context.** First full render of the five figures at 75-run scope, reviewed
against the `scientific-visualization` checklist (D-061). Three legibility
defects were found by looking at the rendered images (the D-039 lesson), and
the worst was measured rather than estimated before any fix.

**Findings and fixes.**

1. **F7 panel (c) distinguished models only by alpha** (0.55 / 0.70 / 0.55) —
   M1 and M3 were drawn identically, so the panel could not show that all
   three models behave the same way, which is part of what it exists for.
   Rebuilt: colour = model (same hues as panels a/b), marker = sweep family
   (circle = block, square = buffer).
2. **F8's concordance tick label for the matched control** rendered as
   "MATCHED / S3 buffered 1r N1324 rep0", its second line spilling into panel
   (e)'s plotting area. `_short_strategy` now trims it to "MATCHED / N1324
   rep0": unique identity kept (sample size + draw), redundancy dropped (the
   caption defines the design).
3. **F5/F6 tick labels at five designs** left a measured **0.8 px** gap
   between "N1324 rep0" and "random patch" at the shared figure geometry
   (measured via text extents in the real figure, not by eye). Two changes:
   "random patch" → "random" (S0's short name; the caption defines the
   class-stratified split), and the width step 1.0 → 1.5 in per design beyond
   three. Re-measured: minimum gap **35.9 px**; all four gaps 66 / 92 / 68 /
   36 px.

**Alternatives rejected**: rotating tick labels (breaks the shared style of
F4–F8); shrinking below 7 pt (these figures are printed smaller than drawn, so
small text becomes illegible); keeping the alpha-only scheme in F7(c) (alpha
cannot carry a category readably at print size).

**Affects**: `scripts/make_figures.py` (`_short_strategy`, `figure_f7`, the F5/F6
width rules); `results/figures/*` re-rendered 2026-10-13; the reviewer final
pass (item J).

---

<!-- Template for future entries:

## D-0NN — <title>
**Date**: YYYY-MM-DD
**Status**: adopted | rejected | superseded by D-0MM
**Decision**:
**Evidence**:
**Alternatives rejected**:
**Affects**:

-->


## D-063 — 交付包结构与门禁完整性修复，结论边界复核
**Date**: 2026-10-13
**Status**: adopted
**Decision**: 源码保留子目录；复制前拒绝冲突/缺失/输出覆盖源项目；补齐三张漏检结果表，缺失值与未知表格不可 PASS；记录实际 GPU 依赖；把“无样本量效应/因果机制/部署距离匹配”改为证据支持的条件性解释，补充 run 级区间不覆盖地理抽样不确定性。验收从历史 21/21 修订为 19 MET / 2 PARTIAL。Word 文字同步但因无渲染器未过视觉门禁，正式包仅含已验证 Markdown。
**Evidence**: 新增六个测试方法全部通过；总计 12 个检查脚本 exit 0；九张表 225 项核对，数字门禁无未溯源值，划分审计 PASS。重算后主对比 run_level 字典和十个次级对比数值与 63-run 快照一致。详见本轮复核报告及 validation/integrity_review_2026-10-13。
**Affects**: 打包器、表格门禁、统计解释文字、稿件、README、依赖说明、项目状态、终期报告和验收清单。无训练变更。


## D-064 — 目标期刊定为 Remote Sensing，初投源稿与科研原稿分开保存
**Date**: 2026-10-13
**Status**: adopted
**Decision**: 用户授权自主选刊，选择 Remote Sensing 常规 Article；依据其遥感验证范围、实验细节报告要求及 AI 披露政策。保留用户指定占位符，不编造作者批准、基金、利益冲突和公开 DOI。
**Evidence**: 投稿说明中的出版商官方网页；208 词摘要、48 条引用、225 项表格核验、75 run OA 恢复、五张 600 dpi 图。
**Affects**: 独立 submission/Remote_Sensing_2026-10-13 定向稿与附件；不改实验登记或既有主要统计对比。正式提交仍需作者审核和文件排版验收。


## D-065 — 投稿前科学一致性修复及同源双语输出
**Date**: 2026-10-13
**Status**: adopted
**Decision**: 修复fold0/CV、同模型、旧配对数字、空间诊断与控制/预算/排行过强解释；保留真实初始化和共享地理样本限制；既有敏感性分析明确事后，不替换主分析。双语共用自动提取的master、表数据和绘图，真实Word渲染并逐页验收。
**Evidence**: FINAL_SUBMISSION/QA；651数字出现位置，28自动检查，48参考meta，5双语图，9正文表及5补充CSV对照，英文29页/中文42页。
**Limits**: 作者声明与公开访问保留占位；完整官方指南实时访问受限，正式递交仍需确认。未标FINAL FREEZE/正式投稿就绪，未覆盖原始实验或新增训练。

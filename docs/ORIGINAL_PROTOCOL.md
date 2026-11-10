# Research Protocol

**Project**: Effects of Spatial Partitioning Strategies on Accuracy Assessment and Geographic Generalization of Remote Sensing AI Models
**Subtitle**: A systematic evaluation based on multi-scale spatial independence
**Working title (Chinese)**: 空间划分策略对遥感人工智能精度评估与空间泛化的影响——基于多尺度空间独立性的系统评价

**Protocol version**: 1.0
**Status**: **FROZEN BEFORE MODEL TRAINING**
**Date frozen**: 2026-09-17
**Rule**: Once frozen, no hypothesis, split definition, metric definition or external-test rule in this document may be changed in response to observed results. Any necessary change is recorded in `DECISIONS.md` with its date, rationale, and the analyses it invalidates. Changes made after seeing results are disclosed as such in the manuscript.

---

## 0. Purpose and scope

This protocol governs a systematic evaluation of how the spatial partitioning of training and validation data affects (a) the estimated accuracy of remote sensing semantic-segmentation models, and (b) whether that estimate correctly predicts performance on a genuinely independent geographic test set.

**What this study is not.** It is not a proposal for a new segmentation architecture. No claim of architectural novelty will be made. Model architectures are treated as *subjects of measurement*, not as contributions.

**The causal chain under study:**

```
spatial dependence
   → validation strategy
      → estimated accuracy
         → geographic generalization
            → model-selection reliability
```

**The object of measurement** is therefore the *estimator*, not the model. Every design choice below exists to keep the estimator and the model separable.

---

## 1. Research questions

**RQ1** — Do random splits and spatially independent splits yield systematically different performance estimates for remote sensing AI models?

**RQ2** — How does the performance estimate change as the spatial isolation distance between training and validation data increases?

**RQ3** — Do different model architectures (CNN encoder–decoder, dilated CNN, Transformer) differ in their sensitivity to spatial dependence / spatial leakage?

**RQ4** — Which validation strategy produces a performance estimate most consistent with a truly independent external geographic test?

**RQ5 (extension)** — Does the validation strategy change the *ranking* of models, and therefore change model-selection conclusions?

---

## 2. Pre-registered hypotheses

These are hypotheses. **They are not findings.** Until the corresponding analysis has been executed and reported in `results/`, none of them may be written as an established result, in any document, draft or figure.

| ID | Hypothesis | Falsified if |
|---|---|---|
| **H1** | There is a systematic difference between the cross-validation performance of a random split and performance on the external geographic test. | The estimated difference is not distinguishable from zero under the pre-specified bootstrap CI. |
| **H2** | As the spatial independence between training and validation samples increases, the deviation between the cross-validation estimate and the external geographic test estimate changes systematically. | The deviation shows no monotone or otherwise systematic relationship with separation distance / block size. |
| **H3** | Different model families are affected to different degrees by the spatial partitioning scheme. | The strategy × model interaction is not distinguishable from zero. |
| **H4** | Different validation strategies can produce different model rankings. | Rankings are concordant across all strategies (Spearman ρ ≈ 1, Kendall τ ≈ 1). |

**Explicitly permitted outcomes.** H1 may fail to be supported. If random and spatial splits produce similar estimates, that is a **valid and reportable result**, and the analysis must then investigate why (e.g. weak spatial structure in the data, coarse inter-region separation relative to the dependence range, or insufficient leakage in the random split). The research question is *how partitioning affects the estimate* — not *whether random splitting can be shown to be wrong*. A null result will not be reframed as a failure, and will not be suppressed.

**Directional neutrality.** `OB` (defined in §10) may legitimately be **negative** — a cross-validation estimate can be *pessimistic*. Negative values will be reported as measured. The phrase "accuracy inflation" will not be applied to a negative `OB`.

---

## 3. Task definition and the two distinct inference targets

The protocol requires a distinction that is frequently conflated in the literature and that this study must maintain throughout:

| Target | Question it answers | Legitimate estimator |
|---|---|---|
| **A. Model generalisation / model selection** | Which model should be chosen, and how well does it transfer to new areas? | Spatial cross-validation, leave-one-region-out, external geographic test |
| **B. Final map product accuracy** | What is the accuracy of the map that was actually produced? | Probability sampling of the map, with design-consistent inference |

**Binding consequence:** spatial cross-validation is **not** treated in this study as a ground-truth estimate of map accuracy. It is treated as an estimator of *generalisation* whose relationship to a genuinely independent geographic hold-out is precisely what is being measured. The manuscript will state this distinction explicitly in the Methods and will not describe any cross-validation strategy as yielding "true" map accuracy. This position is consistent with Wadoux et al. (2021) [L037] and is taken as a framing constraint, not as a result.

---

## 4. Dataset

**Primary dataset**: OpenEarthMap (Xia et al., WACV 2023) [L047]. Full verification in `data/metadata/dataset_source_verification.md`.

| Property | Value |
|---|---|
| Images | 5,000 × 1024 × 1024 px |
| Classes | 8 (Bareland, Rangeland, Developed space, Road, Tree, Water, Agriculture land, Building) |
| Regions | 97 nominal; **≤69 obtainable** (28 xBD regions unavailable — see §4.2) |
| Countries | 44 nominal |
| GSD | 0.25–0.5 m |
| Official split | random, within-region, 6:1:3 (3000/500/1500) |
| Licence | heterogeneous; CC BY-NC-SA 4.0 and CC BY 4.0 and public domain (see verification doc) |

### 4.1 Treatment of the official split

The official split is **random and within-region**, therefore **not geographically disjoint**. It is retained in this study **only** as:
- a reference point for reproducing published benchmark behaviour, and
- the definition of the S0 random-split baseline (S0 is rebuilt under the project's own seeding for comparability with the other strategies, and the correspondence to the official split is reported).

It is **never** used as the external geographic test.

### 4.2 Region pool determination (empirical, not assumed)

The region pool is determined by what is actually present in the retrieved archive, enumerated during data QC. It is never taken from the paper's nominal 97, nor from a mirror's documentation. If QC finds fewer than the nominal count, the actual count is what is reported.

xBD-sourced regions are excluded because the xBD RGB imagery is not redistributed with OpenEarthMap and its source is unreachable. The affected count is verified at QC.

### 4.3 External replication dataset (deferred)

LoveDA is designated a **P2** extension and is not started until the OpenEarthMap main experiment is complete and stable. Two datasets will not be processed concurrently. LoveDA's own licence and access route will be verified before use, applying the same standard as §4 of the verification document.

---

## 5. Geographic metadata and the coordinate contingency

The study requires geographic coordinates to compute continuous train–validation distances. **The availability of usable coordinates is not assumed.** The protocol specifies three tiers, and the tier actually achieved is recorded as a limitation:

| Tier | Condition | What is computed | Limitation to disclose |
|---|---|---|---|
| **T1 (preferred)** | Per-image or per-patch geocoordinates are present and valid (plausible CRS, plausible within-country bounds) | True great-circle distances; empirical dependence range `r`; distance-resolved block sizes | none beyond GSD uncertainty |
| **T2** | Only region-level centroids are recoverable | Distances between regions; region-level spatial weights; `r` estimated at region scale | no within-region distance resolution; block sizes coarser than the region scale |
| **T3 (fallback)** | No usable coordinates | Spatial grouping at region / site / source-acquisition level; adjacency from shared source and acquisition campaign | **continuous distance cannot be computed; the dependence range `r` is not estimated**; block-size sensitivity is replaced by grouping-level sensitivity |

**Binding rule:** under T3 the study will **not** claim a distance–similarity relationship, will **not** report a numeric `r`, and will **not** assert that a specific block size is optimal. Any statement of the form "1 km is the appropriate block size" is prohibited unless `r` was actually estimated and its estimation is documented. **Coordinates will not be fabricated, interpolated from region names, or borrowed from an unrelated source.**

The tier achieved is recorded in `data/metadata/spatial_metadata_audit.md` with the evidence for the determination.

---

## 6. External Geographic Test (EXT) — frozen before any training

This is the study's load-bearing design element. It is constructed **before any model is trained** and then frozen.

### 6.1 Selection principles

1. **Whole regions only.** External regions are held out in their entirety. No patch from an external region may appear in any training or validation set, under any strategy, at any stage.
2. **Geographic isolation.** Selected regions are geographically separated from the development pool. Selection prefers regions whose distance to the nearest development region is large.
3. **Continental and landscape coverage.** Selection is stratified so that the external set spans multiple continents and multiple landscape types, rather than concentrating in one biome or one continent.
4. **Class preservation.** The pooled external set must retain all eight land-cover classes at a usable pixel share, so that per-class IoU is computable. A class-coverage constraint is applied as a hard filter.
5. **No performance-based selection.** Regions are **not** chosen or excluded on the basis of any model's accuracy, any preliminary result, or any intuition about difficulty. Only geographic dispersion, continent, and class composition enter the selection.
6. **Deterministic and seeded.** The selection procedure is a deterministic algorithm with a recorded seed, so that it is exactly reproducible. It is executed once, and its output is hashed.

### 6.2 Procedure

1. Assemble the obtainable region pool with continent, country, centroid coordinates (subject to §5 tier), and per-class pixel shares.
2. Hard-filter to regions satisfying the minimum class-presence requirement.
3. Apply continent-stratified selection with a target of **12–15 external regions** (exact count fixed by the algorithm's output on the actual pool, not by preference), maximising the minimum distance from any selected region to any non-selected region.
4. Verify the pooled external set's per-class pixel share against the pooled development set's share; record the divergence. Class-composition divergence is a **disclosed limitation**, not something to be corrected by re-selecting regions.
5. Write `data/metadata/external_regions.json` containing: the selection algorithm and its parameters, the seed, the input pool, the selected regions with their rationale, the class-composition comparison, and a SHA-256 of the file's canonical content.
6. **Freeze.** The file is thereafter read-only. A hash check is part of every downstream QC gate.

### 6.3 Prohibited uses of EXT — leakage rules

After freezing, the external test set may **not** be used for:

- hyperparameter tuning
- early stopping
- architecture selection
- threshold or post-processing optimisation
- augmentation policy selection
- loss-function selection
- any exploratory analysis whose outcome feeds back into a training decision

Violation of any of these constitutes test-set leakage and invalidates the affected results. The QC gate in §13 enforces this mechanically: EXT sample IDs are checked against every training and validation manifest, and any intersection fails the gate.

**Permitted use**: a single, final, one-shot evaluation of the already-trained and already-selected models, performed once, after all training is complete.

### 6.4 Development pool

Everything not in EXT is the **development pool**. All strategy development, hyperparameter selection, early stopping and exploratory analysis use only the development pool. Under S4 (leave-one-region-out), the held-out region is always a development region; external regions are excluded from the S4 rotation entirely.

---

## 7. Spatial dependence analysis (development pool only)

All of §7 is computed **within the development pool only**. Computing it on the full dataset would expose information about external regions before the freeze.

### 7.1 Required analyses

1. **Train–validation nearest geographic distance** — minimum, median and mean of the nearest-neighbour distances from each validation sample to the training set, computed per split.
2. **Class composition similarity** — divergence between train and validation class distributions (per-class pixel share; Jensen–Shannon divergence).
3. **Feature similarity** — distributional distance between train and validation image features. Two complementary representations are used: (a) hand-crafted descriptors (per-channel moments, texture, spectral statistics) and (b) frozen pretrained embeddings. **Pretrained embeddings are computed with weights that are frozen and never fine-tuned**, so no information about external regions enters a trainable component.
4. **Spatial autocorrelation** — Moran's I on image-level features using a k-nearest-neighbour or distance-band spatial weights matrix (`libpysal`), reported per feature and with a permutation-based pseudo p-value.
5. **Semivariogram / correlogram** (conditional on §5 tier) — empirical semivariance and correlation as a function of distance.
6. **Distance–similarity relationship** `S(d)` — similarity of paired samples as a function of their separation. A monotone decreasing `S(d)` supports a dependence structure; its absence is itself a finding.

### 7.2 Estimation of the dependence range `r`

If §5 tier permits, `r` is estimated as the distance beyond which the semivariogram reaches its sill (or at which the correlogram falls below a stated threshold), fitted with a documented parametric model and reported with a confidence interval. **The estimator, its fit quality, and the sensitivity of `r` to the choice of model are all reported.** If the fit is unstable across plausible models, that instability is reported and `r` is treated as a range rather than a point value.

**Implementation constraints established from the data (D-025 … D-027).** These are obligations, not preferences; each was adopted because the alternative was measured to be wrong.

1. **All distances are geodesic.** The corpus spans 98° of latitude. A single-reference-latitude planar approximation overstates east–west distance by 73–81% at high latitude, and that error propagates directly into `r`. Distance statistics use exact great-circle distances; nearest-neighbour queries use a unit-sphere KD-tree (chord length is strictly monotone in great-circle angle, so the search is exact). `src/spatial/geometry.py` is the single authority, and the planar helper **raises** beyond a 1° latitude span.

2. **The curve that drives `r` uses WITHIN-REGION pairs only.** Over all pairs the 99th-percentile separation is 18,382 km, so 20 linear bins are **459.5 km** wide — wider than the phenomenon itself, which then sits entirely inside the first bin. Within-region pairs give a 1.403 km bin width over **49,381** pairs, few enough to enumerate exhaustively. The all-pairs curve is retained as **context only** and is flagged `used_for_range: false`.

3. **The fit window is set by pair support, not by shape.** The empirical semivariance over `n` pairs has relative standard error ≈ `sqrt(2/n)`, so requiring **≥ 500 pairs** bounds per-bin noise at ~6% by a stated rule rather than by a visually chosen distance. A monotone-rise walk was swept over tolerance ∈ {0, 0.02, …, 0.20} × smoothing ∈ {off, 3-bin} and found **degenerate**: for every tolerance ≥ 0.12 it returns exactly the support window, and below that it truncates *inside* a genuine plateau. It is therefore disabled by default.

4. **The fit is weighted by pair count** (`sigma = 1/sqrt(n_pairs)`). Near-origin bins hold ~9,500 pairs and the tail a few hundred, so an unweighted fit effectively fits tail noise.

5. **Identifiability is assessed per model, not only for the selected one.** A fitted range counts as a measurement only if `r < last fitted lag`; otherwise the model never attained its sill inside the window and the value is an **extrapolation**. On this dataset the exponential fit returns 23.377 km from a 17.57 km window and is excluded on that basis; admitting it would inflate the reported spread from 0.20 to 0.93 of the median and wrongly declare the range unstable. All three models use the *effective-range* parameterisation (spherical reaches its sill at `a`; exponential `1 − exp(−3h/r)` and gaussian `1 − exp(−3(h/r)²)` both reach 95% of the sill at `h = r`), so their disagreement is genuine model-form disagreement rather than a parameterisation artefact.

**Result to be reported.** `r = 12.137 km` (median of identified fits; spherical the best fit at R² = 0.962), interval **10.77–13.50 km** across identified models, `range_identified = true`, `stable = true`. The semivariance attains a sill of γ ≈ 6.2 by **~10.5 km** and stays at that level to ~17.6 km under every window and model tested; this is the one statement about the dependence scale that is robust to both choices. Beyond ~19 km γ steps to ~8.4 and oscillates, coinciding exactly with pair support collapsing below 500 and with within-region pairs beginning to cross between sub-landscapes — 25 of 62 development regions contain more than one spatial cluster at 5 km single-linkage (108 clusters in total), and 3.4% of tiles have no same-region neighbour within 2 km. Beyond that point the semivariance is no longer a function of distance alone, and the value is not a dependence scale.

### 7.3 Consequence for block sizes

Block sizes are **derived from `r`**, at `0.5r`, `1r` and `2r`. If `r` cannot be estimated (T3), the protocol falls back to multiple grouping scales justified by the study-area extent, and the fallback is stated explicitly with the reasoning. Under no circumstance is a block size asserted as "most appropriate" without the analysis to support it.

**Two binding qualifications.**

1. **The ladder is a sensitivity range, not a fixed optimum.** Because the range interval spans a factor of 1.25 (10.77–13.50 km), the block sizes are **6.07 / 12.14 / 24.27 km** with the interval implying 5.39–6.75 / 10.77–13.50 / 21.55–27.00 km. No block size may be described as optimal, and no single scale may be reported without the ladder.

2. **A fallback that is not derived from `r` must be refused, not invented.** Block sizes are read from `data/metadata/spatial_dependence.json` (`range.r_km`). If no range is available *and* no fallback is explicitly configured, the split stage **fails**. Substituting study-extent scales silently would let the manuscript describe S2/S3 as "derived from the dependence range" when they were not — which would defeat RQ2, the study's central question (D-030).

**Measured justification for S3 (buffering), from the built designs.** Block size alone does not control the minimum train–validation separation:

| design | min train–val km | median km | same-region contamination |
|---|---|---|---|
| S0 random | 0.099 | 0.435 | 1.000 |
| S1 group (region) | 29.383 | 318.674 | 0.000 |
| S2 block 0.5r / 1r / 2r | 0.167 / 0.167 / 0.255 | 2.54 / 5.52 / 22.98 | 0.87 / 0.79 / 0.63 |
| S3 buffered 0.5r / 1r / 2r | **6.071 / 12.154 / 24.333** | 18.29 / 94.57 / 223.57 | 0.62 / 0.35 / 0.14 |
| S4 leave-one-region-out | 7.060 | 220.133 | 0.000 |

S2 keeps the minimum separation at **0.17–0.26 km regardless of block size**, because block boundaries can still leave training and validation blocks adjacent; S3's minimum equals the buffer exactly, which directly verifies that the buffer is applied as specified. This contrast is a result of the study, not merely a design detail.

---

## 8. Validation strategies

| ID | Strategy | Definition |
|---|---|---|
| **S0** | Random patch split | Class-stratified random partition of patches. Reproduces the conventional practice. |
| **S1** | Group / scene split | Patches sharing a scene, source group or spatial adjacency group cannot straddle train and validation. |
| **S2** | Spatial block cross-validation | Spatial blocks constructed from coordinates; block size from `r` (or the documented fallback). Tested at `0.5r`, `1r`, `2r`. |
| **S3** | Buffered spatial cross-validation | As S2, plus a buffer around each validation block; samples inside the buffer are **excluded from training** (not merely relabelled). Buffer widths tested at `0.5r`, `1r`, `2r`. |
| **S4** | Leave-one-region-out | Each development region is held out in turn as validation. External regions excluded from the rotation. |
| **EXT** | External geographic test | The frozen external regions. Evaluated once, at the end. |

### 8.1 Split output format

Every split is written to `splits/*.csv` with **at least** these columns:

```
sample_id, region, country, x, y, split, fold, strategy, block_id, distance_to_training
```

- `split` ∈ {`train`, `val`, `test`}
- `distance_to_training` = nearest distance from this sample to any training sample (null for training samples; for S3, computed to the *unbuffered* training set so the buffer's effect is measurable separately)
- `x`, `y` carry the actual coordinate tier used, with the tier recorded in the file header comment or a companion JSON

### 8.2 Matched sample size — mandatory control

Spatial blocking and buffering reduce the effective training sample count. The protocol requires that the alternative explanation — *"performance changed only because there were fewer training samples"* — be ruled out rather than assumed away.

**Procedure:** for every spatial strategy whose training count `N2` is less than the random-split count `N1`, construct a **Random-Matched-`N2`** split: a random class-stratified draw of exactly `N2` training samples. Where feasible, **multiple independent matched draws** are generated so that the result does not hinge on one draw, and the mean and spread across draws are reported.

**Interpretation rule:** a difference between a spatial strategy and Random-Matched-`N2` is attributable to spatial independence rather than to sample count. A difference that disappears under matching is attributable to sample count. Both outcomes are reportable.

### 8.3 Comparability rule

Every strategy uses the **same validation sample set where the design permits**, so that differences in the estimate reflect the training–validation relationship rather than a change of evaluation data. Where this is impossible (S4 holds out whole regions by construction), the difference is stated and the comparison is framed accordingly.

---

## 9. Spatial leakage diagnostics

Independence is **measured, not asserted from the strategy's name**. For every split, the following are computed:

| Diagnostic | Definition |
|---|---|
| Minimum train–validation distance | nearest training sample to any validation sample |
| Median nearest train–validation distance | per-validation-sample nearest distance, median |
| Mean nearest train–validation distance | as above, mean |
| Adjacent-patch proportion | share of validation patches with a training patch sharing a border |
| Duplicate proportion | share of validation samples with an exact SHA-256 match in training |
| Near-duplicate proportion | share of validation samples with a perceptual-hash match within threshold in training |
| Same-scene contamination | share of validation samples whose scene appears in training |
| Same-region contamination | share of validation samples whose region appears in training |

Output: **Spatial Independence Diagnostics Table** (`results/tables/spatial_independence_diagnostics.csv`).

**Interpretation rule:** no claim of spatial independence may rest on the strategy label. A strategy named "spatial block" whose diagnostics show a median nearest distance of a few metres is not spatially independent, and the diagnostic table — not the name — is the evidence.

---

## 10. Models

Three models, chosen to span the architectural families named in RQ3. Architecture is not a contribution.

| ID | Model | Family | Rationale |
|---|---|---|---|
| **M1** | U-Net + ResNet-34 encoder | CNN encoder–decoder, skip connections | the field's default; strong local inductive bias |
| **M2** | DeepLabV3+ | dilated / atrous CNN with ASPP | wide effective receptive field without pooling loss |
| **M3** | SegFormer-B0 | Transformer (hierarchical, attention) | global attention; weakest spatial inductive bias |

**Controlled across models** (differences here would confound RQ3, so they are held constant):

- input resolution and crop size
- augmentation policy
- number of epochs and early-stopping rule
- optimiser family, learning rate, schedule, weight decay
- loss definition
- pretraining policy (ImageNet-1k initialisation for all three)
- evaluation pipeline and metric implementation
- mixed-precision policy

**Unavoidable differences, recorded explicitly:** encoder parameter count; the need for a different decoder head; and any model-specific normalisation. These are recorded per run in the experiment registry so that RQ3's interpretation can account for them. **Model capacity differences are a confounder for RQ3 and will be stated as such rather than corrected by post-hoc tuning.**

---

## 11. Training protocol

| Item | Specification |
|---|---|
| Seeds | **5** (minimum 3; 5 targeted) |
| Determinism | global seed for `random`, `numpy`, `torch`; `PYTHONHASHSEED`; cuDNN deterministic flags where compatible with performance |
| Optimiser | AdamW, weight decay fixed across models |
| Schedule | cosine decay with linear warmup; identical across models |
| Loss | identical composite loss across models |
| Batch size | fixed across models (subject to VRAM); if VRAM forces a reduction for one model, the reduction is recorded and the affected comparison is flagged |
| Epochs | fixed budget with early stopping on the strategy's own validation split |
| Early stopping | identical patience and monitored metric across models |
| Checkpointing | best-by-validation-metric, plus final-epoch checkpoint |
| Precision | AMP if GPU available; recorded per run |

### 11.1 Run provenance

Every run persists: config, seed, git commit, split ID, dataset version, checkpoint path, training log, best epoch, validation metrics. Runs are **append-only** — an existing experiment is never overwritten.

**`experiment_id` scheme:**
```
{dataset}-{strategy}-{strategy_param}-{model}-{seed}
e.g.  OEM-S2-b1.0r-unetr34-s42
      OEM-S0-na-segformerb0-s17
      OEM-EXT-na-deeplabv3p-s07
```

**`experiments_registry.csv` columns (minimum):**
```
experiment_id, dataset, split_strategy, block_size, buffer_size, model, seed,
train_samples, val_samples, epochs, best_epoch, checkpoint, status, notes
```

### 11.2 Mandatory smoke test before any full training

Full-batch training is **prohibited** until all of the following pass on a small subset (~50 images):

1. ~50-image debug run
2. single-batch forward pass
3. single-batch backward pass
4. one complete epoch
5. prediction visualisation saved and inspected
6. metric sanity check against a hand-computed reference

Confirmed: image–mask correspondence, label encoding, loss decreasing, metric function correctness, spatial alignment of predictions against input. Results recorded in `logs/smoke_test_report.md`. Training proceeds only on PASS.

---

## 12. Metrics

### 12.1 Performance metrics

- **Primary: mIoU**
- Macro F1, Overall Accuracy
- Per-class IoU, Precision, Recall, F1

Overall accuracy is **never** reported alone, because class imbalance (Bareland 1.5 % vs Rangeland 22.9 %) makes OA misleading.

### 12.2 Estimator-evaluation metrics

Let `CV_metric` be the performance estimated by a validation strategy on its own validation data, and `External_metric` the performance on the frozen external geographic test.

| Metric | Definition | Interpretation |
|---|---|---|
| **OB** — Optimism Bias | `CV_metric − External_metric` | signed; **positive = optimism, negative = pessimism**; negative values reported as measured |
| **AEE** — Absolute Estimation Error | `|CV_metric − External_metric|` | magnitude of error regardless of direction |
| **GG** — Generalization Gap | `Validation_metric − External_metric` | `Validation_metric` is the score on the selection validation set used for early stopping. For hold-out strategies GG coincides with OB; for k-fold strategies they differ, because `CV_metric` is fold-averaged while `Validation_metric` is a single split. **This distinction is reported explicitly wherever both appear.** |
| **RI** — Relative Inflation | `(CV_metric − External_metric) / External_metric × 100 %` | scale-free; undefined and not reported when `External_metric` is zero |

All four are computed per model, per strategy, per seed. `RI` is reported with the caveat that it is unstable when `External_metric` is small.

---

## 13. Statistical analysis

### 13.1 Uncertainty

- mean, standard deviation
- **bootstrap confidence intervals** (resampling unit chosen to respect the data structure: **regions** for geographic strategies, patches for random splits) — the resampling unit is stated for every interval
- effect sizes, reported alongside every interval

### 13.2 Modelling

Where the data structure supports it, a mixed-effects model is fitted:

```
Metric ~ Strategy + Model + Strategy×Model + (1 | Region)
```

- Random intercept for region, because samples within a region are not independent.
- Model assumptions checked (residual normality, homoscedasticity, random-effect variance). Diagnostics reported.
- If assumptions are not met, a non-parametric or bootstrap alternative is used instead, and the switch and its reason are reported.

### 13.3 Ranking stability (RQ5)

For each strategy, models are ranked by primary metric. Concordance with the external-test ranking is quantified by **Spearman ρ** and **Kendall's τ**. Rank reversals are reported and examined individually.

**Directional neutrality:** the analysis does not presuppose that rank reversal occurs. If rankings are concordant, that is the finding.

### 13.4 Multiplicity and analytic integrity

- **One primary contrast is pre-specified**: S0 versus EXT on mIoU. It is tested without multiplicity adjustment.
- All other comparisons are **secondary and exploratory**. Where a family of tests is run, false-discovery-rate control (Benjamini–Hochberg) is applied and the family is defined in advance.
- **Prohibited**: iterating the analysis until `p < 0.05`; changing the analysis to obtain significance; dropping seeds, regions or strategies after seeing results; reporting only favourable block sizes.
- Every analysis script is committed with its output so the analysis path is auditable.

---

## 14. Figures and tables (planned)

| Figure | Content |
|---|---|
| F1 | Overall research framework |
| F2 | Geographic distribution of the dataset and the external regions |
| F3 | Illustration of the different spatial splitting strategies |
| F4 | Spatial similarity / autocorrelation as a function of geographic distance |
| F5 | mIoU and F1 under different validation strategies |
| F6 | Optimism bias / estimation error |
| F7 | Performance and estimation error vs spatial separation or block size |
| F8 | Model ranking under cross-validation vs external geographic test |

Supplementary: confusion matrices; per-class IoU; all random seeds; all hyperparameters; additional block sizes; additional regions; training curves.

**Figure standards:** high resolution (vector or ≥300 dpi raster), consistent fonts and terminology, legible legends, directly usable in a journal. **No unfavourable result may be removed for appearance.** Figures are generated by code from files in `results/`; no number is typed into a figure by hand.

---

## 15. Ablation / experiment priority ladder

Resources are allocated strictly in this order. **P0 must be complete before P1 is started**, and the completed subset must remain a coherent, reportable study if execution is truncated.

| Priority | Requirement |
|---|---|
| **P0** | Random vs Spatial Block vs Leave-Region-Out vs External |
| **P0** | At least 3 models |
| **P0** | Multiple seeds (≥3, target 5) |
| **P0** | Spatial independence diagnostics |
| **P0** | Matched sample size control |
| **P1** | Multiple block sizes |
| **P1** | Buffer distance sensitivity |
| **P1** | Ranking stability analysis |
| **P2** | LoveDA replication |
| **P2** | Additional foundation model |

**Resource rule:** when resources are limited, protect P0 rather than adding models.

---

## 16. Quality-control gates

Each gate outputs PASS / PARTIAL / FAIL and is run as code, not by inspection.

| Gate | Check |
|---|---|
| **Data QC** | image/mask counts, correspondence, dimensions, channels, label range, missing/corrupt files, NoData, class/region/country distributions, duplicate and near-duplicate detection |
| **Split QC** | train / val / test sample IDs are mutually disjoint; fold coverage complete; every sample assigned exactly once |
| **External QC** | no external region ID appears in any training or validation manifest, for any strategy, any seed; `external_regions.json` hash unchanged since freeze |
| **Experiment QC** | every run has a traceable config, seed, git commit, split ID, dataset version and checkpoint |
| **Statistics QC** | sample counts consistent; no sample double-counted; bootstrap resampling unit matches the stated unit |
| **Manuscript QC** | every number in the manuscript traces to a file in `results/`; no hand-typed experimental number |

**Manuscript QC is a hard requirement.** Any number in the manuscript that cannot be traced to a machine-generated artefact in `results/` must be removed.

---

## 17. Reproducibility

- Environment frozen (`requirements.txt`, `environment.yml`); verification script `scripts/check_env.py`
- Config-driven execution; no hard-coded absolute paths in code (paths resolved from a project-root setting)
- Seeds recorded per run
- `experiments_registry.csv` append-only
- `PROGRESS.md` — append-only log in the format: Date / Task / Input / Method / Output / Problems / Decision / Next step
- `DECISIONS.md` — every methodological change with rationale, including why a dataset, block size or region was chosen or excluded
- All random processes seeded; all splits written to disk and hashed

---

## 18. Known limitations to be disclosed (pre-registered)

Declared now, before results, so that they cannot be selectively introduced later:

1. **Single primary dataset.** OpenEarthMap only, unless the P2 LoveDA replication is reached. Cross-dataset generalisation is therefore not established by this study.
2. **xBD regions missing** — up to 28 of 97 regions unobtainable; geographic coverage is reduced, and the reduction is uneven across continents.
3. **Mirror-acquired data.** The canonical Zenodo artefact was unreachable; hash verification establishes transport integrity, not cross-archive equivalence.
4. **Coordinate tier uncertainty.** If only region-level coordinates are recoverable (T2/T3), continuous distance resolution is limited and the dependence range may be inestimable; the study's distance-resolved claims would then be correspondingly weakened.
5. **Class imbalance.** Bareland at 1.5 % pixel share makes its per-class IoU unstable; per-class results for rare classes will be reported with intervals and treated cautiously.
6. **Model capacity confound.** The three architectures differ in parameter count, which is not fully separable from architecture family in RQ3.
7. **Single sensor family / GSD range** — 0.25–0.5 m aerial and satellite imagery; conclusions may not transfer to coarser sensors.
8. **Label uncertainty.** The dataset's own reported inter-annotator agreement is 78 % for pixel-level class agreement (per the original paper), which bounds achievable metric precision. This is a property of the data, and it caps how finely differences between strategies can be interpreted.
9. **No claim about map-product accuracy.** As established in §3, this study evaluates estimators of generalisation, not the accuracy of a delivered map product.

---

## 19. Manuscript structure (target)

Generic SCI structure, used because the target journal is not yet fixed.

```
Title
Abstract
Keywords
1 Introduction
2 Related Work
3 Materials and Methods
  3.1  Study design
  3.2  Dataset
  3.3  Data quality control
  3.4  Geographic external test design
  3.5  Spatial dependence analysis
  3.6  Validation strategies
  3.7  Models
  3.8  Training protocol
  3.9  Performance metrics
  3.10 Optimism / generalisation metrics
  3.11 Statistical analysis
  3.12 Reproducibility
4 Results
  4.1  Dataset and spatial dependence characteristics
  4.2  Effect of partitioning strategy on estimated accuracy
  4.3  Effect of spatial separation / block size
  4.4  Model-specific sensitivity
  4.5  Comparison with external geographic performance
  4.6  Model ranking stability
  4.7  Matched-sample-size experiment
5 Discussion
6 Conclusions
Data Availability
Code Availability
Author Contributions (placeholder)
Funding (placeholder)
Conflict of Interest (placeholder)
References
Supplementary Materials
```

**Writing rules**: formal academic English; no marketing language (`revolutionary`, `groundbreaking`, `perfect`, `unprecedented`) absent overwhelming evidence; no `obviously`, `clearly`, `it is well known`; every conclusion evidenced; negative results not hidden; mechanism explanations clearly separated from reported facts. Claims about the literature are limited to what the verified reference base supports — the phrase "few studies exist" is prohibited unless the search actually demonstrates it.

**Contributions will be written to match what was actually found.** Candidate contributions (C1 systematic quantification of partitioning effects on accuracy estimates; C2 use of a genuinely independent geographic hold-out as an external benchmark rather than treating a CV method as ground truth; C3 the relationship between spatial isolation scale and estimation error; C4 the effect of validation strategy on model ranking and selection) are **candidates only** and will be rewritten or dropped to match the results.

---

## 20. Phase acceptance template

Every phase reports:

```
[PHASE]     phase name
[STATUS]    PASS / PARTIAL / FAIL
[COMPLETED] what was done
[FILES]     files produced
[QC]        quality-control result
[ISSUES]    problems
[NEXT]      next action
```

PASS → continue automatically. PARTIAL without blocking → record and continue. FAIL and blocking → state the cause, the measures already attempted, and the available paths forward. **Failures are never obscured with vague language.**

---

## 21. Final acceptance conditions

"Manuscript draft complete" may be declared only when **all** of the following hold:

- [ ] Data provenance traceable
- [ ] Data licence recorded
- [ ] Data QC complete
- [ ] External geographic test genuinely independent
- [ ] Random split executed
- [ ] Spatial split executed
- [ ] Region-level validation executed
- [ ] At least 3 models
- [ ] At least 3 seeds
- [ ] Matched sample size control complete
- [ ] Spatial independence diagnostics complete
- [ ] External test complete
- [ ] mIoU / F1 / per-class metrics complete
- [ ] Optimism / generalisation metrics complete
- [ ] Statistical uncertainty complete
- [ ] All core figures generated by code
- [ ] All Results numbers traceable
- [ ] All references verified
- [ ] Limitations complete
- [ ] README supports reproduction
- [ ] Complete `manuscript_v1` produced

If any of these is unmet, the unmet items are listed explicitly and the manuscript is **not** described as complete.

---

## 22. Protocol freeze record

| Field | Value |
|---|---|
| Version | 1.0 |
| Frozen | 2026-09-17 |
| Frozen before any model training | **yes** |
| External regions defined before any training | yes (at freeze of `external_regions.json`) |
| Hypotheses registered before any training | yes (§2) |
| Metric definitions fixed before any training | yes (§12) |
| Statistical plan fixed before any training | yes (§13) |

**Change control.** Any post-freeze modification is appended to `DECISIONS.md` with date, rationale, and the list of analyses invalidated by the change. Modifications made after results were observed are labelled as post-hoc in the manuscript.

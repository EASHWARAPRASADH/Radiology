# 13 — Prototype Implementation Plan (Phase 1) and Results

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

Phase 1 delivers a runnable, tested chest X-ray pipeline in this repository. It proves the *system design*: the tri-state IR, calibrated deferral, abstention, grounded reporting, review workflow, standards exports, audit and licence gate. It also gives a first feasibility signal on commercially usable public data. It does **not** establish clinical performance.

## 1. What is implemented

| Pipeline stage (06) | Module | Status | Notes |
|---|---|---|---|
| 1 Ingest + validate | `ingest/dicom.py` | Done | CR/DX only; frontal PA/AP only; VOI/modality LUT; MONOCHROME1 inversion; PHI-tag detection. PNG/JPEG with asserted metadata |
| 2–3 Preprocess | `preprocess/transforms.py` | Done | Letterbox with invertible geometry (boxes map back to source pixels). **No flips** |
| 4a Quality gate | `qc/quality.py` | Done | Classical checks (resolution, blank, saturation, exposure, aspect, AP caveat). `non_diagnostic` → abstain |
| 4b OOD gate | `qc/ood.py` | Done | Mahalanobis (Ledoit-Wolf) on encoder embeddings; threshold at the 99.5th percentile of validation → abstain |
| 5–6 Encoder + classifier | `models/encoder.py`, `models/classifier.py` | Done | DINOv2 ViT-S/14 (Apache-2.0) + linear head on mean patch tokens. Exact per-class activation maps |
| 7 Localisation | `models/detector.py`, `pipeline/run.py` | Interface + CAM boxes | Activation-map boxes stored `validated=false`. Faster R-CNN builder ready for Phase 2 |
| 9 Calibration | `calibrate/temperature.py`, `calibrate/conformal.py` | Done | Per-label temperature. FN-controlled `t_absent` and FP-controlled `t_present`. Labels with too few calibration positives (< 30) or negatives (< 100) → `not_assessed` |
| 10 Normalisation | `ontology/cxr.py` | Done | 17 internal concepts; NIH and ChestX-Det label maps; RadLex / SNOMED IDs deliberately empty until verified |
| IR | `schema/findings.py` | Done | Pydantic v2 with cross-field invariants; JSON Schema exported |
| 11 Report generation | `report/generator.py` | Done | Deterministic templates. Optional `Rephraser` hook whose sentences are re-validated individually |
| 12 Consistency check | `report/validator.py` | Done | 18 issue codes covering grounding, status ↔ sentence kind, unsupported mention, negation polarity, hedging, laterality, numbers, comparison without prior, normal-summary preconditions, omissions, critical finding in impression, not-assessed disclosure, guideline-backed recommendations, disclaimer |
| 13–14 Review + approval | `schema/findings.py::apply_review`, `api/app.py` | Done (API) | Radiologist-only approve / amend / reject. Plain approve is blocked if validation failed. Immutable AI draft. OHIF UI is Phase 5 |
| Exports | `export/dicom_sr.py`, `export/fhir.py` | Done | TID 1500 Comprehensive SR (PARTIAL/UNVERIFIED until approved); FHIR R4B `DiagnosticReport` (preliminary → final) |
| Audit | `audit/log.py` | Done | Hash-chained JSONL; tamper test |
| Licence gate | `data/registry.py`, `train/train_classifier.py` | Done | `purpose: product` refused unless every dataset and the weight initialisation is OK. Writes `lineage.json` and MLflow tags if MLflow is installed |
| Modality plug-ins | `pipeline/analyzer.py` | Interface | `Analyzer` protocol + `CXRAnalyzer` adapter + router |
| Metrics | `eval/metrics.py`, `eval/report_metrics.py` | Done | AUROC, AUPRC, ECE, Brier, deferral, sensitivity counting deferrals, patient-level bootstrap CIs; report-level P/R, hallucination, contradiction, laterality error, edit distance |

**Tests:** `uv run pytest` runs 35 tests in about 20 s on CPU with no downloads. They cover:
- schema invariants;
- report grounding, including a 300-case randomised property test that no positive sentence mentions an un-asserted finding, plus an adversarial rephraser;
- validator rules;
- patient-disjoint splits;
- the licence gate;
- exact activation-map decomposition for CNN and ViT;
- end-to-end pipeline and abstention paths (OOD, CT, lateral view, blank image);
- API review flow with role checks;
- SR and FHIR exports;
- audit tamper detection;
- empirical conformal error control.

## 2. Phase-1 experiment (this Mac)

| Item | Setting |
|---|---|
| Data | NIH ChestX-ray14 batch `images_001` (4,999 frontal PNGs) from the official NIH Box, plus `Data_Entry_2017_v2020.csv`. Licence verdict OK (04) |
| Split | Test = NIH official `test_list.txt` patients in this batch. Validation = 25% of the remaining patients by salted hash. Train = the rest. All patient-disjoint (asserted in code) |
| Model | DINOv2 ViT-S/14 (self-supervised pretraining on the LVD-142M web-image set), 224 px input (16×16 patches), linear head, masked BCE, AdamW (encoder 3e-5, head 1e-3), cosine schedule, 8 epochs, bf16 autocast on MPS |
| Augmentation | Affine ±7°, ±4% translate, 0.92–1.08 scale; brightness / contrast ±15%; no flips |
| Calibration | Validation split: per-label temperature, α_FN = α_FP = 0.05, min 30 positives / 100 negatives |
| Config | `configs/classifier_nih_mac.yaml` |
| Measured throughput | 224 px ≈ 64 img/s training on MPS; 448 px ≈ 7 img/s (so 448 px is reserved for cloud) |

### 2.1 Results (measured 2026-11-25, `runs/nih_dinov2s_224`)

**Run:**

| Item | Value |
|---|---|
| Split (images) | train 3,095 / val 937 / test 967 |
| Split (patients) | train 983 / test 62; all patient-disjoint |
| Training | 8 epochs, ≈ 65 s/epoch on MPS, ≈ 10 min total |
| Validation mean AUROC by epoch | 0.61 → 0.73, still rising at epoch 8, so the run is under-trained |
| Licence lineage | `research_only=false`: NIH CXR14 (OK) + DINOv2 (OK) |

**Held-out test set** (NIH official test patients in batch 1; 300 patient-level bootstrap resamples):

| Finding | Test positives | AUROC [95% CI] | AUPRC | ECE | Calibrated? | Deferred | Positives flagged (present or uncertain) |
|---|---|---|---|---|---|---|---|
| atelectasis | 125 | 0.57 [0.49–0.63] | 0.17 | 0.039 | yes | 81% | 100% |
| cardiomegaly | 94 | 0.65 [0.55–0.74] | 0.16 | 0.077 | no (13 val positives) → not_assessed | — | — |
| pleural_effusion | 150 | 0.71 [0.64–0.78] | 0.28 | 0.044 | yes | 78% | 99% |
| lung_opacity | 216 | 0.60 [0.54–0.65] | 0.31 | 0.035 | yes | 70% | 97% |
| mass | 42 | 0.46 [0.26–0.61] | 0.04 | 0.015 | no (20 val positives) → not_assessed | — | — |
| nodule | 35 | 0.60 [0.45–0.73] | 0.05 | 0.025 | yes | 92% | 100% |
| pneumonia | 18 | 0.59 [0.42–0.75] | 0.06 | 0.007 | no (14 val positives) → not_assessed | — | — |
| pneumothorax | 116 | 0.62 [0.51–0.75] | 0.18 | 0.095 | no (15 val positives) → not_assessed | — | — |
| consolidation | 62 | 0.65 [0.56–0.73] | 0.10 | 0.019 | yes | 81% | 97% |
| edema | 35 | 0.81 [0.61–0.93] | 0.19 | 0.017 | no (8 val positives) → not_assessed | — | — |
| emphysema | 42 | 0.56 [0.44–0.67] | 0.05 | 0.029 | no (13 val positives) → not_assessed | — | — |
| fibrosis | 30 | 0.79 [0.69–0.87] | 0.11 | 0.003 | yes | 75% | 100% |
| pleural_thickening | 58 | 0.61 [0.53–0.68] | 0.09 | 0.027 | no (29 val positives) → not_assessed | — | — |
| hernia | 15 | 0.61 [0.36–0.93] | 0.03 | 0.010 | no (4 val positives) → not_assessed | — | — |

Mean test AUROC across all 14 labels: **0.63** (967 images, 62 patients).

**What this shows:**
- **The system behaves as designed when the model is weak.** Only 6 of 14 findings met the calibration minimum, and the other 8 are emitted as `not_assessed` rather than guessed. Where the model's discrimination is weak, the conformal thresholds open a wide `uncertain` band (70–92% deferred). That band keeps 97–100% of true positives flagged for the radiologist instead of calling them `absent`.
- **Discrimination is far below a usable level.** That is expected with about 3k training images, NLP labels, 224 px input and a model still improving. It is the motivation for Phase 2 (more data, 448–518 px on cloud GPUs, continued SSL). Pneumothorax (116 test positives but only 15 in validation) shows that the test patients differ from the validation patients. NIH's official test list is enriched with heavily imaged patients, which is another reason for site-matched calibration (08 §1).
- **End-to-end checks on real and synthetic inputs:**

| Input | Result |
|---|---|
| NIH test image `00000013_021.png` (NIH label: effusion), as PNG and wrapped as DICOM | Draft lists effusion as *possible* (deferred). It calls a nodule present, which NIH does not label and is likely a false positive. 11 concepts are disclosed as not assessed. Validation passes |
| Synthetic phantom DICOM | **Abstained** (OOD Mahalanobis 61.9 > threshold 34.7) |
| Live API (`radio serve`) | 401 without auth; 403 for a technologist approving; amend → approve by a radiologist; 409 on re-approval; FHIR `preliminary` → `final`; SR `PARTIAL/UNVERIFIED` → `COMPLETE/VERIFIED`; audit chain valid across 12 events |

### 2.2 How to read these numbers

- **Not comparable to the literature.** One NIH batch (about 4% of the dataset) and a test set of only a few dozen patients give wide CIs. NIH labels are NLP-mined, the "No Finding" label does not mean normal, and published NIH benchmarks use the full dataset.
- **Deferral is a feature.** `uncertain` findings go to the radiologist. The clinically relevant miss rate is the share of true positives called `absent`, which `t_absent` controls at α_FN on the calibration distribution.
- **`not_assessed` is expected.** On this small subset most labels lack 30 calibration positives, so the system declines to output them rather than guess.

## 3. Phase-2 work packages (localisation), with exit criteria

| WP | Task | Exit criterion |
|---|---|---|
| 2.1 | ChestX-Det loader (boxes + masks, Apache-2.0) → `data/chestx_det.py`; map classes via `CHESTX_DET_LABEL_MAP` | Patient-disjoint from NIH CXR14 test patients (same image source: dedupe by image index) |
| 2.2 | Train the Faster R-CNN baseline (`models/detector.build_faster_rcnn`) on cloud GPU; then RT-DETR / D-FINE (Apache) | FROC at 0.5 / 1 / 2 FP per image reported with CIs (08 §2) |
| 2.3 | Detector-score calibration + fusion with the classifier (06 stage 8) | ECE not worse than the classifier alone |
| 2.4 | Laterality from box centroid vs. segmented midline (nnU-Net 2D lungs / heart) | Laterality accuracy ≥ proposal in 08 §8 on boxed test cases |
| 2.5 | Flip `Localization.validated=true` and `Provenance.image_observed` only for the validated detector; cap focal findings without localisation at `uncertain` | Schema tests updated; validator laterality rules exercised with real laterality |
| 2.6 | Pointing-game / IoU evaluation of activation maps vs. ChestX-Det boxes | Decide whether maps stay display-only (expected) |
| 2.7 | Continued self-supervised pretraining (DINOv2 recipe) on all OK-licensed CXRs (NIH full + MIDRC once signed) on cloud | Linear-probe AUROC gain over vanilla DINOv2 on the internal validation split |
| 2.8 | Persist the prototype store in PostgreSQL per 12-database-design.md; move inference to a queue worker | API contract unchanged (OpenAPI diff empty except additions) |

## 4. Phase 2 (localisation): implementation and results

### 4.1 Implemented

| WP | Module / command | Status |
|---|---|---|
| 2.1 ChestX-Det loader | `data/chestx_det.py`; `scripts/fetch_chestx_det.sh` (archive validation before extraction) | Done |
| 2.1 Overlap / leakage audit | `radio dedupe-chestx-det`, `radio leak-check-chestx-det` | Done. Findings in §4.2 |
| 2.2 Detector | `models/detector.py` (Faster R-CNN R50-FPN v2, BSD-3), `train/train_detector.py`, `configs/detector_chestxdet.yaml` | Done; results in §4.3 |
| 2.2 Metrics | `eval/detection.py`: mAP@0.5 / @0.25, FROC at 0.25/0.5/1/2 FP per image (hit = IoU ≥ 0.25), pointing accuracy of true positives | Done (unit-tested on hand-computed cases) |
| 2.3 Operating points | Per-class box-score threshold chosen on the validation split: 1 FP/image for focal findings, **0.25 FP/image for whole-image findings** (cardiomegaly, emphysema: at most one per image). Each class is gated at the same rate (`global_concepts`, `global_fppi`; see §6.3.2) | Done |
| 2.5 Gating + fusion | Per-class gate on the **leak-reduced** test split (`radio regate-detector`): the **lower end of a 95% image-level bootstrap CI** (2,000 replicates) for FROC sensitivity at 1 FP/image must be ≥ 0.80, with ≥ 30 reference boxes (`gate_statistic: lower_bound`; `point` reproduces the original point-estimate gate). Only gated classes create `image_observed` findings with `validated=true` boxes | Done |
| 2.5 Fusion rules (`pipeline/run.py`) | Validated box + classifier `absent` → `uncertain`. Classifier `present` without a validated box for a focal finding → capped at `uncertain`. Detector-only concepts (fracture, calcification): validated box → `present`, otherwise `not_assessed` (0.80 sensitivity cannot rule out). Concept the classifier cannot assess (no calibration data) but a gated detector boxed → `uncertain`, `image_observed`, never `present`. Added in Phase 3 (pipeline v0.3); without a box it stays `not_assessed` | Done, tested end to end |
| 2.6 Activation-map audit | `eval/pointing.py`, `radio eval-pointing` | Done. Findings in §4.2 |
| 2.4 Laterality from anatomy | — | **Blocked.** Needs licence-clean lung/heart masks (Montgomery/JSRT terms unverified). Laterality stays `unknown` |
| 2.7 Continued SSL | — | **Deferred to cloud** (14) |
| 2.8 PostgreSQL + queue | — | Deferred to Phase 5 (no change to the API contract) |

### 4.2 Data findings (measured 2026-11-25)

**ChestX-Det overlaps heavily with our NIH data.**
- 950 of 3,578 ChestX-Det images are pixel-identical to images in NIH batch `images_001`. That is 40 of 40 sampled matches at 256 px, with correlation 1.000 and mean absolute difference 0, against a median of 0.58 for random pairs.
- 782 of those copies are NIH images the Phase-1 classifier trained or calibrated on. They are excluded from classifier-side evaluation on ChestX-Det.

**The official ChestX-Det split is not patient-disjoint.**
- 139 of the 553 test images map to NIH patients. 111 of those belong to 73 patients who also appear in ChestX-Det train.
- The 414 unmapped test images can't be checked, so the true leak is likely larger.
- All gating therefore uses the test split minus the 111 known-leaky images. Leaky-only metrics are reported alongside so the inflation is visible (16 R26).

**Classifier vs radiologist labels.** On ChestX-Det test images the classifier never saw (444 images), its image-level AUROC against **radiologist** labels is:

| Finding | AUROC | Positive images |
|---|---|---|
| Pleural effusion | 0.81 | 219 |
| Consolidation | 0.77 | 243 |
| Emphysema | 0.71 | 30 |
| Pneumothorax | 0.68 | 34 |
| Cardiomegaly | 0.67 | 62 |
| Pleural thickening | 0.65 | 71 |
| Fibrosis | 0.64 | 58 |
| Atelectasis | 0.59 | 47 |
| Mass | 0.54 | 27 |
| Nodule | 0.50 | 95 |

For effusion, this is higher than the 0.71 measured on NIH's NLP-mined labels, which is consistent with label noise depressing the NIH scores. ChestX-Det is abnormality-enriched, so these numbers are not prevalence-representative.

**Activation maps do not localise.** The pointing game (map peak inside a radiologist box) gives a mean hit rate of 0.11 against a chance rate of 0.10, over classes with ≥ 10 lesion images. Effusion is 0.02 against 0.10. Visual audit (`runs/cam_effusion_debug.png`, not committed) shows peaks on image borders and corners. This is consistent with DINOv2 high-norm artifact tokens (Darcet et al., *Vision Transformers Need Registers*, ICLR 2024). Activation maps remain display-only and `validated=false`. Next encoder: DINOv2 with registers (16 R27).

### 4.3 Detector results

Measured 2026-11-25. Run `runs/chestxdet_frcnn_512` (not committed): Faster R-CNN R50-FPN v2, ImageNet-initialised ResNet-50 backbone, 512 px, 8 epochs, batch 4, MPS on an M4 Pro, 2.0 h in total (about 13–17 min per epoch).

- **Split:** ChestX-Det official train 3,025 images, divided into 2,581 train and 444 validation by an image-name hash. That split is **not** patient-disjoint either, which affects only threshold selection, not the test numbers. Test is 553 images, of which 442 are leak-reduced and 111 leaky-only.
- **Training curve:** validation mAP@0.25 rose every epoch except one, from 0.105 to 0.308. Validation mean sensitivity at 1 FP/image went from 0.28 to 0.68. The model had not plateaued; more epochs or higher resolution would probably help.
- **Operating points:** each class's box-score threshold is chosen on validation at 1 FP/image. The validation gate is then applied on the **leak-reduced** test split (hit = IoU ≥ 0.25 with a radiologist box). The gate requires the lower end of the 95% CI for sensitivity at 1 FP/image to reach 0.80. The CI comes from an image-level bootstrap: whole images are resampled, so lesions and false positives from one image move together.

| Class | Test boxes (clean) | AP@0.25 all | AP@0.25 clean | Sens@1FP all | Sens@1FP clean | 95% CI clean | Sens@1FP leaky-only | Gate (lower bound) | Point-estimate gate |
|---|---|---|---|---|---|---|---|---|---|
| Cardiomegaly | 54 | 0.65 | 0.65 | 0.99 | 0.98 | 0.93–1.00 | 1.00 | **pass**; re-gated at 0.25 FP/image (0.91, CI 0.81–0.98): still passes (§6.3.2) | pass |
| Emphysema | 50 | 0.31 | 0.34 | 0.92 | 0.90 | 0.79–0.98 | 1.00 | fail; at 0.25 FP/image 0.66, CI 0.49–0.81 | pass |
| Consolidation | 367 | 0.65 | 0.67 | 0.82 | 0.85 | 0.81–0.90 | 0.67 | **pass** (lower bound 0.807) | pass |
| Atelectasis | 47 | 0.43 | 0.44 | 0.82 | 0.81 | 0.69–0.91 | 1.00 (4 boxes) | fail | pass |
| Pleural effusion | 318 | 0.60 | 0.62 | 0.76 | 0.76 | 0.71–0.81 | 0.82 | fail | fail |
| Mass | 27 | 0.13 | 0.16 | 0.59 | 0.63 | 0.41–0.80 | 0.57 | fail (< 30 boxes) | fail |
| Pleural thickening | 89 | 0.17 | 0.18 | 0.63 | 0.60 | 0.48–0.69 | 0.81 | fail | fail |
| Fibrosis | 78 | 0.22 | 0.18 | 0.58 | 0.55 | 0.43–0.67 | 0.62 | fail | fail |
| Calcification | 51 | 0.22 | 0.24 | 0.54 | 0.53 | 0.36–0.66 | 0.69 | fail | fail |
| Fracture | 97 | 0.18 | 0.18 | 0.43 | 0.41 | 0.31–0.52 | 0.50 | fail | fail |
| Pneumothorax | 33 | 0.12 | 0.08 | 0.40 | 0.36 | 0.20–0.55 | 0.56 | fail | fail |
| Nodule | 219 | 0.16 | 0.17 | 0.38 | 0.37 | 0.30–0.43 | 0.36 | fail | fail |
| **Mean** | | **0.32** | **0.32** | **0.66** | **0.65** | | **0.72** | **2 of 12** | 4 of 12 |

Leak-reduced test overall: mAP@0.5 0.23, mAP@0.25 0.32, mean sensitivity at 1 FP/image 0.65. Among true-positive hits, the box centre falls inside the radiologist box 92–100% of the time, depending on class. Contrast this with the classifier's activation maps in §4.2, which score at chance.

**How to read these numbers**

- **The leak inflates scores where it can be measured.** On the 111 leaky-only images, mean sensitivity at 1 FP/image is 0.72, against 0.65 on the clean images. The per-class leaky numbers rest on very few boxes (atelectasis has 4), so they show the direction of the bias, not its size. The 414 test images that cannot be mapped to NIH patients may still leak, so even the "clean" numbers are an upper bound.
- **The gate now uses the lower confidence bound, not the point estimate.** The first version of the gate compared the point estimate with 0.80, and passed four classes. Atelectasis passed at 0.81 on only 47 boxes, which is not evidence that sensitivity is at least 0.80. The gate now uses the lower end of the image-bootstrap 95% CI. Atelectasis (lower bound 0.69) and emphysema (0.79) drop out. Consolidation passes only just (0.807), so it is the first class to recheck on new data. Set `gate_statistic: point` to reproduce the original gate.
- **Effusion fails the localisation gate even though the classifier detects it well** (AUROC 0.81 against radiologist labels). The pipeline handles this as designed: effusion can still be `present` as `model_inferred`, but its boxes are shown with `validated=false` and never support an `image_observed` claim.
- **Every detector output is `research_only=true`.** The backbone is torchvision's ImageNet ResNet-50, whose licence verdict is CONDITIONAL (`registry.py`, `NOTICE-DATA.md`). A product build must retrain from a cleared initialisation: DINOv2, or our own SSL weights.

**End-to-end check.** Run on ChestX-Det test image `40749.png`, which is not in the leaky set. Radiologist boxes: effusion ×2 (right of image), consolidation (left), pleural thickening.

```bash
uv run radio infer runs/nih_dinov2s_224/bundle data/chestx_det/test/40749.png --view-position PA \
    --detector runs/chestxdet_frcnn_512/det_bundle --json runs/e2e_40749.json
```

| Concept | Output | Comment |
|---|---|---|
| Pleural effusion | `present`, `model_inferred`, 5 boxes all `validated=false` | Correct finding. The top-scoring box barely overlaps the reference (IoU 0.09); the fourth box matches it (IoU 0.78). Effusion localisation is ungated, so all boxes are display-only |
| Consolidation | `uncertain`, `image_observed`, validated box | The validated box covers the **effusion** region (IoU 0.78 with the reference effusion box, 0 with the reference consolidation box). The fusion rules cap it at `uncertain`, so it goes to the radiologist |
| Atelectasis | `uncertain`, `model_inferred`, box `validated=false` | This box matches the reference **consolidation** box (IoU 0.81); atelectasis and consolidation are visually confusable. Under the point-estimate gate, this box was `validated=true` and the finding was `image_observed`. The lower-bound gate removed that claim |
| Cardiomegaly, emphysema | `not_assessed` | No box above threshold (cardiomegaly is validated, emphysema is not). A detector negative is not validated to rule out, and the classifier lacks calibration data |
| Fracture, calcification | `not_assessed` (`detector_not_validated`) | Detector-only classes that failed the gate |
| Report | `validation_passed=True`, OOD score 20.5 (threshold 34.7) | "Pleural effusion. Possible consolidation, lung opacity, nodule, atelectasis and fibrosis…" |

This one image shows why validated boxes still need radiologist review: a class-level gate does not make each box correct. The fusion rules kept the mislocalised consolidation claim at `uncertain` rather than `present`.

**Phase-3 actions from this run:**

1. ~~Gate on the lower confidence bound (image-level bootstrap) instead of the point estimate.~~ Done; see the table above.
2. Retrain from a commercially clean initialisation.
3. Train longer and at higher resolution; the curve was still rising.
4. Get a patient-disjoint external box test set; the ChestX-Det test split cannot fully rule out leaks.
5. Add cross-class box suppression or anatomy constraints for confusable pairs: consolidation/atelectasis, effusion/consolidation.

## 5. How to reproduce

```bash
uv sync && uv run pytest
scripts/fetch_nih_subset.sh 1
uv run radio train configs/classifier_nih_mac.yaml
uv run radio calibrate runs/nih_dinov2s_224
uv run radio evaluate runs/nih_dinov2s_224
uv run radio serve runs/nih_dinov2s_224/bundle   # then see 11-api-spec.md §2.2
# Phase 2
scripts/fetch_chestx_det.sh
uv run radio dedupe-chestx-det && uv run radio leak-check-chestx-det
uv run radio train-detector configs/detector_chestxdet.yaml
uv run radio regate-detector runs/chestxdet_frcnn_512
# Phase 3
uv run radio validator-suite --n 5000 --out runs/validator_suite.json
uv run radio report-eval runs/nih_dinov2s_224/bundle --detector runs/chestxdet_frcnn_512/det_bundle
```

## 6. Phase 3: structured reporting (started 2026-11-25)

Scope on this Mac with public data only. Items that need radiologists or partner report text are listed as blocked rather than approximated.

| WP | Task | Status |
|---|---|---|
| 3.1 Template library | Templates moved from code to versioned data: `src/radio/report/templates/cxr_v1.yaml`, loaded by `report/templates.py`. Loading is fail-closed: a missing sentence or provenance suffix raises. Every draft records `radio-cxr-templates@1.0.0`. Statements now carry provenance wording, per doc 06 §5. `model_inferred` reads "(AI-inferred from the whole image; location unverified)" and `image_observed` reads "(AI-localised; see overlay)". Before this, a whole-image classifier call read exactly like a localised one | Done. Exhaustive test: all 954 schema-valid combinations of concept, status, laterality, provenance and critical render a draft that passes validation, names the finding and states its side. **The wording is an engineering draft:** `clinical_review.reviewed_by` is null until radiologists author and sign it |
| 3.2 Synthetic perturbation suite | `report/perturb.py`, `radio validator-suite`. Injects 20 classes of known error into template drafts of random IRs and counts how many the validator catches | Done; results below. The suite found and fixed one real validator gap |
| 3.3 Comparison with priors | `report/comparison.py`. Compares only against a prior's **radiologist-verified structured findings** (`PriorStudy.findings_verified_by`). New `comparison` sentence kind and report section. The validator recomputes every claimed change from the IR and the prior | Done for CLI (`infer --prior`) and API (`prior_study_id`). Patient matching is the caller's job and is recorded in audit |
| 3.4 Report-level evaluation | `eval/report_eval.py`, `radio report-eval`. Runs the full pipeline on ChestX-Det test images and scores what each draft asserts against radiologist labels | Done; results below |
| 3.5 Constrained LLM rephraser | Closed-vocabulary guard added (below). The LLM itself is not selected or downloaded | Pending: needs a licence check and approval to download a model |
| 3.6 Gold QA set (~300 reports) | — | **Blocked:** needs radiologist authorship |
| 3.7 Guideline-backed recommendations | `APPROVED_GUIDELINES` stays empty | **Blocked:** needs clinical safety officer approval of specific rules |
| 3.8 Report metrics dashboard | JSON plus CLI tables only | Pending |

### 6.1 Comparison rules

| Prior (verified) | Current | Statement |
|---|---|---|
| absent | present | "X: new since the prior study of DATE (AI comparison)." |
| present | present | "X: previously reported on the prior study of DATE (AI comparison)." No claim about size or severity |
| present | absent | "X: reported on the prior study of DATE, not identified by AI analysis on this study (AI comparison)." |
| absent | absent | Nothing |
| any | uncertain or not assessed | Nothing |
| side known on both and different | any | Nothing |

- **Which priors can be compared:** a prior is used only if a radiologist approved the AI draft, or amended it through structured finding overrides only. If the final report is radiologist free text, its structured findings may not match what the radiologist wrote, so the prior is refused for comparison. The CLI warns; the API returns 409. Unreviewed AI drafts are never used as priors.
- **The validator side** has three error codes:
  - `comparison_mismatch`: the claimed change, or its marker phrase, differs from the change computed from the IR and the prior.
  - `comparison_omitted`: a computed change is not stated.
  - `comparison_without_verified_prior`: the sentence cites no verified prior.
- **Comparison wording is now allowed only in comparison sentences.** Before this, any sentence could say "unchanged" as long as some prior was attached.

### 6.2 Validator perturbation suite (5,000 random IRs, half with a verified prior)

Each of the 5,000 unperturbed drafts passed. Detection rate = share of perturbed drafts where the validator raised the expected error code.

| Injected error | Applied | Detected |
|---|---|---|
| Side swapped (left ↔ right, bilateral → left) | 3,917 | 1.000 |
| Side added to a finding of unknown side | 1,846 | 1.000 |
| Positive statement negated / "No" removed from a negative | 4,696 / 4,998 | 1.000 / 1.000 |
| Hedging removed from an uncertain finding | 4,676 | 1.000 |
| Uncited finding mentioned (canonical term) | 5,000 | 1.000 |
| Comparison wording in a finding sentence | 4,696 | 1.000 |
| Invented measurement | 4,696 | 1.000 |
| Present finding dropped / critical finding dropped from impression / not-assessed list dropped | 4,696 / 1,878 / 4,986 | 1.000 each |
| Unapproved recommendation / disclaimer altered / false "normal" / ungrounded sentence / sentence kind swapped | 4,989 / 5,000 / 4,989 / 5,000 / 4,696 | 1.000 each |
| Comparison: change flipped / wording flipped / statement dropped | 2,467 each | 1.000 each |
| **Uncited finding mentioned by a synonym** ("pleural fluid", "volume loss", "enlarged heart"…) | 5,000 | **0.040** |

How to read this:

- **The suite found a real gap, now fixed.** Side words were checked per sentence, not per finding. In a grouped sentence ("Possible left lung opacity and bilateral nodule"), moving a side from one finding to another went unnoticed 20% of the time. The validator now binds each side word to the finding term it modifies.
- **100% partly reflects construction.** The probes were written by the same people as the rules, so a perfect score shows the rules do what they say. It does not show the rule set is complete. The gold QA set (3.6) and a radiologist-written error taxonomy are the independent check.
- **The validator alone misses findings described by synonyms.** It recognises findings by canonical terms, so synonyms slip through. That matters only for text the generator did not write, which in this design means the optional LLM rephraser. Rephrased sentences now face a **closed vocabulary**: a rephrase may reorder template words and add a short list of neutral words ("is", "seen", "noted"…), and nothing else. A hostile rephraser that appended a synonym to every sentence got 0 of 46,132 sentences through. The validator is not suitable for checking free text such as radiologist reports (16 R29).

### 6.3 Report-level results on radiologist labels (ChestX-Det test, measured 2026-11-25)

The full pipeline ran on 416 test images: the Phase-1 classifier `nih_dinov2s_224`, the Phase-2 detector gated on the lower confidence bound, the templates and the validator. Excluded were 137 test images: copies of NIH images the classifier trained or calibrated on, plus images that share a patient with detector training. View was set to PA because the PNGs carry no view tag. Run time was 7.5 min on MPS. The reference is the radiologist boxes: a concept is present if a radiologist boxed it. Only the 12 ChestX-Det concepts are scored.

| Report-level | Value |
|---|---|
| Drafts passing validation | 416 / 416 |
| Abstentions | 0 |
| False "normal" impressions | 0. The normal summary was never used: some finding was always present or uncertain |
| Contradictions (one finding both positive and negative) | 0 |
| Reports with a false "absent" (says "no X", radiologist marked X) | **13 (3.1%)** |
| Mean findings asserted per report (present + uncertain) | 5.5, of which **5.0 uncertain** |
| Finding precision / recall (uncertain counts as asserted) | 0.31 / 0.67 |

| Concept | Ref. positive | Radiologist-positive images → present / uncertain / **absent** / not assessed | Radiologist-negative images → present / uncertain / absent | PPV of `present` |
|---|---|---|---|---|
| Pleural effusion | 209 | 0.23 / 0.74 / **0.03** / 0 | 0.00 / 0.77 / 0.23 | 0.98 |
| Consolidation | 227 | 0.12 / 0.86 / **0.02** / 0 | 0.02 / 0.84 / 0.15 | 0.90 |
| Nodule | 95 | 0.07 / 0.89 / **0.03** / 0 | 0.06 / 0.92 / 0.02 | 0.26 |
| Fibrosis | 53 | 0.21 / 0.79 / 0 / 0 | 0.07 / 0.86 / 0.07 | 0.30 |
| Atelectasis | 44 | 0.16 / 0.84 / 0 / 0 | 0.11 / 0.88 / 0.01 | 0.15 |
| Cardiomegaly, emphysema, mass, pneumothorax, calcification, fracture, pleural thickening | 25–70 each | not assessed on every image | — | — |

Localisation: the detector made 303 validated consolidation box claims (`image_observed`). Of these, 66% hit a radiologist consolidation box at IoU ≥ 0.25.

What this shows:

- **The reporting layer behaves safely.** Every draft validated. No report called an abnormal image normal. When the system was confident (`present`), effusion was right 98% of the time and consolidation 90%.
- **Misses are rare and stay within the calibration target.** Radiologist-positive images were called `absent` in 2–3% of cases for effusion, consolidation and nodule. The conformal thresholds were set for a false-negative rate of at most 0.05 on NIH labels, so this transfer to radiologist labels is roughly in line. Those 13 reports still contain a confident false negative, and they are the cases for the reader study to look at.
- **The drafts are not useful yet: uncertainty overwhelms them.** About 85% of scored statuses are `uncertain`, even on images the radiologist marked negative. A typical draft lists five "possible" findings, which invites alert fatigue and automation bias (16 R28). The cause is the perception model, not the reporting layer. The classifier was trained on one NIH batch at 224 px (AUROC 0.6–0.8). A weak score means the band between the conformal miss-control threshold and the false-alarm-control threshold covers most images. The reporting layer is ready for a stronger classifier: full-data training at higher resolution in the cloud (14).
- **Fusion gap: validated detector classes were dropped when the classifier could not assess them (fixed in pipeline v0.3).** Cardiomegaly passes the detector gate (sensitivity lower bound 0.93), yet in the run above it was `not_assessed` on every image. The classifier has too little calibration data for cardiomegaly, and `pipeline/run.py` stopped at that point without looking at detector evidence. Decision (2026-11-25): a validated box in that case becomes an `uncertain`, `image_observed` claim, with uncertainty reason `classifier_not_calibrated_for_concept`. It never becomes `present`, because the gate measures sensitivity, not precision.

#### 6.3.1 Re-run with pipeline v0.3

Same 416 images. Cardiomegaly is now reported. The rest of the report-level picture:

| | v0.2 | v0.3 |
|---|---|---|
| Validation pass / false "normal" / reports with a false "absent" | 416 of 416 / 0 / 13 | 416 of 416 / 0 / 13 |
| Mean asserted (uncertain) findings per report | 5.5 (5.0) | 6.3 (5.75) |
| Finding precision / recall | 0.31 / 0.67 | 0.29 / 0.73 |
| Cardiomegaly: radiologist-positive images flagged | 0 of 52 (not assessed) | 52 of 52 |
| Cardiomegaly: images flagged in all | 0 | **326 of 416 (78%)**; 16% of the boxes hit a reference box |

- **The new rule exposed an operating-point problem.** Per-class box thresholds are chosen at 1 false-positive box per image. That suits focal lesions, which can be many per image. Cardiomegaly is a global finding that occurs at most once per image, so 1 FP/image means flagging almost every image: on the leak-reduced test set, image-level specificity is 0.27.
- **Lower false-positive rates for cardiomegaly**, threshold chosen on validation and measured on the 442 leak-reduced test images:

| Validation FP/image | Box score threshold | Image sensitivity | Image specificity | Images flagged | FROC sensitivity, 95% CI |
|---|---|---|---|---|---|
| 1.0 (current) | 0.014 | 0.98 | 0.27 | 338 | 0.98 [0.94, 1.00] |
| 0.25 | 0.374 | 0.91 | 0.71 | 163 | 0.91 [0.81, 0.98] |
| 0.1 | 0.736 | 0.70 | 0.90 | 77 | 0.70 [0.59, 0.83] |
| 0.05 | 0.862 | 0.50 | 0.97 | 39 | 0.63 [0.45, 0.76] |

  At 0.25 FP/image, cardiomegaly still passes the lower-bound gate (0.81 ≥ 0.80) and flags half as many images. Emphysema, the other global class, fails the gate at any rate below 1 FP/image (lower bound 0.50 at 0.25).

#### 6.3.2 Whole-image findings at 0.25 FP/image (decision 2026-11-25)

Cardiomegaly and emphysema (`global_concepts`) now take their box threshold, and their gate, at 0.25 FP/image. Focal findings stay at 1 FP/image. `radio regate-detector` re-picks thresholds from the saved validation predictions, so this needed no retraining. Focal thresholds came out identical.

| Concept | Gate rate | Threshold | Sensitivity at gate (95% CI), leak-reduced test | Gate |
|---|---|---|---|---|
| Cardiomegaly | 0.25 FP/image | 0.014 → 0.374 | 0.91 (0.81–0.98) | pass, just (lower bound 0.810) |
| Emphysema | 0.25 FP/image | 0.073 → 0.282 | 0.66 (0.49–0.81) | fail |

Report-level effect, same 416 images:

| | v0.2 (no detector fallback) | v0.3 (fallback, 1 FP/image) | v0.3 + 0.25 FP/image for global findings |
|---|---|---|---|
| Validation pass / false "normal" / reports with a false "absent" | 416 / 0 / 13 | 416 / 0 / 13 | 416 / 0 / 13 |
| Mean asserted (uncertain) findings per report | 5.5 (5.0) | 6.3 (5.75) | 5.9 (5.35) |
| Finding precision / recall | 0.31 / 0.67 | 0.29 / 0.73 | 0.31 / 0.72 |
| Cardiomegaly: radiologist-positive images flagged as possible | 0 of 52 | 52 of 52 | **49 of 52**; the other 3 are `not_assessed`, never `absent` |
| Cardiomegaly: radiologist-negative images flagged | 0 of 364 | 274 of 364 | **110 of 364** (image specificity 0.70) |
| Cardiomegaly: box claims hitting a reference box | — | 16% | 31% |

Cardiomegaly is now reported for review on 38% of images instead of 78%, at the cost of three positives left unassessed and disclosed as such. It is still a `possible` finding for the radiologist, not a confident one. Its lower bound sits exactly on the gate (0.810), so it is the first class to recheck on new data, together with consolidation (0.807).

### 6.4 Classifier retrained with radiologist labels (2026-11-25)

The report results above pointed to the perception model as the bottleneck, so the classifier was retrained on NIH batch 1 (NLP labels) plus ChestX-Det train (radiologist labels). Run `mixed_dinov2s_224`, config `configs/classifier_mixed_mac.yaml`, code `data/mixed.py`. Encoder, resolution (224 px), schedule (8 epochs) and seed match the Phase-1 run, so the comparison isolates the data and labels. Training took 12 min on MPS.

**How the two datasets are merged.** Both are NIH images, so the merge has to avoid duplicates and leaks:

- **Duplicates:** 673 ChestX-Det images are pixel copies of local NIH images. Each keeps its radiologist labels and inherits its NIH patient and split, and the NIH copy is removed.
- **NIH test patients:** 138 ChestX-Det images belong to NIH test patients and are dropped.
- **Unmapped images:** these are split by file-name hash, with 20% going to validation. That gives at least 30 calibration positives for emphysema.
- **Label masks:** ChestX-Det annotates 10 of the 14 NIH concepts. The other four are masked out on ChestX-Det rows, unless the NIH copy supplies them.
- **Calibration policy `radiologist_where_available`:** for the 10 shared concepts, temperature scaling and conformal thresholds use radiologist labels only. NIH validation rows are masked out for those concepts.

**Calibration.** 9 of 14 concepts now have enough radiologist-labelled calibration data, against 6 before. Newly assessed: cardiomegaly, emphysema, pleural thickening. Still not assessed: mass (19 positives), pneumothorax (19), pneumonia, edema, hernia.

**Image-level AUROC against radiologist labels.** Same 416 ChestX-Det test images as §6.3: known patient leaks and NIH copies the classifier trained on are excluded.

| Concept | Phase-1 classifier | Retrained | Activation-map hit rate (chance), retrained |
|---|---|---|---|
| Pleural effusion | 0.819 | **0.919** | 0.36 (0.11); was 0.02 |
| Cardiomegaly | 0.668 | **0.890** | 0.19 (0.10) |
| Consolidation | 0.766 | **0.842** | 0.59 (0.13) |
| Pleural thickening | 0.665 | **0.826** | 0.10 (0.03) |
| Emphysema | 0.704 | **0.816** | 0.67 (0.30) |
| Pneumothorax | 0.646 | **0.799** | 0.04 (0.04) |
| Atelectasis | 0.588 | **0.762** | 0.14 (0.04) |
| Mass | 0.522 | **0.750** | 0.20 (0.03) |
| Fibrosis | 0.663 | 0.691 | 0.32 (0.10) |
| Nodule | 0.499 | 0.662 | 0.34 (0.10) |
| **Mean pointing hit rate** | 0.11 (chance 0.10) | **0.29** (chance 0.10) | |

How to read the AUROC gains:

- **Patient overlap does not seem to drive them.** The remaining 414 unmapped ChestX-Det test images could still share patients with ChestX-Det train, which the retrained model saw. Including the 111 *known* overlapping images barely moves AUROC (effusion 0.910 with them against 0.919 without), which suggests patient overlap does not drive the gain.
- **On NIH's own test set (NLP labels, 62 patients) mean AUROC rose from 0.63 to 0.69.** Calibration there is worse, though: ECE is 0.23 for effusion and 0.31 for consolidation. ChestX-Det contains only abnormal images, so thresholds and temperatures fitted on it are miscalibrated for NIH's mostly normal population. **Calibration must be refitted on each deployment site's own population** (09 §7, PCCP).
- **Activation maps localise better but still unreliably.** The pointing hit rate rose from chance to about 3× chance, so the maps now carry some localisation signal. They stay `validated=false` and display-only (R27).

**Report-level effect.** Same pipeline: detector v0.3 with 0.25 FP/image for global findings, templates, validator.

| | Phase-1 classifier | Retrained |
|---|---|---|
| Validation pass / false "normal" | 416 of 416 / 0 | 416 of 416 / 0 |
| Effusion on radiologist-positive images: present / possible / **absent** | 0.23 / 0.74 / **0.03** | **0.59** / 0.36 / **0.05** |
| Effusion on radiologist-negative images: present / possible / absent | 0.00 / 0.77 / 0.23 | 0.03 / 0.39 / **0.58** |
| Consolidation on radiologist-negative images called absent | 0.15 | **0.33** |
| Precision of `present` calls: effusion, consolidation, nodule, atelectasis | 0.98, 0.90, 0.26, 0.15 | 0.95, 0.90, **0.73**, **0.41** |
| Concepts assessed by the classifier | 6 | 9 |
| Mean possible findings per report | 5.35 | 6.2 |
| Reports with a false "absent" | 13 | 23 |

- **Decisions got sharper where it matters.** For effusion, the share of images left "possible" halved on both positive and negative images. Precision of `present` calls rose for nodule and atelectasis.
- **More false "absent" calls, but within design.** For effusion, 5% of radiologist-positive images are now called absent, exactly the α_FN = 0.05 the thresholds target, against 3% before. The old model's very low `t_absent` (0.009) kept misses rare only by leaving almost everything "possible".
- **More "possible" findings per report.** The count rose because three more concepts are now assessed, and on this abnormality-only test set they mostly land in the indeterminate band.
- **Relaxing the false-positive limit does not reduce clutter.** Offline, with the same calibrated probabilities, raising α_FP from 0.05 to 0.10 / 0.20 / 0.30 lowers mean "possible" findings from 6.2 to 5.4 / 4.2 / 3.2. But precision of `present` calls falls from 0.73 to 0.60 / 0.50 / 0.42, and the total mentioned per report stays at 7.3. α_FP stays 0.05. Fewer, more correct findings per report need a stronger model: full NIH at 448 px plus more radiologist-labelled data (14, R28).

The demo server (`.claude/launch.json`, `radio-api`) served this classifier until §6.5. The review page leads with a short summary, ranks "possible" findings by probability, and folds ruled-out and not-assessed findings away.

### 6.5 Same data at 448 px (2026-11-25)

The 224 px model still left about 6 "possible" findings per report, and nodule AUROC was only 0.66. Small findings lose detail at 224 px, so the same run was repeated at 448 px. Run `mixed_dinov2s_448`, config `configs/classifier_mixed_mac_448.yaml`. Data, split, calibration policy, epochs and seed are unchanged. The only other changes are the ones 448 px forces: batch 16 instead of 32 (memory), and learning rates and warm-up adjusted to match.

**First attempt stalled; the learning rate was the cause, not the features.** With the 224 px learning rates at batch 16, validation AUROC stuck at about 0.56 (run kept as `runs/mixed_dinov2s_448_failed_lr`). To check whether DINOv2 features break at 448 px (high-norm artefact tokens are a known ViT issue at larger inputs), a frozen linear probe was trained on both resolutions: 448 px features scored 0.744 against 0.735 at 224 px, and showed no high-norm tokens. Halving the learning rates (encoder 1e-5, head 5e-4) and doubling the warm-up fixed it: epoch 0 reached 0.657, the same as the 224 px run. Training took about 8 min per epoch on an idle machine (about 70 min in all; wall time was longer because the Mac was in use).

**Image-level AUROC against radiologist labels** (same 416 ChestX-Det test images as §6.3 and §6.4):

| Concept | 224 px | 448 px | Activation-map hit rate, 224 → 448 |
|---|---|---|---|
| Pleural effusion | 0.919 | **0.933** | 0.36 → 0.22 |
| Cardiomegaly | 0.890 | **0.896** | 0.19 → 0.29 |
| Consolidation | 0.842 | **0.861** | 0.59 → 0.56 |
| Pleural thickening | 0.826 | **0.843** | 0.10 → 0.10 |
| Emphysema | 0.816 | **0.840** | 0.67 → 0.43 |
| Pneumothorax | 0.799 | **0.840** | 0.04 → 0.00 |
| Atelectasis | 0.762 | **0.770** | 0.14 → 0.07 |
| Mass | 0.750 | **0.770** | 0.20 → 0.28 |
| Nodule | 0.662 | **0.756** | 0.34 → 0.40 |
| Fibrosis | 0.691 | **0.750** | 0.32 → 0.30 |
| **Mean pointing hit rate** | 0.29 | 0.26 | chance 0.10 |

- **AUROC improved for all 10 concepts.** Nodule gained most (+0.09) and fibrosis next (+0.06), as expected for small or fine-textured findings.
- **Activation maps did not localise better** (mean hit rate 0.29 → 0.26). They remain display-only; localisation comes from the detector.
- **NIH test set (NLP labels, 62 patients):** mean AUROC 0.693 → 0.715. Calibration on NIH is still poor for effusion (ECE 0.21) and consolidation (0.29), for the reason given in §6.4.
- **Calibration coverage is unchanged:** the same 9 concepts are validated. Mass and pneumothorax (19 calibration positives each) still need more radiologist-labelled data.

**Report-level effect** (same pipeline and detector as §6.4):

| | 224 px | 448 px |
|---|---|---|
| Validation pass / false "normal" | 416 of 416 / 0 | 416 of 416 / 0 |
| Mean "possible" findings per report | 6.2 | **5.4** |
| Mean findings mentioned per report | 7.4 | **6.7** |
| Finding precision / recall | 0.28 / 0.82 | **0.31** / 0.81 |
| Reports with a false "absent" | 23 | 28 |
| Effusion on positive images: present / possible / absent | 0.59 / 0.36 / 0.05 | **0.69** / 0.26 / 0.05 |
| Nodule on positive images: present / possible / absent | 0.17 / 0.79 / 0.04 | **0.34** / 0.60 / 0.06 |
| Cardiomegaly on negative images called absent | 0.23 | **0.40** |
| Precision of `present`: effusion, consolidation, nodule, atelectasis | 0.95, 0.90, 0.73, 0.41 | 0.94, 0.92, 0.63, 0.32 |

- **Fewer, more decisive findings.** "Possible" findings fell by 0.9 per report. Twice as many radiologist-positive nodules are now called present, and more negatives are ruled out for every assessed concept.
- **The cost is more false "absent" calls**: 5 more reports, from atelectasis (0 → 5% of positives), emphysema (0 → 7%) and pleural thickening (3 → 7%). These sit at the α_FN = 0.05 target, but with 30–70 positives per concept the estimates are uncertain by several points, so they are tracked under R05 rather than accepted as proven.
- **Precision of `present` dropped for nodule and atelectasis** because many more calls are made; the number of correct nodule calls still roughly doubled.

**Decision:** the 448 px model replaces the 224 px model on the demo server (`.claude/launch.json`, `radio-api`). Inference stays under 1 s per image on the Mac. The review page leads with a short summary, ranks "possible" findings by probability, and folds ruled-out and not-assessed findings away. Uncertainty overload (R28) is reduced but not solved: 5.4 possible findings per report is still well above what a pilot could accept. The next gains need more data, not more resolution: the remaining NIH batches and more radiologist-labelled images, especially for mass and pneumothorax.

## Sources

- NIH ChestX-ray14 share and terms: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Wang et al., ChestX-ray8, CVPR 2017.
- DINOv2 (Oquab et al., TMLR 2024): https://github.com/facebookresearch/dinov2
- ChestX-Det: https://github.com/Deepwise-AILab/ChestX-Det-Dataset
- Guo et al. 2017 (temperature scaling), arXiv:1706.04599; Angelopoulos & Bates, arXiv:2107.07511; Angelopoulos et al., arXiv:2208.02814; Lee et al. 2018, arXiv:1807.03888.

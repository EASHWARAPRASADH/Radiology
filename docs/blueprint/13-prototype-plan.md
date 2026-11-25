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

## 4. How to reproduce

```bash
uv sync && uv run pytest
scripts/fetch_nih_subset.sh 1
uv run radio train configs/classifier_nih_mac.yaml
uv run radio calibrate runs/nih_dinov2s_224
uv run radio evaluate runs/nih_dinov2s_224
uv run radio serve runs/nih_dinov2s_224/bundle   # then see 11-api-spec.md §2.2
```

## Sources

- NIH ChestX-ray14 share and terms: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Wang et al., ChestX-ray8, CVPR 2017.
- DINOv2 (Oquab et al., TMLR 2024): https://github.com/facebookresearch/dinov2
- ChestX-Det: https://github.com/Deepwise-AILab/ChestX-Det-Dataset
- Guo et al. 2017 (temperature scaling), arXiv:1706.04599; Angelopoulos & Bates, arXiv:2107.07511; Angelopoulos et al., arXiv:2208.02814; Lee et al. 2018, arXiv:1807.03888.

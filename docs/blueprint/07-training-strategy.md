# 07 — Training Strategy

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

This document specifies the seven training stages, the concrete recipes for each, and which stages need radiologist-annotated data. Dataset roles and licences are in 03-datasets.md and 04-dataset-licensing-matrix.md.

**Ground rule:** shipped weights are trained only on **commercially cleared** data, meaning data whose licence has been confirmed by counsel and recorded in the data registry. Research-only data (MIMIC-CXR, CheXpert, VinDr-CXR, PadChest, BRAX, etc.) may be used in a separate research environment for benchmarking and method development. Its weights, embeddings, pseudo-labels and distilled outputs never flow into the clinical model lineage. MLflow lineage enforces this (see §9).

## Overview

| Stage | Goal | Data (Phase 1 → product) | Radiologist annotation needed? | Code |
|---|---|---|---|---|
| 1 Data preparation | Clean, de-identified, deduplicated, patient-split datasets with a normalised label space | NIH CXR14 subset → licensed partner + MIDRC (commercial agreement) | Only for label-quality audit (sample) | `src/radio/data`, `src/radio/ingest`, `src/radio/ontology` |
| 2 Representation learning | CXR-adapted image encoder | DINOv2 init → continued SSL on licensed CXRs | No | `src/radio/models/encoder.py` |
| 3 Abnormality classification | Multi-label finding probabilities | NIH CXR14 labels (NLP-mined) → partner labels + radiologist-labelled test sets | **Yes** for validation and test sets; recommended for a "gold" fine-tuning subset | `src/radio/train` |
| 4 Localisation | Boxes, masks, anatomy segmentation | ChestX-Det, SIIM-ACR, RSNA Pneumonia (after RSNA confirmation) → partner annotations | **Yes**, for every box and mask | `src/radio/models/detector.py` |
| 5 Image–report alignment | Shared image–text embedding for retrieval, QA and weak labels | Research env only (MIMIC-CXR, CheXpert Plus) → partner image–report pairs | No (reports are the supervision), but report de-identification audit is needed | research env |
| 6 Report generation | Templates + optional constrained rephraser | Template library authored with radiologists → rephraser tuned on partner reports | **Yes**, for template authoring and review | `src/radio/report` |
| 7 Clinical QA | Validator rules + error-detection evaluation | Synthetic perturbations + radiologist-reviewed drafts | **Yes**, for error taxonomy and the gold QA set | `src/radio/report/validator.py`, `src/radio/eval/report_metrics.py` |

## Stage 1 — Data preparation

### 1.1 Ingestion and DICOM preprocessing

1. Parse with pydicom. Record `Modality`, `ViewPosition`, `BodyPartExamined`, `PhotometricInterpretation`, `BitsStored`, `PixelSpacing`/`ImagerPixelSpacing`, `Manufacturer`, `ManufacturerModelName`, `PatientAge`/`PatientSex`. Unmodified headers stay in the restricted zone (10-system-architecture.md §4).
2. Pixel pipeline (`src/radio/preprocess/transforms.py`): Modality LUT → VOI LUT (or window centre/width) → invert if `MONOCHROME1` → scale to [0,1] float32 → store a **16-bit PNG** cache at long side 1024 px for training, plus the original pixel spacing.
3. PNG-only datasets (NIH CXR14, 8-bit) are tagged `source_bit_depth=8`. Mixing 8-bit and 16-bit sources is a known domain shift, so bit depth is recorded as a covariate for subgroup analysis.

### 1.2 De-identification

- Inbound data from partners is de-identified **before** it leaves the partner network: DICOM PS3.15 Annex E Basic Application Confidentiality Profile, plus Retain Longitudinal Temporal Information with Modified Dates if longitudinal priors are needed, plus Clean Pixel Data where burned-in text is possible.
- Burned-in text check: OCR over corners/edges on a sample plus header `BurnedInAnnotation`. Flagged images are quarantined.
- Free-text reports go through a PHI scrubber and a manual audit of a random sample (sample size per partner set; e.g. 200 reports per release, estimate).
- A pseudonymous `patient_key` is HMAC-SHA256 of the source MRN with a per-partner secret held by the partner or an honest broker. Radio never sees MRNs.

### 1.3 Patient-level deduplication and splitting

- **The split unit is the patient, never the image or study.** NIH CXR14 provides `Patient ID`; partner data uses `patient_key`.
- Near-duplicate detection: perceptual hash on 256 px thumbnails plus embedding cosine similarity > 0.98. Cross-split duplicates are removed from the larger split.
- Default split per source dataset: train 80 / validation 10 / calibration 5 / internal test 5 by patient, stratified on rare findings using iterative stratification. **External test sets are whole institutions**, never a slice of a training institution (08-evaluation-protocol.md §5).
- Calibration data is held out from model selection to keep conformal guarantees valid (`src/radio/calibrate/conformal.py`).

### 1.4 Label normalisation

- Every source label maps into the internal ontology (`src/radio/ontology/cxr.py`) with four target states: `positive`, `negative`, `uncertain`, `missing`.
- **"Missing" is not "negative".** NIH "No Finding" means "none of the 14 NLP labels were mined". It does not mean all other findings are absent, so labels outside the 14 are `missing`.
- CheXpert-style `-1` (uncertain) maps to `uncertain`. Label policies follow Irvin et al. 2019: ignore (mask), ones, zeros, or soft labels. We default to **mask** for training and keep uncertain cases out of test-set positives and negatives unless adjudicated.
- Mapping tables are versioned and unit-tested (each source label maps to exactly one ontology concept or to `unmapped`).

### 1.5 Data quality analysis

Each dataset release must have a report covering:

- label prevalence by split;
- view mix (PA/AP/LAT);
- age and sex distribution;
- manufacturer distribution;
- duplicate rate;
- the label-noise audit: a radiologist reviews a random 200 images per major label (estimate) against the dataset label, with disagreement rate and 95% CI.

## Stage 2 — Image representation learning

**Recommendation:** continue DINOv2 self-supervised pretraining on CXRs. Justification:

- DINOv2 is Apache-2.0 (weights and code).
- RAD-DINO (Pérez-García et al., Nature Machine Intelligence 2025) showed that DINOv2-style SSL on CXRs alone yields a strong CXR encoder without text supervision.
- SSL needs no labels, so it can use every licensed image, including unlabelled partner archives.

| Item | Mac smoke recipe | Cloud full recipe (estimate) |
|---|---|---|
| Init | `dinov2_vits14` (timm/torch hub) | `dinov2_vitb14` or `dinov2_vitl14` |
| SSL | Skip (fine-tune directly) | DINOv2 objective (self-distillation + iBOT masked-token loss), teacher EMA 0.994→1.0 |
| Data | — | All licensed CXRs (target ≥ 300k frontal images; estimate) |
| Resolution | 448 px (1,024 patch tokens) | 518 px global crops, 98 px … 224 px local crops |
| Augmentations | see below | Random resized crop (scale 0.5–1.0 global), small rotation ±10°, brightness/contrast/gamma jitter, Gaussian blur, CLAHE p=0.2; **no horizontal flip**; no colour jitter (greyscale) |
| Optimiser | — | AdamW, lr 5e-4 × batch/1024 (DINOv2 default scaling), wd 0.04→0.2, cosine schedule, 10-epoch warm-up |
| Compute | — | See 14-infrastructure-costs.md |

**Why no horizontal flip.** Laterality is a reported attribute (left vs right effusion, pneumothorax side, device position), and anatomy is asymmetric (heart, aortic arch, gastric bubble). Flipping teaches the model to treat dextrocardia-like appearances as normal and corrupts laterality supervision for detection. Vertical flips and large rotations are excluded for the same anatomical-plausibility reason.

**Benchmarks to beat** (research env only): RAD-DINO and MedSigLIP linear-probe AUROC on the same internal test split. The product encoder should be within a pre-declared margin (e.g. ≤ 0.02 macro-AUROC, proposal) of the best research-only encoder. Otherwise the shortfall is a documented trade-off.

## Stage 3 — Abnormality classification

| Item | Recipe |
|---|---|
| Model | Encoder + linear head (linear probe first) → full fine-tune with layer-wise LR decay 0.65–0.75 |
| Labels | Phase-1 targets from NIH CXR14: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia. These map to ontology concepts; "Infiltration" maps to `pulmonary_opacity` (documented as lossy) |
| Loss | **Masked BCE**: per-label mask = 0 for `uncertain`/`missing`, 1 for `positive`/`negative`. Alternative: Asymmetric Loss (ASL; Ben-Baruch et al. ICCV 2021) with γ⁻=4, γ⁺=0, clip 0.05, compared on validation AUPRC. Class-balanced positive weights capped at 10 |
| Optimiser | AdamW, lr 1e-4 (head 1e-3), wd 0.05, cosine decay, 1-epoch warm-up, grad-clip 1.0, EMA of weights 0.999 |
| Resolution | 448 px (Mac) / 518–768 px (cloud; larger helps small nodules and pneumothorax, an assumption to test) |
| Augmentation | Random resized crop scale 0.8–1.0, ratio 0.9–1.1; rotation ±7°; translation ±5%; brightness/contrast/gamma ±10%; optional simulated text-marker cut-out. **No horizontal or vertical flip** |
| Sampling | Patient-balanced: max k images per patient per epoch (k=2) to stop heavy-follow-up patients dominating (an NIH CXR14 property) |
| Batch / precision | Mac: batch 16–32, autocast fp16/bf16 on MPS if supported by the installed PyTorch, else fp32. Cloud: batch 256 across GPUs, bf16 |
| Model selection | Mean validation AUPRC over findings with ≥ 50 positives; early stop patience 3 epochs |
| Shortcut checks | Saliency audit on 100 positive pneumothorax cases (tube vs no tube); stratified AUROC by presence of chest drain where labelled |

**Mac smoke run** (Phase 1): 2,000–10,000 images from 1–2 NIH batches, ViT-S/14 at 448 px, 3–5 epochs. Its purpose is to prove the pipeline end to end (loader → training → calibration → evaluation → IR → report), **not** to produce clinical-grade AUROC.

**Cloud full run:** see 14-infrastructure-costs.md. NIH CXR14 is the full public corpus (112,120 images). Partner data joins once agreements are in place.

### Radiologist annotation for Stage 3

| Set | Purpose | Size (estimate) | Rationale |
|---|---|---|---|
| Validation gold | Threshold and model selection free of NLP-label noise | ~1,000 studies per data source | Enough for ~50–100 positives in common findings at typical prevalence |
| Calibration | Conformal thresholds | ≥ 500 studies per site, ideally ≥ 100 positives per finding | Split-conformal guarantees need exchangeable labelled data; very rare findings get wider margins |
| Test (internal + external) | Claims | ≥ 150 positives per claimed finding per test set | For sensitivity ≈ 0.90, a 95% CI half-width of ±0.05 needs n ≈ 1.96² × 0.9 × 0.1 / 0.05² ≈ 139 positives; ±0.03 needs ≈ 385 |

**Annotation protocol** (all radiologist-labelled sets):

- **Readers:** board-certified (or equivalent) radiologists; three independent readers per study for test sets, two for validation.
- **Blinding:** readers see the image (and, for the test set, the indication only), never AI output or the original report.
- **Label form:** per finding, choose present / absent / uncertain, plus laterality and location, using the ontology vocabulary in the annotation tool (MONAI Label / CVAT forms).
- **Adjudication:** majority vote among three. If all three disagree (present / absent / uncertain), a fourth senior reader adjudicates. Uncertain-majority cases are excluded from binary metrics and reported separately.
- **Quality:** a calibration session before annotation (20 practice cases with feedback); 5% repeated cases to measure intra-reader agreement; inter-reader agreement reported (Fleiss κ, per finding).
- **Governance:** annotation contract specifies IP assignment, confidentiality, and that annotators have no conflict with the model under test.

## Stage 4 — Localisation

| Item | Recipe |
|---|---|
| Detector | Baseline: torchvision Faster R-CNN with the Stage-2 ViT encoder via a simple feature pyramid (or ResNet-50-FPN trained from scratch / licence-clean init). Production candidate: RT-DETR / D-FINE (Apache-2.0) with the same encoder |
| Segmentation | nnU-Net 2D for lungs, heart, mediastinum, diaphragm and devices (lines and tubes). Gives the midline for laterality, CTR for cardiomegaly, and zone masks for location naming |
| Phase-2 data | ChestX-Det (3,578 NIH images, 13 classes, boxes + masks; Apache-2.0), SIIM-ACR pneumothorax masks (commercial use allowed per Kaggle rules; no redistribution), RSNA Pneumonia boxes (commercial use pending RSNA's written confirmation) |
| Loss | Detection: focal / GIoU per framework defaults. Segmentation: Dice + CE (nnU-Net default) |
| Augmentation | As Stage 3; boxes and masks transformed consistently; **no flips** |
| Weak localisation | Heatmaps (Grad-CAM on the classifier) are **display aids only**. They are never used as evidence for `present` or for measurements |
| Evaluation | FROC, mAP@[0.25, 0.5] IoU, pointing-game hit rate, Dice/HD95 (08-evaluation-protocol.md §3) |

**Annotation:** every box and mask needs a radiologist. Estimate: ~1,000–3,000 positive images per localised finding for a production detector. Rationale: public CXR detection sets of 15k–18k images with ~3 readers per image (VinDr-CXR) underpin current baselines, and rare classes in those sets have a few hundred instances and perform poorly. This should be confirmed with learning-curve experiments on ChestX-Det subsets during Phase 2. Protocol: two readers draw boxes independently; boxes are fused by weighted-box-fusion at IoU ≥ 0.4; disagreements on presence are adjudicated by a third reader.

## Stage 5 — Image–report alignment

- **Objective:** CLIP/ConVIRT-style contrastive learning between the Stage-2 image encoder and a clinical text encoder, plus local alignment (GLoRIA-style region–phrase).
- **Uses (none in the product's generation path):**
  1. **Retrieval:** nearest-neighbour reports as a reference for template authors.
  2. **Weak labels:** zero-shot scoring of findings outside the supervised label set, for research triage of what to annotate next.
  3. **QA flagging:** a "possible missed finding" score that sends a draft to closer review but never writes text.
- **Data:** commercially licensed image–report pairs are the bottleneck. NIH CXR14 has no reports. MIMIC-CXR, CheXpert Plus and PadChest are research-only. BIMCV-COVID19+ may permit commercial use (agreement unread). **Product-grade Stage 5 therefore depends on partner data.** Until then, Stage 5 runs only in the research environment and its weights are tagged `research_only` in MLflow.
- **Recipe (estimate):** symmetric InfoNCE, temperature learned (init 0.07), batch ≥ 1,024 pairs (gradient caching), text encoder = an Apache/MIT-licensed clinical BERT-class model, verified at selection time. 20–50 epochs.

## Stage 6 — Report generation

1. **Template library (primary).** Radiologists author sentence templates per ontology concept × status × laterality × severity × provenance, plus section ordering rules and a critical-finding-first impression rule. Templates are versioned, unit-tested, and reviewed by ≥ 2 radiologists per release.
2. **Constrained rephraser (optional).** A small Apache-2.0 instruction-tuned LLM. The candidate is chosen at implementation time after re-verifying its licence (Qwen / Gemma-4-class per the 2026-11-25 survey).
   - **Training:** supervised fine-tuning (LoRA) on pairs (template sentences + IR JSON → radiologist-style phrasing with the same `finding_ids`). Pairs come from partner reports aligned to radiologist-confirmed IR.
   - **Decoding:** grammar-constrained JSON (outlines/xgrammar), greedy or temperature ≤ 0.3.
   - **Acceptance:** zero increase in hallucination and contradiction rate versus templates-only on the gold QA set (08-evaluation-protocol.md §4). Otherwise the rephraser stays disabled.
3. **No end-to-end VLM generation in the product path** (06-recommended-architecture.md §4.4).

## Stage 7 — Clinical quality assurance

| Check | Implementation | Test data |
|---|---|---|
| Contradictory findings | Rules (same concept with two statuses; "no effusion" + "small effusion"); KB of mutually exclusive pairs | Synthetic perturbation suite |
| Unsupported diagnoses | Sentence without a `source_finding_id` or KB-backed inference | Synthetic + LLM red-team prompts |
| Missing important findings | IR finding with status `present` (or a critical finding `uncertain`) not mentioned in the draft | Synthetic deletion |
| Incorrect laterality | Sentence laterality ≠ IR laterality; also box centroid vs midline consistency upstream | Synthetic swap |
| Inconsistent measurements | Number in text not equal to the IR `Measurement` value (with unit) | Synthetic |
| Unsupported recommendations | Recommendation without `guideline_id` + satisfied applicability conditions | Synthetic + KB tests |
| Report completeness | Required sections present; every assessed finding accounted for; quality limitations stated | Unit tests |

The **gold QA set** is ~300 drafts reviewed by two radiologists who tag every error with the taxonomy above (estimate; sized to detect a 1–2% hallucination rate with useful precision). It measures validator recall and drives new rules.

## Which stages require radiologist-annotated data — summary

| Stage | Radiologist data | Why |
|---|---|---|
| 1 | Label-noise audit sample | Quantify NLP-label error |
| 2 | None | Self-supervised |
| 3 | Validation, calibration and test gold labels; optional gold fine-tune set | NLP labels are too noisy for claims and thresholds |
| 4 | All boxes and masks | No commercially clean, sufficiently large public box set |
| 5 | None for training; PHI audit | Reports are the supervision |
| 6 | Template authoring, rephraser target review | Clinical language ownership |
| 7 | Gold QA error taxonomy | Validator recall measurement |

## 9. Lineage and reproducibility

- Every run logs to MLflow (git SHA, config, dataset manifest hash, licence class of every dataset, and the parent model run).
- **Licence propagation rule:** a run's licence class is the most restrictive class of its inputs. The model registry refuses promotion of any run with class `research_only` to the `clinical-candidate` stage.
- Seeds fixed. Deterministic algorithms are enabled where available. Each set of reported metrics is tied to a `(model_version, dataset_manifest, eval_code_sha)` triple.

## Sources

- DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024, https://arxiv.org/abs/2304.07193 · code/weights Apache-2.0, https://github.com/facebookresearch/dinov2
- RAD-DINO: Pérez-García et al., Nature Machine Intelligence 2025; model card https://huggingface.co/microsoft/rad-dino
- CheXpert uncertainty-label policies: Irvin et al., *CheXpert*, AAAI 2019, https://arxiv.org/abs/1901.07031 (citation from memory; arXiv ID not re-opened this session)
- Asymmetric Loss: Ben-Baruch et al., ICCV 2021, https://arxiv.org/abs/2009.14119 (citation from memory; not re-opened this session)
- ConVIRT: Zhang et al., https://arxiv.org/abs/2010.00747; GLoRIA: Huang et al., ICCV 2021 (citations from memory; not re-opened this session)
- Temperature scaling: Guo et al. 2017, https://arxiv.org/abs/1706.04599 · Conformal risk control: https://arxiv.org/abs/2208.02814
- DICOM PS3.15 Annex E (de-identification profiles): https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html
- Dataset facts (NIH CXR14 112,120 images / 30,805 patients; ChestX-Det 3,578 images; VinDr-CXR 18,000 images, 17 radiologists): see 03-datasets.md and its sources (NIH Box FAQ, Deepwise-AILab/ChestX-Det-Dataset, PhysioNet VinDr-CXR)
- outlines https://github.com/dottxt-ai/outlines · xgrammar https://github.com/mlc-ai/xgrammar
- nnU-Net https://github.com/MIC-DKFZ/nnUNet · torchvision https://github.com/pytorch/vision · RT-DETR https://github.com/lyuwenyu/RT-DETR

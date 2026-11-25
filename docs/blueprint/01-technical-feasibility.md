# 01 — Technical Feasibility Report

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

## 1. Question

Can a small team build a **radiologist-in-the-loop chest X-ray (CXR) system** that:

- detects and localizes abnormalities;
- emits a structured, provenance-tagged findings intermediate representation (IR);
- drafts a grounded report;

and do it on a path that ends in FDA, EU MDR and CDSCO market authorization? This report also asks which parts can start **today**, given four constraints:

- an Apple M4 Pro Mac;
- public data only;
- no annotators;
- no clinical partner.

## 2. Verdict

| Capability | Technically feasible? | Feasible *now* with our constraints? | Blocker |
|---|---|---|---|
| DICOM ingestion, validation, VOI LUT, MONOCHROME1 handling | Yes (mature OSS: pydicom, highdicom) | **Yes** | — |
| Classical quality gate (exposure, size, blank/saturated) | Yes | **Yes** | Thresholds need clinical data to tune |
| OOD gate (Mahalanobis/energy on encoder features) | Yes (well-studied, see 02) | **Prototype yes** | Real OOD test set (non-CXR, pediatric, lateral, implants) must be assembled |
| Multi-label CXR classifier (14-ish findings) | Yes; well-established | **Smoke-scale yes** (NIH CXR14 subset) | Commercial-grade accuracy needs >100k clean-rights images and cloud GPUs |
| Localization (boxes / masks / heatmaps) | Yes | **Small-scale yes** (ChestX-Det 3,578 images, NIH ~1,000 boxes) | Large expert box sets are research-only (VinDr-CXR) → need own annotation |
| Calibration + conformal thresholds → `present / uncertain / absent` | Yes | **Yes** (method); clinical validity needs site-matched calibration data | Site data |
| Structured findings IR with `not_assessed` | Yes (engineering) | **Yes** | — |
| Grounded draft report from IR (templates + optional constrained LLM) | Yes | **Yes** (templates); LLM rephrase optional | Clinical style guide from radiologists |
| End-to-end image→report VLM | Yes (research) | **No for product** | Almost every strong CXR VLM is research-only or trained on research-only data (see 05) |
| Prior-study comparison | Yes | **No** | Needs longitudinal data with rights + registration; public commercial-use sets lack priors+reports |
| PACS/RIS integration (DICOMweb, SR TID 1500, FHIR) | Yes (mature OSS) | **Yes (lab)** | Real PACS needs a partner site |
| Regulatory-grade evidence (standalone + MRMC) | Yes | **No** | Needs IRB/ethics, multi-site data, ≥ dozens of radiologist readers, QMS |

**Conclusion.** The architecture is feasible and most of it can be built and unit-tested now. Three things are **not** feasible on current resources: (a) commercially shippable model weights, (b) any clinical-performance claim, (c) report generation from images by a learned VLM. These are blocked by **data rights and clinical partnerships, not by technology**.

## 3. Resource constraints

### 3.1 Compute: this Mac vs. cloud

Local machine: Apple M4 Pro, 12 CPU cores, 16-core GPU, 24 GB unified memory, PyTorch MPS backend. Free disk is **~41 GB (96% full)**.

| Work item | Mac (MPS) | Cloud (A100/H100 class) |
|---|---|---|
| Unit/integration tests, API, report engine, SR/FHIR export | ✔ | — |
| DINOv2 ViT-S/14 fine-tune, 224–448 px, ≤ 20k images | ✔ smoke/feasibility | — |
| ViT-B/L fine-tune on 100k+ images, multi-seed, HPO | ✘ (memory/time) | ✔ |
| Continued self-supervised pretraining (DINOv2-style) on CXR | ✘ | ✔ (multi-GPU) |
| Faster R-CNN / RT-DETR on ChestX-Det (3.5k images) | ✔ small runs | ✔ full |
| 2D nnU-Net segmentation | ✔ slow but works (2D only) | ✔ |
| 3D CT/MRI (Phase 7) | ✘ (MPS lacks 3D conv per nnU-Net docs — unverified for current PyTorch) | ✔ |
| LLM ≤ ~4B for constrained rephrasing, inference only | ✔ (quantized) | ✔ |
| VLM fine-tuning (≥ 3B) | ✘ | ✔ |

**Disk plan:** keep the local data footprint ≤ 10 GB.
- Download 1–2 NIH CXR14 image batches plus `Data_Entry_2017.csv`, and later ChestX-Det.
- Extract, then delete the archives.
- Put `data/`, `runs/` and `mlruns/` in `.gitignore`.
- Do any larger work on cloud or an external drive.

Cloud cost estimates are in `14-infrastructure-costs.md`.

### 3.2 Data rights

From `03-datasets.md` and `04-dataset-licensing-matrix.md`:

- **Usable now for the shipped model:** NIH ChestX-ray14 (112,120 images, unrestricted with attribution); ChestX-Det (3,578 NIH images, boxes and masks, Apache-2.0); SIIM-ACR Pneumothorax (Kaggle rules allow commercial use, no redistribution).
- **Conditional:** RSNA Pneumonia 2018. Its terms PDF and its dataset sheet conflict, so we need written RSNA confirmation.
- **Research-only (benchmark and external test only):** MIMIC-CXR, CheXpert/CheXpert Plus, VinDr-CXR, PadChest, BRAX.

**Consequences:**
- NIH CXR14 labels were extracted from reports by NLP. They are noisy, and "No Finding" ≠ normal. Our training signal is therefore weak until we get radiologist-labelled data.
- No commercially usable public CXR set ships paired free-text reports. The image–report alignment stage (Stage 5 in `07-training-strategy.md`) cannot be trained for the product on public data. Our design, where reports come from the IR, partly avoids that dependency.
- No Indian CXR dataset with verified commercial terms exists. The CDSCO guidance expects validation on Indian populations, so this needs a local partner.

### 3.3 Annotators and clinical partners

| Need | Phase | Minimum |
|---|---|---|
| Radiologist test-set labels (consensus of ≥ 3) | 1–4 | ~500–1,000 studies per finding group, multi-site |
| Expert boxes/masks for localization training | 2 | Thousands of positives for common findings; tens to hundreds for rare ones |
| Report-quality review (error taxonomy, edit distance) | 3–4 | Panel of ≥ 3 radiologists; blinded |
| MRMC reader study | 4 | Sized by power analysis (see `08-evaluation-protocol.md`) |
| PACS/RIS integration and pilot | 5–6 | One US, one EU and one Indian site, each with a DUA and ethics approval |

## 4. What Phase 1 can honestly show

It **can** show:

1. A working, tested pipeline from DICOM to IR to grounded draft report to review (`src/radio/pipeline/run.py`), which abstains on bad input.
2. A schema that encodes `present | absent | uncertain | not_assessed` and provenance (`image_observed | model_inferred | clinical_history | prior_report | radiologist_entered`). A report generator and validator (`src/radio/report/`) that **cannot** state an unsupported finding, by construction and by property tests.
3. A DINOv2-based multi-label classifier trained on an NIH subset on MPS. Per-label AUROC, AUPRC and ECE are reported with patient-level bootstrap CIs on a patient-disjoint split. These are **feasibility numbers, not clinical claims**. NIH labels are noisy, and the test labels are NLP-derived too.
4. Calibration (temperature scaling) and conformal thresholding producing the three-way status.
5. DICOM SR TID 1500 and FHIR DiagnosticReport export, both marked preliminary/AI-generated.
6. An audit trail with a hash chain.

It **cannot** show:
- clinical accuracy;
- generalization to other sites;
- radiologist benefit;
- report quality versus human reports;
- anything about prior-study comparison.

## 5. Top feasibility risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| F1 | Cannot get enough **commercially licensed, radiologist-labelled** CXR data | High | Critical | Start MIDRC commercial agreement and RSNA confirmation now; open hospital DUAs (US/EU/India) in parallel; budget for annotation |
| F2 | NIH-label noise caps achievable performance, and research-set benchmarks overstate it | High | High | Radiologist-adjudicated test sets; treat MIMIC/CheXpert results as research benchmarks only |
| F3 | A foundation-model shortcut (RAD-DINO, MedGemma, MAIRA-2) becomes load-bearing and later blocks commercialization | Medium | Critical | Licence gate in model registry (`05-model-comparison.md` §7); product path uses DINOv2 (Apache-2.0) plus our own pretraining |
| F4 | Shortcut learning / site confounding (Zech 2018; DeGrave 2021) | High | High | Multi-site external validation; site-stratified metrics; OOD gate |
| F5 | Subgroup underdiagnosis (Seyyed-Kalantari 2021) | Medium | High | Demographic subgroup metrics from day 1 wherever metadata exist |
| F6 | No regulatory predicate for **image-based** report drafting in US (likely De Novo) | High | High | Ship detection/localization (510(k), CADe, 21 CFR 892.2070) first; report drafting as a later indication |
| F7 | Mac compute/disk insufficient beyond smoke scale | Certain | Medium | Cloud budget from Phase 2 |
| F8 | Automation bias degrades radiologist accuracy when AI is wrong (Dratsch 2023; Yu 2024) | Medium | High | UI design (confidence display, no pre-filled "normal"), reader-study measurement |

## 6. Go / no-go criteria per phase

Thresholds marked † are internal engineering targets to be refined with clinical advisors. They are **not** clinical acceptance criteria.

| Phase | Go if… | No-go / rethink if… |
|---|---|---|
| 1 Research prototype | All pipeline tests pass. Classifier beats a trivial prevalence baseline on every label (AUROC CI excludes 0.5). Validator rejects 100% of injected unsupported findings. | Validator can be bypassed; patient leakage found |
| 2 Localization | Detection on held-out ChestX-Det/NIH boxes is non-trivial for target findings (mAP / FROC vs. baseline†). Heatmap pointing-game accuracy above random. | Localization no better than coarse lung-zone prior |
| 3 Reporting | Hallucination rate (sentences not traceable to IR) = 0 by construction. Radiologist panel accepts templated language for ≥ the agreed share of normal and simple-abnormal cases† | Panel rejects style; IR too coarse to express findings |
| 4 Retrospective clinical validation | Pre-registered primary endpoints met on ≥ 2 external sites. Subgroup gaps within pre-specified margins. MRMC shows non-inferior or better aided reading. | Any critical finding sensitivity below pre-specified floor; large site/subgroup gaps |
| 5 Workflow integration | Round-trip PACS → AI → SR/FHIR → RIS at partner site in test environment. Audit complete. | Integration requires modifying AGPL components (Orthanc plugins) |
| 6 Controlled pilot | Regulatory/ethics approvals in hand. Prospective safety monitoring shows no excess harm. | Unreviewed report reaches a patient record by any path |
| 7 Expansion (CT/MRI) | CXR product stable. Data rights for the new modality secured. `Analyzer` interface absorbs 3D without changes to IR. | 3D needs IR redesign |

## 7. Dependencies to start now (no code required)

1. File the MIDRC commercial data-use agreement.
2. Get written RSNA clarification on Pneumonia 2018 commercial use.
3. Obtain a Stanford AIMI commercial licence quote (FY25 pricing: $70,000 per dataset per year), only if CheXpert-scale data is justified.
4. Identify clinical partners in the US, EU and India. Draft DUA, ethics and IRB templates.
5. Engage regulatory counsel and a QMS consultant (ISO 13485, IEC 62304).

## Sources

- NIH ChestX-ray14 FAQ / terms: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/249502714403
- ChestX-Det: https://github.com/Deepwise-AILab/ChestX-Det-Dataset
- Stanford AIMI commercial use: https://aimi.stanford.edu/datasets-commercial-use
- MIDRC FAQ (commercial agreement): https://www.midrc.org/faqs
- PhysioNet MIMIC-CXR licence: https://physionet.org/content/mimic-cxr/view-license/2.1.0/
- RSNA Pneumonia terms: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf
- FDA 21 CFR 892.2070 reclassification (Federal Register 2020): https://www.govinfo.gov/content/pkg/FR-2020-01-22/pdf/2020-00494.pdf
- Aidoc chest X-ray report drafting — Breakthrough designation: https://www.mobihealthnews.com/news/aidoc-lands-fda-breakthrough-status-ai-drafts-radiology-reports
- CDSCO MDSW guidance summary: https://natlawreview.com/article/code-compliance-decoding-indias-medical-device-software-guidance
- Zech et al. 2018, PLOS Med: https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/
- Seyyed-Kalantari et al. 2021, Nat Med: https://pubmed.ncbi.nlm.nih.gov/34893776/
- Dratsch et al. 2023, Radiology: https://research.vu.nl/en/publications/automation-bias-in-mammography-the-impact-of-artificial-intellige/
- Yu et al. 2024, Nat Med: https://pmc.ncbi.nlm.nih.gov/articles/PMC10957478
- nnU-Net (MPS note in install docs): https://github.com/MIC-DKFZ/nnUNet

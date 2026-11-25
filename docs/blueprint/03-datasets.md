# 03 — Dataset Research

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

How these were checked: official pages (PhysioNet, Stanford AIMI, NIH Box, RSNA terms PDFs, Kaggle rules, BIMCV, Synapse, TCIA, figshare, GitHub), read on 2026-11-25. Any value that couldn't be confirmed from an official source is written **unverified**. Licence verdicts are summarised here; the full matrix and acquisition strategy are in `04-dataset-licensing-matrix.md`.

> Not legal advice. Counsel must confirm every licence before data enters a training set for a shipped model.

Abbreviations:

| Abbreviation | Meaning |
|---|---|
| NLP | labels extracted from reports by text mining |
| Rad | radiologist-annotated |
| DUA | data use agreement |
| CITI | CITI "Data or Specimens Only Research" human-subjects course |

## 1. Required datasets

### 1A. Content and annotations

| Dataset | Official source | Modality | # Images | # Patients / studies | Labels (how made) | Reports | Boxes | Masks | DICOM |
|---|---|---|---|---|---|---|---|---|---|
| **MIMIC-CXR v2.1.0** | https://physionet.org/content/mimic-cxr/2.1.0/ | CXR (frontal + lateral) | 377,110 | 227,835 studies; patients 65,379 per Johnson et al. 2019 (not re-checked) | None shipped with DICOM release | **Yes** (free text) | No | No | **Yes** |
| **MIMIC-CXR-JPG v2.1.0** | https://physionet.org/content/mimic-cxr-jpg/2.1.0/ | CXR | 377,110 | 227,827 studies | 14 labels via CheXpert and NegBio labelers (NLP), uncertain = −1; test set has a radiologist labelling | Via MIMIC-CXR | No | No | No (JPG) |
| **CheXpert** | https://stanfordmlgroup.github.io/competitions/chexpert | CXR | 224,316 | 65,240 patients | 14 labels from a rule-based NLP labeler with uncertain class. Validation: 200 studies, 3-radiologist consensus. Test: 500 studies, 8 radiologists | No | No | No | No |
| **CheXpert Plus** | https://aimi.stanford.edu/datasets/chexpert-plus | CXR | 223,228 (a paper table states 223,462) | 64,725 patients; 187,711 studies | 14 labels plus 8 demographic fields (age, sex, race, ethnicity, insurance, BMI, deceased, interpreter) | **Yes** (11 report sections) | No | No | **Yes** (+PNG) |
| **NIH ChestX-ray14** | https://nihcc.app.box.com/v/ChestXray-NIHCC | CXR, frontal only | 112,120 PNG (1024²) | 30,805 patients | 14 labels via NLP; accuracy estimated >90% (by the authors) | No | ~1,000 images (8 classes) | No | No (PNG only, per NIH) |
| **VinDr-CXR** | https://physionet.org/content/vindr-cxr/1.0.0/ | CXR | 18,000 (15,000 train / 3,000 test) | Not stated | 22 local findings (boxes) + 6 global labels. Rad: 17 radiologists, 3 per train image, 5-reader consensus on test | No | **Yes** | No | **Yes** |
| **PadChest** | https://bimcv.cipf.es/bimcv-projects/padchest | CXR, 6 view types | >160,000 | ~67,000 patients | 174 findings, 19 diagnoses, 104 locations. 27% labelled manually by physicians, the rest by a neural network | Spanish report-derived text (raw reports unverified) | No | No | Unverified |
| **RSNA Pneumonia Detection 2018** | https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018 | CXR (16,248 PA / 13,752 AP) | ~30,000 (drawn from NIH); test set 4,527 | NIH patients | Pneumonia-like opacity boxes. Rad: 18 radiologists from 16 institutions | No | **Yes** | No | Yes (converted from NIH PNG) |
| **RSNA Intracranial Hemorrhage 2019** | https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019 | Non-contrast head CT | Unverified (slice count) | >25,000 exams | 5 hemorrhage subtypes per slice. Rad: 60 neuroradiologists; train single-read, test 3-read | No | No | No | **Yes** |
| **RSNA Pulmonary Embolism 2020** | https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pe-detection-challenge-2020 | CT pulmonary angiography | Unverified | >12,000 studies, 5 centres | Exam- and image-level PE labels. Rad: >80 thoracic radiologists; train single-read, test 3-read | No | No | No | **Yes** |
| **BraTS** (2021 release on TCIA; 2025/2026 editions on Synapse) | https://www.synapse.org/Synapse:syn64153130 ; TCIA 2021 collection | Brain MRI (T1, T1c, T2, FLAIR) | 2025/2026: unverified | 2021: 1,480 patients (TCIA) | Tumour sub-region masks, drawn manually | No | No | **Yes** | 2021: NIfTI + DICOM |
| **Medical Segmentation Decathlon** | http://medicaldecathlon.com | CT + MRI, 10 tasks (brain, heart, liver, hippocampus, prostate, lung, pancreas, hepatic vessel, spleen, colon) | Unverified | Unverified | Expert-checked masks | No | No | **Yes** | NIfTI |

### 1B. Access, licence, limitations, role

| Dataset | Access requirements | Licence / commercial terms | Known limitations | Recommended role in Radio |
|---|---|---|---|---|
| MIMIC-CXR (+JPG) | PhysioNet credentialed account, CITI course, signed DUA | PhysioNet Credentialed Health Data License 1.5.0: data used "for the sole purpose of lawful use in scientific research and no other". No sharing. No commercial track exists. | Single centre (BIDMC, Boston); ICU-heavy; NLP labels | **RESEARCH-ONLY.** Internal benchmark and report-metric research. Never in shipped weights. |
| CheXpert | Stanford AIMI registration + research use agreement | Free for non-commercial research. Paid commercial licence via AIMI ($70k/dataset/yr FY25); whether CheXpert is in the commercial catalogue is unverified | Single centre (Stanford); NLP labels with uncertainty | **RESEARCH-ONLY** unless licensed. Benchmark. |
| CheXpert Plus | Stanford AIMI / Redivis | Paper: "may not be used for any commercial purposes", "only available for research uses" | Single centre | **RESEARCH-ONLY.** Report-generation benchmark. |
| NIH ChestX-ray14 | Open download | NIH FAQ: "usage of the data set is unrestricted". Must link the NIH site, acknowledge the NIH Clinical Center, and cite Wang et al. CVPR 2017. Kaggle mirror lists CC0 | Noisy NLP labels; "No Finding" ≠ normal (Oakden-Rayner 2020); frontal only; single centre; no reports | **OK.** Primary training/pretraining source now. Never sole test ground truth. |
| VinDr-CXR | PhysioNet credentialed + CITI + DUA | PhysioNet Credentialed 1.5.0 (research). Kaggle VinBigData copy: "Non-commercial, Research/Academic Use Only" | 2 Vietnamese hospitals; population shift | **RESEARCH-ONLY.** External localization test. Annotation-protocol template. |
| PadChest | Google Form; full clinical data by formal request | PadChest Research Use Agreement: "research purposes only"; "Its sale is prohibited"; non-academic use "evaluated case by case"; no derivative works | Single hospital (Spain, 2009–2017); Spanish text; partly model-labelled | **RESEARCH-ONLY.** External test (EU population). Ask BIMCV for commercial terms. |
| RSNA Pneumonia 2018 | Open (RSNA) / Kaggle account | RSNA terms PDF: "academic research and education, and other commercial or non-commercial purposes"; Kaggle rules agree. **Conflict:** RSNA dataset-description sheet says "Non-commercial purpose" | NIH images, so same population as CXR14; opacity ≠ pneumonia diagnosis | **CONDITIONAL.** Detection fine-tuning once RSNA confirms in writing. |
| RSNA ICH 2019 | Kaggle / RSNA | "academic research and education, and other non-commercial purposes"; Kaggle "Non-Commercial Use" | One site resampled to 5 mm; single-read train labels | **RESEARCH-ONLY.** Phase-7 CT research. |
| RSNA PE 2020 | Kaggle | Dataset sheet "Non-commercial purpose"; Kaggle "Non-Commercial Use" | Coarse labels | **RESEARCH-ONLY.** Phase-7 CT research. |
| BraTS | Synapse registration; post-challenge data moving to TCIA | 2021 challenge data on TCIA: **CC BY 4.0**; TCIA-derived DICOM segmentations are "TCIA Restricted". 2025/2026 terms: unverified | Pre-processed (skull-stripped, co-registered), so differs from clinical input | **OK (2021 CC BY 4.0 subset)** for Phase-7 MRI. |
| MSD | Open download | **CC-BY-SA 4.0** ("permissive copyright-license") | Small per-task sizes; share-alike may apply to redistributed derivatives (counsel: do trained weights count?) | **OK / CONDITIONAL** for Phase-7 CT/MRI segmentation pretraining. |

## 2. Additional CXR datasets checked for commercial usability

| Dataset | Size / content | Annotations | Licence / terms | Verdict |
|---|---|---|---|---|
| **SIIM-ACR Pneumothorax** (Kaggle 2019) | Size unverified; source images reportedly NIH (unverified) | Pneumothorax masks | Kaggle rules: "Any purpose, including commercial use"; no redistribution to non-participants | **OK** (no redistribution) |
| **ChestX-Det** (Deepwise) | 3,578 NIH images, 13 classes, 3 radiologists | Boxes **and** masks | Repo Apache-2.0; images carry NIH unrestricted terms | **OK.** Key Phase-2 localization set |
| **CANDID-PTX** | 19,237 DICOM images with reports (Dunedin, NZ) | Pneumothorax, rib-fracture, chest-tube masks | Figshare lists CC BY 4.0, but since Feb 2024 access is limited to Health New Zealand users | **UNAVAILABLE** |
| **Shenzhen / Montgomery TB** (NLM) | 662 (326 normal / 336 TB) / 138 (80 / 58) | Montgomery lung masks | Request form; "do not share … outside of their research groups"; credit NLM; no explicit licence | **UNVERIFIED** (ask NLM) |
| **JSRT** | 247 (154 nodule / 93 non-nodule) | Nodule locations | Official page 404; licence unverified | **EXCLUDE** until clarified |
| **Object-CXR** | 10,000 (5,000 with foreign objects) | Boxes / ellipses / polygons | CC BY-NC 4.0 (Academic Torrents listing) | **RESEARCH-ONLY** |
| **VinDr-PCXR** (pediatric) | 9,125 images, 51 labels | Boxes | PhysioNet Restricted Health Data License 1.5.0; commercial terms unverified | **RESEARCH-ONLY** (assume). Useful pediatric OOD test |
| **BIMCV-COVID19+** | 21,342 CR + 34,829 DX + 7,918 CT studies | Reports, labels | BIMCV page: research, education and "otros fines comerciales o no comerciales" (other commercial or non-commercial purposes), under an agreement not read; GitHub MIT covers code only | **CONDITIONAL.** Read agreement |
| **BRAX** (Brazil) | 40,967 images / 24,959 studies / 19,351 patients; DICOM | 14 labels from Portuguese reports | PhysioNet Credentialed 1.5.0 | **RESEARCH-ONLY** |
| **MIDRC** | Multi-institution US imaging commons (sizes not checked) | Varies | Two agreements: non-commercial and **commercial**; "data access is free" | **CONDITIONAL → OK** after commercial DUA. Priority acquisition |

**Kaggle default terms.** Standard competition Rule 7.A: "You may access and use the Competition Data for non-commercial purposes only…". Rule 7.B prohibits sharing with non-participants. A sponsor may override 7.A, as SIIM-ACR and RSNA Pneumonia do. **Always read the specific competition's rules page.**

## 3. How each dataset maps to the training pipeline

Stages follow `07-training-strategy.md`.

| Stage | Product-eligible data (now) | Research / benchmark data |
|---|---|---|
| S2 Representation learning (SSL) | NIH CXR14 images (no labels needed); later MIDRC and partner data | MIMIC, CheXpert, PadChest (to compare encoders only) |
| S3 Multi-label classification | NIH CXR14 (noisy labels, masked loss), SIIM-ACR (pneumothorax), RSNA Pneumonia (pending) | CheXpert (radiologist val/test), MIMIC-CXR-JPG test, PadChest |
| S4 Localization | ChestX-Det (boxes + masks), NIH ~1k boxes, SIIM-ACR masks, RSNA boxes (pending) | VinDr-CXR (22 findings, consensus test), Object-CXR (devices) |
| S5 Image–report alignment | **None** with commercial rights → deferred to partner data | MIMIC-CXR, CheXpert Plus |
| S6 Report generation | IR→template, so no image–report pairs needed. Phrase bank from radiologist-authored templates | MIMIC-CXR / CheXpert Plus for comparison with VLM baselines |
| S7 QA rule tuning | Synthetic IR perturbations + partner reports | MIMIC reports for error-taxonomy research |
| External validation | Partner sites (US/EU/India), MIDRC | VinDr-CXR (Vietnam), PadChest (Spain), BRAX (Brazil) as research-only robustness checks |
| OOD test | Non-CXR images, lateral views, MSD CT slices (as negatives) | VinDr-PCXR (pediatric) |

## 4. Gaps

1. No commercially usable public CXR set with **paired reports**.
2. No commercially usable public CXR set with **prior studies** (needed for longitudinal comparison).
3. No verified-commercial **Indian** CXR data. CDSCO's MDSW guidance (Jul 2026) expects validation on Indian populations.
4. Few commercially usable **device-position** labels (tubes and lines). Object-CXR is non-commercial.
5. Pediatric data is research-only, so the product must **exclude pediatrics** at the OOD gate until licensed data exist.

## Sources

- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.1.0/ ; licence https://physionet.org/content/mimic-cxr/view-license/2.1.0/
- MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert
- CheXpert Plus: https://aimi.stanford.edu/datasets/chexpert-plus ; paper https://arxiv.org/html/2405.19538
- Stanford AIMI commercial: https://aimi.stanford.edu/datasets-commercial-use
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC ; FAQ https://nihcc.app.box.com/v/ChestXray-NIHCC/file/249502714403
- VinDr-CXR: https://physionet.org/content/vindr-cxr/1.0.0/ ; Kaggle rules https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/rules
- PadChest: https://bimcv.cipf.es/bimcv-projects/padchest ; RUA https://bimcv.cipf.es/?p=4923
- RSNA Pneumonia terms: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf
- RSNA ICH terms: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/rsna_ich_ai_challenge_2019_terms_of_use_and_attribution_final.pdf
- RSNA PE Kaggle rules: https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/rules
- BraTS: https://www.synapse.org/Synapse:syn64153130 ; TCIA BraTS 2021: https://wiki.cancerimagingarchive.net/x/sIbECQ
- Medical Segmentation Decathlon: http://medicaldecathlon.com
- SIIM-ACR rules: https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/rules
- ChestX-Det: https://github.com/Deepwise-AILab/ChestX-Det-Dataset
- CANDID-PTX: https://figshare.com/articles/dataset/CANDID-PTX/14173982
- Shenzhen/Montgomery: https://qims.amegroups.org/article/view/5132/html
- Object-CXR: https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
- BIMCV-COVID19+: https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
- BRAX: https://physionet.org/content/brax/1.1.0/
- MIDRC FAQ: https://www.midrc.org/faqs
- Oakden-Rayner 2020 (CXR14 label audit) — see 02-literature-review.md

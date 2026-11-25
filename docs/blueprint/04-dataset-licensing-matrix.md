# 04 — Dataset Licensing Matrix & Data Acquisition Strategy

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

> Not legal advice. Every "OK" below means "no blocker found in the official terms read on 2026-11-25". Counsel must confirm before data enters a training set for a shipped model.

## 1. Verdict key

| Verdict | Meaning | Allowed in Radio |
|---|---|---|
| **OK** | Terms explicitly permit commercial use (attribution may be required) | Training data for shipped weights; test data; benchmarks |
| **CONDITIONAL** | Commercial use likely possible but needs a signed agreement, written confirmation, or counsel review of a specific clause | Research use now. Shipped weights only after the condition is closed and recorded in the data registry |
| **RESEARCH-ONLY** | Terms restrict use to research/education or ban commercial use | Internal R&D, benchmarking, external robustness tests. **Never** in training data for shipped weights, never in marketing claims |
| **UNAVAILABLE** | Cannot currently be obtained | — |
| **UNVERIFIED** | Terms could not be found or are ambiguous | Treat as RESEARCH-ONLY until resolved |

## 2. Matrix

Columns:

| Column | What it answers |
|---|---|
| Commercial training | Can the data train weights we ship? |
| Research / benchmark | Can it be used internally for research and evaluation? |
| Redistribution | Can we redistribute the images? |
| Derivative weights | Do the terms address model weights trained on the data? |
| Attribution | What citation or acknowledgement is required? |
| Access path | How to obtain it |
| Cost | Price of access |

| Dataset | Verdict | Commercial training | Research / benchmark | Redistribution | Derivative weights | Attribution | Access path | Cost |
|---|---|---|---|---|---|---|---|---|
| NIH ChestX-ray14 | **OK** | Yes ("usage … unrestricted") | Yes | Not prohibited in FAQ (don't redistribute anyway; link to source) | Not addressed (unrestricted) | Link NIH site, acknowledge NIH Clinical Center, cite Wang et al. CVPR 2017 | Open download (NIH Box) | Free |
| ChestX-Det | **OK** | Yes (Apache-2.0 annotations; NIH images) | Yes | Apache-2.0 for annotations | Not addressed | Apache NOTICE + NIH attribution | GitHub (annotations); images from the dataset owner's HTTP host, validated before extraction (`scripts/fetch_chestx_det.sh`) | Free |
| SIIM-ACR Pneumothorax | **OK** (no redistribution) | Yes ("Any purpose, including commercial use") | Yes | **No** (Kaggle 7.B) | Not addressed | Per rules page | Kaggle account + accept rules (user action) | Free |
| RSNA Pneumonia 2018 | **CONDITIONAL** | Terms PDF says yes; dataset sheet says "Non-commercial", so **get RSNA in writing** | Yes | Per terms (unverified detail) | Not addressed | RSNA attribution per terms PDF | RSNA / Kaggle | Free |
| MIDRC | **CONDITIONAL → OK** | Yes, under the **commercial** DUA | Yes (non-commercial DUA) | No | Per DUA (to review) | Per DUA | MIDRC application, commercial agreement | Free ("data access is free") |
| BIMCV-COVID19+ | **CONDITIONAL** | Page mentions commercial purposes; agreement text unread | Yes | Unverified | Unverified | Per agreement | BIMCV request | Unverified |
| Medical Segmentation Decathlon | **OK / CONDITIONAL** | Yes (CC-BY-SA 4.0) | Yes | Yes, under CC-BY-SA | **Counsel:** whether share-alike reaches trained weights | CC-BY-SA attribution | Open download | Free |
| BraTS 2021 (TCIA) | **OK** | Yes (CC BY 4.0) | Yes | Yes (CC BY) | Not addressed | CC BY attribution + TCIA citation | TCIA | Free |
| BraTS 2025/2026 (Synapse) | **UNVERIFIED** | Unverified | Yes (challenge use) | Unverified | Unverified | Per Synapse terms | Synapse registration | Free |
| MIMIC-CXR / -JPG | **RESEARCH-ONLY** | **No** ("sole purpose of lawful use in scientific research") | Yes, credentialed | **No** | Not addressed; treat weights as tainted | Cite PhysioNet + paper | PhysioNet credentialing + CITI + DUA | Free; **no commercial track exists** |
| CheXpert | **RESEARCH-ONLY** (CONDITIONAL with a paid licence) | Only via Stanford AIMI commercial licence; catalogue inclusion unverified | Yes | No | Not addressed | Cite Irvin et al. 2019 | Stanford AIMI | Commercial: **$70,000 per dataset per year** (FY25), ~4-week committee review |
| CheXpert Plus | **RESEARCH-ONLY** | No ("may not be used for any commercial purposes") | Yes | No | Not addressed | Cite paper | Stanford AIMI / Redivis | Free (research) |
| VinDr-CXR / VinBigData | **RESEARCH-ONLY** | No | Yes, credentialed | No | Not addressed | Cite Nguyen et al. 2022 | PhysioNet credentialing | Free |
| VinDr-PCXR | **RESEARCH-ONLY** (assumed) | Unverified → no | Yes | No | — | Cite | PhysioNet (Restricted licence) | Free |
| PadChest | **RESEARCH-ONLY** (case-by-case) | No; non-academic use "evaluated case by case" | Yes | No ("Its sale is prohibited"; no derivative works) | **"No derivative works"** clause → counsel | Cite Bustos et al. 2020 | Google Form; BIMCV | Free (research); commercial unverified |
| BRAX | **RESEARCH-ONLY** | No | Yes, credentialed | No | — | Cite | PhysioNet credentialing | Free |
| RSNA ICH 2019 | **RESEARCH-ONLY** | No | Yes | No | — | RSNA attribution | Kaggle / RSNA | Free |
| RSNA PE 2020 | **RESEARCH-ONLY** | No | Yes | No | — | RSNA attribution | Kaggle | Free |
| Object-CXR | **RESEARCH-ONLY** | No (CC BY-NC 4.0) | Yes | Non-commercial only | — | CC BY-NC | Academic Torrents | Free |
| Shenzhen / Montgomery | **UNVERIFIED** | Unverified | Yes (within research group) | No ("do not share outside research group") | — | Credit NLM | Request form | Free |
| JSRT | **UNVERIFIED** | Unverified | Unverified | Unverified | — | — | Official page 404 | — |
| CANDID-PTX | **UNAVAILABLE** | (CC BY 4.0 on figshare, but access restricted to Health NZ since Feb 2024) | — | — | — | — | Closed | — |

## 3. Policy rules enforced in the codebase

1. **Data registry.** Every dataset has a manifest entry: `dataset_id`, `version`, `licence_verdict`, `source_url`, `terms_snapshot_date`, `allowed_uses[]`. Training configs list `dataset_id`s. The training entrypoint (`src/radio/train/train_classifier.py`) refuses a run tagged `purpose: product` if any dataset's verdict ≠ OK. Model-registry lineage (MLflow tags) records the dataset IDs.
2. **Weight taint.** Any checkpoint that has ever seen RESEARCH-ONLY data (including through a pretrained initialisation, e.g. RAD-DINO or TorchXRayVision MIMIC/CheXpert weights) is tagged `research_only=true` and can't be promoted to a product model. This is the same rule as in `05-model-comparison.md` §7.
3. **No redistribution.** Raw images never leave the training environment. Demo assets come only from OK datasets whose terms allow redistribution (e.g. MSD, BraTS 2021) or from synthetic data.
4. **Attribution file.** `NOTICE-DATA.md` lists every dataset that went into a shipped model, with its required attribution text.

## 4. Data acquisition strategy

### Now (0–3 months), no partners needed

| Action | Output | Owner |
|---|---|---|
| Download NIH CXR14 subset (≤10 GB on Mac) plus ChestX-Det. Full NIH on cloud storage later | Phase-1/2 training data | Eng |
| Accept SIIM-ACR rules (user must do this personally on Kaggle) | Pneumothorax masks | Founder |
| Email RSNA for written confirmation of commercial terms for Pneumonia 2018 | Converts CONDITIONAL → OK or excludes it | Founder / counsel |
| Apply for the **MIDRC commercial agreement** | Multi-institution US data at no cost | Founder / counsel |
| Get PhysioNet credentialing (CITI) for 1–2 researchers | MIMIC-CXR, VinDr-CXR, BRAX for **benchmarking only** | Research lead |
| Read BIMCV-COVID19+ agreement; ask BIMCV about PadChest commercial terms | Possible EU data | Counsel |
| Ask NLM about Shenzhen/Montgomery commercial use | Possible TB data | Counsel |

### 6 months, first partners

| Action | Detail |
|---|---|
| **US hospital / imaging-group partner(s)** | DUA covering (a) retrospective de-identified DICOM plus reports and priors, (b) commercial model training rights, (c) radiologist annotation time, (d) publication rights. De-identification under HIPAA Safe Harbor (45 CFR 164.514(b)(2)) or Expert Determination. Burned-in-text OCR scrub. IRB determination (exempt or waiver). |
| **Annotation programme** | Radiologist-adjudicated test sets (≥3 readers). Box/mask annotation for priority findings on partner data. Tooling: MONAI Label / CVAT (see `10-system-architecture.md`). |
| **Stanford AIMI commercial licence** | Only if the gap analysis shows CheXpert-scale data is needed beyond MIDRC and partners. Cost: $70k per dataset per year. |
| **Data vendor evaluation** | Commercial de-identified imaging vendors (not researched yet). Diligence on consent, provenance and exclusivity. |

### 12+ months, multi-market

| Market | Action | Regulatory anchor |
|---|---|---|
| **EU** | 1–2 hospital partners (e.g. via academic medical centres). Legal basis under GDPR Art 6 plus an Art 9(2) condition (scientific research / public interest, or consent). DPIA. Pseudonymisation with keys held by the hospital. Ethics committee approval. Later, EHDS secondary-use access (applies mostly from 2029). | GDPR; EHDS Regulation (EU) 2025/327 |
| **India** | Partner hospital network for **Indian-population validation** (CDSCO MDSW guidance, Jul 2026, expects disclosed dataset provenance and Indian-population validation). Ethics approval following ICMR AI ethics guidelines (2023). Consent and notice under the DPDP Act 2023 / DPDP Rules 2025 (core obligations from 13 May 2027). | CDSCO MDSW guidance; DPDP; ICMR |
| **All** | Federated training/evaluation where data can't leave the site (NVFlare / Flower; MedPerf for federated benchmarking). This avoids moving data across borders. | Data-residency constraints |

### Requirements for every partner DUA

- Explicit **commercial-use** and **derivative-model** rights.
- Rights to keep and use model weights after the agreement ends.
- De-identification standard, with DICOM PS3.15 Annex E confidentiality profile plus burned-in-pixel PHI removal.
- Data provenance metadata: site, scanner vendor/model, acquisition dates, patient-demographic fields where lawful, to support subgroup analysis.
- Radiologist reports and priors included (needed for Stage 5 and for longitudinal comparison).
- Audit and deletion obligations, breach notification, data-residency constraints.
- Publication / regulatory-submission use of results.

## Sources

- PhysioNet Credentialed Health Data License 1.5.0 (MIMIC-CXR): https://physionet.org/content/mimic-cxr/view-license/2.1.0/
- Stanford AIMI commercial use: https://aimi.stanford.edu/datasets-commercial-use
- CheXpert Plus paper: https://arxiv.org/html/2405.19538
- NIH CXR14 FAQ: https://nihcc.app.box.com/v/ChestXray-NIHCC/file/249502714403
- ChestX-Det: https://github.com/Deepwise-AILab/ChestX-Det-Dataset
- SIIM-ACR rules: https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/rules
- RSNA Pneumonia terms: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/pneumonia-detection-challenge-terms-of-use-and-attribution.pdf
- RSNA ICH terms: https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/rsna_ich_ai_challenge_2019_terms_of_use_and_attribution_final.pdf
- RSNA PE rules: https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/rules
- VinBigData rules: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/rules
- PadChest RUA: https://bimcv.cipf.es/?p=4923
- BIMCV-COVID19+: https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
- BRAX: https://physionet.org/content/brax/1.1.0/
- MIDRC FAQ: https://www.midrc.org/faqs
- TCIA BraTS 2021: https://wiki.cancerimagingarchive.net/x/sIbECQ
- MSD: http://medicaldecathlon.com
- Object-CXR: https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
- CANDID-PTX: https://figshare.com/articles/dataset/CANDID-PTX/14173982
- Shenzhen/Montgomery: https://qims.amegroups.org/article/view/5132/html
- CDSCO MDSW guidance summary: https://natlawreview.com/article/code-compliance-decoding-indias-medical-device-software-guidance
- DPDP Rules 2025: https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf
- ICMR AI ethics guidelines: https://www.icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare
- EHDS: https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
- MedPerf: https://github.com/mlcommons/medperf ; NVFlare: https://github.com/NVIDIA/NVFlare ; Flower: https://github.com/flwrlabs/flower

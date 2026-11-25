# 16 — Risk Register

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

**Scales.**

- **Likelihood (L):** 1 rare · 2 unlikely · 3 possible · 4 likely · 5 almost certain.
- **Impact (I):** 1 minimal · 2 minor · 3 moderate · 4 major · 5 severe.
- **Score = L×I.** ≥ 15 red, 8–14 amber, ≤ 7 green.

Scores are initial estimates, reviewed monthly. Patient-safety hazards are analysed in detail in the ISO 14971 table in 09-safety-validation.md. This register tracks programme-level risks and references those hazards.

| ID | Category | Risk | L | I | Score | Mitigation | Owner (role) | Trigger / early warning |
|---|---|---|---|---|---|---|---|---|
| R01 | Data / legal | Shipped weights trained on research-only data (MIMIC, CheXpert, VinDr, PadChest), making the product unlawful to sell or forcing a costly retrain | 3 | 5 | 15 | Licence class in the data registry; MLflow lineage propagation; registry blocks promotion of `research_only` lineage; counsel sign-off per dataset | Head of Data + Legal | Any training config referencing an unregistered dataset; lineage check failure |
| R02 | Data / legal | Not enough commercially licensed CXR data (especially with reports and boxes) | 4 | 5 | 20 | Early partnerships in US, EU and India with DUAs; MIDRC commercial agreement; RSNA confirmation; budget for Stanford AIMI licence if justified; federated training (NVFlare/Flower) where data cannot leave a site | CEO / BD + Head of Data | No signed DUA by month 6 |
| R03 | Data / legal | Ambiguous licence terms (RSNA Pneumonia sheet vs terms; RAD-DINO card; HAI-DEF; MedImageInsight) | 4 | 3 | 12 | Written confirmations from licensors; default to exclusion until confirmed | Legal | Open licence questions at a phase gate |
| R04 | Data / privacy | PHI leak via burned-in text, headers or reports in training data | 2 | 5 | 10 | On-site de-identification (DICOM PS3.15 Annex E), OCR burned-in checks, report scrubber + sample audit; research environment isolated | DPO / Security lead | De-id audit finds PHI |
| R05 | Clinical | False negatives on critical findings plus automation bias (H1) | 3 | 5 | 15 | Conformal α_FN thresholds; "no acute" rule; UI approval gate; MRMC automation-bias probes; post-market monitoring | Clinical lead | Silent-mode misses; reader-study over-reliance signal |
| R06 | Clinical | Hallucinated or contradictory text in drafts (H3, H11) | 2 | 4 | 8 | IR-only generation; validator; templates default; gold QA set | NLP lead | Any validator escape found in QA |
| R07 | Clinical | Laterality errors (H4) | 2 | 5 | 10 | Geometry-based laterality; no flip augmentation; validator; orientation markers | CV lead | Laterality accuracy < 0.98 |
| R08 | Clinical | Subgroup performance gaps (age, sex, ethnicity, portable AP, devices) (H13) | 3 | 4 | 12 | Diverse sites; subgroup gates in SAP; targeted data acquisition | Clinical scientist | Fairness flag in evaluation |
| R09 | Technical | Label noise from NLP-mined labels inflates or deflates metrics, leading to wrong model choices | 4 | 3 | 12 | Radiologist gold validation/test sets; label-noise audit | ML lead | Large gap between NLP-label and gold-label metrics |
| R10 | Technical | Shortcut learning (chest tubes, text markers, hospital-specific cues) | 4 | 4 | 16 | Saliency audits; stratified evaluation by device presence; multi-site training; marker cut-out augmentation | ML lead | Performance collapses on tube-free positives |
| R11 | Technical | Domain shift at new sites or after scanner changes (H12) | 4 | 4 | 16 | OOD gate; site acceptance tests; drift monitoring; PCCP recalibration | ML-Ops lead | PSI > 0.2; OOD rate spike |
| R12 | Technical | Mac-only compute and ~41 GB free disk slow Phase 1 | 5 | 2 | 10 | Subset smoke runs; external SSD; early small cloud budget (14-infrastructure-costs.md) | Eng manager | Disk < 10 GB free; runs > overnight |
| R13 | Technical | Dependency abandonment (MONAI Deploy slowing; TorchServe archived; MMDetection stale) | 3 | 2 | 6 | Prefer actively maintained components (Triton, ONNX Runtime); thin adapters; vendor-neutral model packaging | Architect | No upstream release in 12 months |
| R14 | Technical / legal | Copyleft contamination (AGPL Ultralytics/Orthanc plugins/Kaapana; SSPL clearml-server; GPL) | 3 | 4 | 12 | Licence scanner in CI blocks AGPL/SSPL/GPL/NC in shipped artefacts; Orthanc only as an unmodified sidecar | Architect + Legal | Scanner finding |
| R15 | Regulatory | No predicate for image-based AI report drafting, so a De Novo is likely (longer, $191,020 FY2027 fee) | 4 | 4 | 16 | Split submissions: CADe module via 510(k) (predicates exist), report drafting via De Novo or a later submission; FDA Pre-Submission (Q-Sub) early | RA lead | Q-Sub feedback |
| R16 | Regulatory | EU MDR IIb Notified Body capacity and timelines; AI Act obligations from 2 Aug 2028; proposed MDR revision (COM(2025) 1023) changes Rule 11 | 4 | 4 | 16 | Engage a Notified Body early; build AI Act technical documentation alongside MDR; monitor the revision | RA lead | NB quote and lead time |
| R17 | Regulatory | India: CDSCO class uncertainty (likely C, unverified); Indian-population validation requirement; DPDP obligations from 13 May 2027 | 3 | 3 | 9 | Indian clinical partner; early CDSCO consultation; DPDP compliance programme | RA lead + DPO | Guidance update; partner delay |
| R18 | Regulatory | Claiming clinical readiness or clearance prematurely | 2 | 5 | 10 | Claims review procedure; "not cleared / not for clinical use" labelling on all prototypes and docs | RA lead + Marketing | Any external material without RA review |
| R19 | Security | Compromise of the inference path or model tampering (H15) | 2 | 5 | 10 | Signed artefacts, mTLS, SBOM, pen-tests, 524B documentation, egress-deny | Security lead | Pen-test high finding; CVE in dependency |
| R20 | Security | Prompt injection through clinical-history text into the LLM rephraser | 2 | 3 | 6 | LLM sees only IR JSON and template text; history passed as quoted data; validator post-check; rephraser off by default | NLP lead | Red-team finding |
| R21 | Commercial | Incumbents (Annalise/Harrison, Qure.ai, Lunit, Aidoc, Microsoft/Nuance) out-compete on breadth and distribution | 4 | 4 | 16 | Focus on grounded, auditable draft reports; integration depth (IHE AIR, OHIF review); India/EU cost-efficient deployment; partner with reporting vendors | CEO | Lost pilots; pricing pressure |
| R22 | Commercial | Radiologist adoption is low because drafts are too stiff or slow | 3 | 4 | 12 | Measure acceptance and edit distance in P4/P6; template iteration with radiologists; optional rephraser | Product lead | Acceptance < target |
| R23 | Commercial | Reimbursement and pricing uncertainty for AI drafting | 4 | 3 | 12 | Value cases on time-to-sign and quality; per-study vs subscription pricing pilots | CEO / Finance | Pilot ROI unclear |
| R24 | People | Radiologist annotator availability and cost (≈ 2,550·R hours, see 14 §6) | 4 | 3 | 12 | Multi-market annotator pool; efficient tooling (MONAI Label); prioritise test sets over training labels | Clinical ops | Annotation throughput below plan |
| R25 | Governance | Use of radiologist corrections for retraining without proper basis | 2 | 4 | 8 | Default `corrections_for_training=false`; governance board approval; contract and consent terms | DPO | Request to use field data |

## Top risks (score ≥ 15)

R02 (licensed data access), R10 (shortcut learning), R11 (domain shift), R15 (FDA pathway for report drafting), R16 (EU NB / AI Act), R21 (competition), R01 (research-only data in shipped weights), R05 (false negatives + automation bias).

## Sources

- Dataset licences and terms: 04-dataset-licensing-matrix.md
- Regulatory facts (De Novo fee, AI Act dates, MDR revision proposal, CDSCO guidance, DPDP dates): 17-commercial-regulatory.md and its sources
- Open-source licence traps and maintenance status: 10-system-architecture.md §9 (GitHub API data 2026-11-25)
- Hazard analysis: 09-safety-validation.md

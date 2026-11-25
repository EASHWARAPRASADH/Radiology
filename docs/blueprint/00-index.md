# 00 — Radio R&D Blueprint: Index and Executive Summary

Status: draft v0.1 — 2026-11-25 · Research prototype; nothing here is a claim of clinical readiness or regulatory status.

## Executive summary

**Goal.** Build a radiologist-in-the-loop imaging AI platform. It detects and localises chest X-ray abnormalities, writes them into a structured findings representation, and drafts an evidence-grounded report that a radiologist must approve. Chest X-ray comes first. CT and MRI plug in later through the same `Analyzer` contract.

**Where we differ.** Rad AI, the stated inspiration, is text-only: it drafts impressions from findings the radiologist has dictated (17 §6). Our product reads the pixels:
- image models produce a validated findings IR;
- a constrained text layer verbalises that IR, and only that IR.

The closest analogues, such as Aidoc's chest X-ray report drafting tool and Harrison.ai's Annalise Reporting, are either under Breakthrough designation or not marketed in the US. No FDA-cleared image-based AI report generator was found (17 §2). Expect a **De Novo** path for report drafting. The detection and localisation module has 510(k) predicates.

**The six decisions that shape everything else:**

| # | Decision | Why | Where |
|---|---|---|---|
| D1 | Hierarchical pipeline (perception → IR → language). No end-to-end VLM in the product path | Faithfulness by construction. Every sentence cites IR findings and a rule validator fails closed. Perception and language can be validated and changed separately (supports a PCCP) | 06 |
| D2 | Tri-state+ findings (`present / absent / uncertain / not_assessed`) with provenance, enforced in the schema | Missing ≠ negative. Uncertainty is kept and deferred to the radiologist. Directly observed, model-inferred, clinical-history and prior-report content are kept apart | 06 §4.3, `src/radio/schema/findings.py` |
| D3 | Licence-clean training: DINOv2 (Apache-2.0) plus commercially usable data only in shipped weights | Nearly every strong CXR model or dataset is research-only (MIMIC-CXR, CheXpert, VinDr, PadChest; CheXagent, MAIRA-2, LLaVA-Rad, BiomedCLIP). Weights inherit data terms. Enforced by a licence gate in training | 04, 05 §7, `src/radio/data/registry.py` |
| D4 | Calibrated decisions with finite-sample error control. Two conformal thresholds per finding (FN-controlled `t_absent`, FP-controlled `t_present`). Under-validated findings → `not_assessed` | Clinically meaningful operating points, explicit deferral, honest "not assessed" | 06 stage 9, 08 §1, `src/radio/calibrate/` |
| D5 | Abstain on anything outside scope: wrong modality, view or body part; non-diagnostic quality; out-of-distribution embeddings | Safety and regulatory hygiene. Abstentions produce no draft | 09, `src/radio/pipeline/run.py` |
| D6 | Reuse open-source infrastructure, build the AI-specific gaps. Use OHIF v3 (MIT), highdicom, pydicom, dcm4chee / an unmodified Orthanc sidecar, MONAI, MLflow. Avoid AGPL (Ultralytics, Orthanc plugins) and non-commercial weights | Speed without licence contamination. Our IP is clean weights, the quality/OOD gate, grounded reporting, the review workflow and monitoring | 10 §9 |

**Starting position (2026-11-25):**
- Public data only.
- One Apple M4 Pro (24 GB, about 41 GB free disk).
- No credentialed datasets, partners, QMS or annotators yet.
- Target markets: US, EU and India.

Phase 1 can therefore show the *system* working end to end, plus a feasibility signal on NIH ChestX-ray14. It cannot show clinical performance. Partner data, radiologist annotation and cloud GPUs gate Phases 2–4 (01, 14, 15).

## Deliverable map (brief §13)

| # | Deliverable | Document / artefact |
|---|---|---|
| 1 | Technical feasibility report | [01-technical-feasibility.md](01-technical-feasibility.md) |
| 2 | Literature review | [02-literature-review.md](02-literature-review.md) |
| 3 | Dataset comparison table | [03-datasets.md](03-datasets.md) |
| 4 | Dataset licensing matrix (+ acquisition strategy) | [04-dataset-licensing-matrix.md](04-dataset-licensing-matrix.md), [`NOTICE-DATA.md`](../../NOTICE-DATA.md) |
| 5 | Model architecture comparison | [05-model-comparison.md](05-model-comparison.md) |
| 6 | Recommended AI architecture (pipeline, multimodal reasoning, Analyzer interface) | [06-recommended-architecture.md](06-recommended-architecture.md) |
| 7 | Training strategy | [07-training-strategy.md](07-training-strategy.md) |
| 8 | Evaluation protocol (incl. MRMC reader study, external validation) | [08-evaluation-protocol.md](08-evaluation-protocol.md) |
| 9 | Safety and validation plan | [09-safety-validation.md](09-safety-validation.md) |
| 10 | System architecture (diagrams, services, serving, security, deployment, UI, reuse map) | [10-system-architecture.md](10-system-architecture.md) |
| 11 | API specification | [11-api-spec.md](11-api-spec.md), [schemas/openapi.json](schemas/openapi.json) |
| 12 | Database design | [12-database-design.md](12-database-design.md) |
| 13 | Prototype implementation plan (+ Phase-1 results) | [13-prototype-plan.md](13-prototype-plan.md) |
| 14 | GPU and infrastructure cost estimates | [14-infrastructure-costs.md](14-infrastructure-costs.md) |
| 15 | Development roadmap (Phases 1–7) | [15-roadmap.md](15-roadmap.md) |
| 16 | Risk register | [16-risk-register.md](16-risk-register.md) |
| 17 | Commercialisation and regulatory considerations (US / EU / India) | [17-commercial-regulatory.md](17-commercial-regulatory.md) |
| — | Structured output specification (brief §7) | [schemas/analysis-result.schema.json](schemas/analysis-result.schema.json), generated from `src/radio/schema/findings.py` |

## Reading order

| Reader | Docs, in order |
|---|---|
| Engineers | 06 → 13 → 11 → 12 → 10 → 07 → 08 |
| Clinical and regulatory | 09 → 08 → 17 → 16 |
| Leadership | this page → 01 → 15 → 14 → 16 |

## How claims are handled

- Every licence, dataset size, regulatory fact, price and paper is cited in its document's **Sources** section.
- Anything that couldn't be confirmed from a primary or official source on 2026-11-25 is marked **"unverified"**. About 50 such flags appear across the set, concentrated in 03, 04, 05 and 17.
- Engineering numbers (GPU-hours, durations, team sizes, annotation volumes, acceptance thresholds) are **estimates or proposals with their assumptions shown**, never measured results.
- Before any of these docs is used externally, the unverified items in 04 (licences) and 17 (regulatory) need a counsel / regulatory-affairs pass. Neither document is legal advice.

## Key open items (owner: to assign)

1. **Licensing confirmations.**
   - Get written RSNA confirmation of commercial use for Pneumonia 2018.
   - Sign the MIDRC commercial DUA.
   - Get a counsel opinion on CC-BY-SA (MSD) and on the weight-taint position.
   - Resolve the conflicting RAD-DINO / MedImageInsight / HAI-DEF terms.
2. **Data partners** in the US, EU and India: DUAs, de-identification, Indian-population data for CDSCO.
3. **Radiologist panel:** annotation protocol and MRMC readers (07, 08 §6).
4. **QMS and IEC 62304 process setup** before design freeze (17 §5).
5. **Regulatory pre-submissions:**
   - FDA Q-Sub on the report-drafting De Novo strategy and PCCP scope.
   - Notified Body selection (MDR IIb + AI Act from 2 Aug 2028).
   - CDSCO classification confirmation.

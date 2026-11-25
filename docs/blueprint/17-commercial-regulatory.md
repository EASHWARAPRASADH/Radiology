# 17 — Commercialisation and Regulatory Considerations

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

> **Radio has no regulatory clearance, approval or CE marking in any market and is not clinically validated.** This document is a planning analysis, not legal or regulatory advice. Facts were checked on 2026-11-25 against the cited sources. Items marked *unverified* must be confirmed with regulatory counsel and, where possible, through formal pre-submission meetings.

## 1. Product definition for regulatory purposes

Radio has two functional modules, which may be submitted separately:

| Module | Function | Regulatory character |
|---|---|---|
| **M1 — Detect & localise** | Identifies and marks specified CXR findings with calibrated confidence; optional critical-finding prioritisation | Computer-assisted detection (CADe), optionally triage (CADt) |
| **M2 — Draft report** | Generates a preliminary report from M1's validated findings IR, for radiologist review and approval | Image-based report drafting. **No cleared or authorised predicate found in the US** |

Software that analyses medical images fails criterion 1 of FDA's Clinical Decision Support (CDS) guidance, so **both modules are device functions**. By contrast, Rad AI's text-only impression generation (from dictated findings) may fall outside the device definition. That is an inference, and the revised CDS guidance is silent on AI specifically.

## 2. United States (FDA)

| Item | Finding |
|---|---|
| M1 classification | 21 CFR 892.2070 *Medical image analyzer*, product code **MYN**, Class II (down-classified from III in 2020) → **510(k)** |
| Triage option | 21 CFR 892.2080: **QAS** (notification/triage, no image marking), **QFM** (prioritisation of lesions; e.g. Lunit INSIGHT CXR Triage K211733) |
| M1 predicates (verified) | Qure.ai **qXR-Detect K251934** (CADe under 892.2070, cleared 16 Jan 2026 **with a PCCP**; predicates include qXR-LN K231805); Annalise Enterprise **K250831** (CXR triage, 5 findings); Lunit **K211733** (QFM). DeepHealth/Gleamer Chest XRay CADe (Sep 2026) reported in trade press; K-number unverified |
| M2 pathway | No classification regulation fits image-based AI report generation, and no cleared product was found. **De Novo is likely** (inference, not an FDA position). The closest precedent is Aidoc's CXR draft-report tool, which received Breakthrough Device designation in Jun 2026 and is not cleared. Use a **Pre-Submission (Q-Sub)** early |
| Evidence | Standalone performance (sensitivity, specificity, localisation, subgroups by site, scanner and demographics); **MRMC reader study** (aided vs unaided) per FDA's CADe clinical performance guidance; for M2, likely an error taxonomy (hallucination, omission, laterality), edit-rate and reader-study endpoints (inference) |
| PCCP | Final guidance (first issued 4 Dec 2024; FDA page shows Aug 2025 version). Plan: retraining on additional data, per-site recalibration within bounds, adding supported manufacturers (09-safety-validation.md §8) |
| AI lifecycle guidance | *AI-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*: **still draft** (Jan 2025); on CDRH FY2027 priority list |
| Cybersecurity | Section 524B; premarket cybersecurity guidance final 27 Jun 2025 (a Feb 2026 reissue is unverified): SBOM, threat model, vulnerability management |
| Quality system | **QMSR effective 2 Feb 2026**, incorporating ISO 13485:2016 by reference |
| Fees (FY2027) | 510(k) **$28,653** (small business $7,163); De Novo **$191,020** (small business $47,755) |
| Review timelines | Not sourced; ask in Q-Sub |

**Recommended US sequence.** (1) Q-Sub covering M1 + M2 strategy → (2) 510(k) for M1 (CADe, possibly with triage) with a PCCP → (3) De Novo for M2 once reader-study evidence for drafting exists. Until M2 is authorised, US deployments ship M1 only, plus a non-device workflow (e.g. radiologist-dictated structured templates without AI image interpretation) if counsel agrees.

## 3. European Union

| Item | Finding |
|---|---|
| Classification | MDR Rule 11: software informing diagnostic decisions is ≥ IIa; **IIb** if decisions may cause serious deterioration of health. Precedents: Annalise CXR **IIb**, Oxipit ChestLink **IIb**. **Plan for IIb** → Notified Body assessment of QMS + technical documentation |
| Guidance | MDCG 2019-11 Rev.1 (17 Jun 2025) on software qualification and classification, now with explicit AI coverage |
| Pending change | Commission proposal to revise MDR/IVDR (COM(2025) 1023, 16 Dec 2025) would amend Rule 11 and move MDR/IVDR to AI Act Annex I Section B. **Still a proposal** |
| EU AI Act | AI that is (a safety component of) an MDR device requiring Notified Body involvement is **high-risk** (Art. 6(1), Annex I Section A). Reg. (EU) **2026/1744** (Digital Omnibus on AI; OJ 24 Jul 2026, in force 27 Jul 2026) moved application of high-risk obligations for Annex I products to **2 Aug 2028** (Annex III: 2 Dec 2027). The Notified Body assesses AI Act conformity with the MDR conformity assessment |
| AI Act obligations to build for | Risk management, data governance (training/validation/test data quality, bias examination), technical documentation, logging, transparency to deployers, human oversight, accuracy/robustness/cybersecurity, QMS, post-market monitoring. Most are already covered by 07–10 if documented in AI Act terms |
| Data | GDPR Art. 6 + Art. 9(2) basis, DPIA, pseudonymisation; the Omnibus adds a legal basis for processing to detect and correct bias. EHDS (Reg. (EU) 2025/327) generally applies from 26 Mar 2027; secondary-use rules mostly from 26 Mar 2029 |
| Evidence | Clinical evaluation per MDR Art. 61 / Annex XIV: performance (standalone + reader study) and clinical benefit; PMCF plan |

## 4. India

| Item | Finding |
|---|---|
| Framework | Medical Devices Rules 2017; risk classes A–D |
| Software guidance | CDSCO final *Medical Device Software* guidance (CDSCO/MD/GD/MDSW/01/2026), **21 Jul 2026**: classification via a significance × criticality matrix. AI expectations: disclose method and degree of autonomy; address bias, drift and hallucination; Algorithm Change Protocol where needed; disclose training and test data sources; **validate in Indian populations**; post-market AI performance monitoring |
| Likely class | **C (unverified)**; the matrix was not reproduced in the source reviewed |
| Licensing route | Import: **MD-14 → MD-15** via SUGAM, through the Central Licensing Authority, with an Indian Authorised Agent. Manufacture in India (Class C/D): **MD-7 → MD-9** (central). Exact use of MD-3/MD-5 for Class A/B is unverified. Whether Rule 63 allows waiving clinical investigation for devices approved in reference countries is **unverified** |
| Data protection | DPDP Act 2023; DPDP Rules 2025 notified 13–14 Nov 2025 with phasing: Board provisions immediate; consent managers 13 Nov 2026; **core fiduciary obligations 13 May 2027**. Whether the Data Protection Board has been constituted is unverified |
| Ethics | ICMR Ethical Guidelines for Application of AI in Biomedical Research and Healthcare (2023) govern validation studies |
| Interoperability | ABDM / NRCeS FHIR R4 IG (DiagnosticReport imaging profile), ABHA linkage. Relevant for hospital customers acting as Health Information Providers; not an entry requirement |

## 5. Cross-market QMS and standards checklist

| Standard / framework | Use in Radio | Status |
|---|---|---|
| ISO 13485:2016 | QMS (FDA QMSR, EU NB, CDSCO) | To establish in Phase 1–2 |
| IEC 62304 | Software lifecycle; likely software safety class **B** (to be confirmed by risk analysis) | Process docs + traceability |
| ISO 14971 | Risk management, incl. AI-specific hazards (09) | Risk file started |
| IEC 62366-1 | Usability, esp. the review/approval and automation-bias scenarios | Formative in P3–P5, summative in P4–P5 |
| IEC 82304-1 | Health software product safety | P5 |
| IEC 81001-5-1 | Secure health software lifecycle (supports 524B) | P2 onwards |
| IMDRF SaMD + GMLP N88 (final 27 Jan 2025); N93 draft (Apr 2026) | ML practice principles, lifecycle | Mapped in technical file |
| DICOM SR TID 1500, SEG, GSPS | Output encoding | Implemented with highdicom |
| IHE AI Results (AIR) / AIRA | Interop of AI results and radiologist assessment of them | Conformance testing with Gazelle AIR test data |
| HL7 FHIR R4 (DiagnosticReport, ImagingStudy, Observation) | Report and findings exchange | fhir.resources |

## 6. Competitive landscape (verified facts only)

| Company / product | What it does | Regulatory status (verified) | Implication for Radio |
|---|---|---|---|
| **Rad AI** (Omni, Omni Reporting, Continuity) | **Text-only**: impressions from radiologist-dictated findings; reporting platform; follow-up tracking. Does not interpret images | No FDA clearance found; likely outside device definition (inference) | Radio's differentiator: **image-grounded findings → grounded draft**, with sentence-to-evidence traceability. Partnering or interoperating with text-reporting vendors is possible |
| **Annalise.ai / Harrison.ai** | CXR detection and triage; Annalise Reporting drafts reports; Harrison.rad.1.5 report-drafting foundation model | FDA triage clearances (e.g. K250831); CE IIb; Annalise Reporting not available in the US; Harrison.rad research-only | Strongest direct competitor in EU/APAC; Radio must compete on auditability, integration and price |
| **Qure.ai qXR** | CXR detection, localisation, triage, nodule quantification | FDA K251934 (CADe + PCCP, Jan 2026), K231805 | Strong in India and globally; sets the PCCP precedent |
| **Lunit INSIGHT CXR** | CXR detection/triage | FDA K211733 (QFM); CE MDR (CXR4) | Incumbent in Korea, EU and US triage |
| **Oxipit ChestLink** (Sectra acquisition per Sectra marketplace, Apr 2026) | Autonomous reporting of normal CXRs | CE IIb (Mar 2022); no FDA clearance found | Shows the autonomy frontier in the EU; Radio stays human-in-the-loop by design |
| **Microsoft / Nuance** (PowerScribe, Dragon Copilot, MAIRA, CXRReportGen) | Reporting platform with impression drafting; research CXR report models; CXRReportGen Premium in private preview | No FDA clearance found for MAIRA/CXRReportGen | Both a distribution channel (PowerScribe integration) and a future competitor |
| **Google** (CXR Foundation, MedGemma) | Developer models under HAI-DEF terms | Not devices | Potential component supplier (subject to HAI-DEF terms) |
| **Aidoc** (CARE, First Read) | CT triage foundation model; CXR draft reports | CARE 510(k) K252970 (Jan 2026); First Read: Breakthrough designation only (Jun 2026) | Likely first mover for FDA-authorised CXR drafting; watch its De Novo outcome as the pathway signal |

## 7. Positioning

1. **Against Rad AI (text-only):** Radio adds image understanding. The draft exists before dictation, and every sentence links to a box or mask plus a calibrated probability. Pitch: "evidence-grounded draft, not autocomplete".
2. **Against image-AI vendors (Annalise, Qure, Lunit):** Radio's moat is the **reporting layer**:
   - validated IR → grounded draft;
   - per-finding accept/reject inside an OHIF-based review;
   - SR/FHIR/IHE AIR outputs;
   - audit-grade traceability.
   Detection quality must be at parity on the claimed findings.
3. **Market sequencing (proposal):**
   - **India** first for pilots (partner access, cost-sensitive teleradiology), subject to CDSCO licensing.
   - **EU** CE IIb next.
   - **US** 510(k) for M1, then De Novo for M2.
   - Final sequence depends on partner data and regulatory consultation.

## 8. Commercialisation models

| Model | Description | Pros | Cons |
|---|---|---|---|
| Per-study fee | Charge per processed CXR | Aligns with volume; easy trials | Revenue volatility; needs metering + audit |
| Subscription per site / per radiologist seat | Annual licence by volume tier | Predictable | Harder to sell to small sites |
| Platform / marketplace | Distribution via PACS/reporting vendor marketplaces (e.g. Sectra Amplifier, reporting-vendor app stores) | Distribution, integration done once | Revenue share; vendor dependence |
| OEM / white-label | Embed M1/M2 into a reporting or teleradiology product | Scale | Brand loss; partner regulatory responsibilities must be contractually allocated |
| Teleradiology partnership | Joint offering with a teleradiology group (draft-first workflow) | Fast real-world evidence; India/US demand | Concentration risk |

Contracts must cover: BAA (US) / DPA (EU, India), data residency, no training use of customer data without explicit agreement, intended-use and labelling obligations, incident reporting cooperation, and licence pass-through if any third-party model terms (e.g. HAI-DEF) are used.

## Sources

- FDA AI-enabled device list: https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
- Product codes QAS / QFM: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpcd/classification.cfm?id=QAS · https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfTPLC/tplc.cfm?id=QFM
- 892.2070 down-classification FR notice (2020): https://www.govinfo.gov/content/pkg/FR-2020-01-22/pdf/2020-00494.pdf
- K251934: https://www.accessdata.fda.gov/cdrh_docs/pdf25/K251934.pdf · K211733: https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K211733 · K250831: https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250831.pdf
- DeepHealth Chest XRay (trade press): https://www.auntminnie.com/clinical-news/digital-x-ray/news/15835309/deephealth-deephealth-gets-fda-nod-for-chest-xray-ai-tool
- Aidoc Breakthrough designation: https://www.medtechdive.com/news/aidoc-wins-breakthrough-nod-for-ai-that-reads-chest-x-rays/823895/ · https://www.mobihealthnews.com/news/aidoc-lands-fda-breakthrough-status-ai-drafts-radiology-reports · Aidoc CARE K252970: https://www.aidoc.com/about/news/aidoc-secures-fda-clearance-for-healthcares-first-comprehensive-foundation-model-ai/
- PCCP guidance: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence · FR 4 Dec 2024: https://www.govinfo.gov/content/pkg/FR-2024-12-04/html/2024-28361.htm
- AI lifecycle draft guidance: https://fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing · CDRH FY2027 agenda: https://www.raps.org/resource/fda-s-device-center-releases-guidance-agenda-for-fy-2027.html
- CDS guidance (revised Jan 2026): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software · analysis: https://www.cooley.com/news/insight/2026/2026-01-20-automation-bias-and-clinical-practice-fda-makes-incremental-updates-to-clinical-decision-support-software-guidance · https://www.cov.com/en/news-and-insights/insights/2026/01/5-key-takeaways-from-fdas-revised-clinical-decision-support-cds-software-guidance
- Cybersecurity guidance FR 27 Jun 2025: https://www.govinfo.gov/content/pkg/FR-2025-06-27/html/2025-11669.htm
- QMSR: https://www.fda.gov/quality-systems-regulation
- FY2027 user fees: https://www.federalregister.gov/documents/2026/07/30/2026-15335/medical-device-user-fee-rates-for-fiscal-year-2027
- MDCG 2019-11 Rev.1: https://health.ec.europa.eu/latest-updates/update-mdcg-2019-11-rev1-qualification-and-classification-software-regulation-eu-2017745-and-2025-06-17_en
- Annalise CXR (RCR AI registry): https://www.rcr.ac.uk/our-services/artificial-intelligence-ai/ai-registry/annalise-enterprise-cxr
- MDR/IVDR revision proposal analyses: https://www.eversheds-sutherland.com/en/united-kingdom/insights/proposed-eu-mdr-ivdr-revision-a-regulatory-system-at-a-crossroads · https://www.aoshearman.com/en/insights/ao-shearman-on-life-sciences/eu-commission-proposes-much-anticipated-mdr-and-ivdr-revision
- Reg. (EU) 2026/1744: https://eur-lex.europa.eu/eli/reg/2026/1744/oj?locale=en · Omnibus GDPR bias basis: https://www.cuatrecasas.com/en/spain/intellectual-property/art/digital-omnibus-ai-has-been-published
- EHDS: https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
- CDSCO MDSW guidance summary: https://natlawreview.com/article/code-compliance-decoding-indias-medical-device-software-guidance · draft: https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/UploadPublic_NoticesFiles/Draft%20guidance%20document%20on%20Medical%20Device%20Software%2021%2010%202025.pdf · MDR 2017: https://cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/medical-device/RegulatoryMDR-2017.pdf
- DPDP Rules 2025: https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf · https://www.mondaq.com/india/data-protection/1708434/operationalising-indias-data-protection-regime-analysing-the-newly-notified-dpdp-rules-2025
- ICMR AI guidelines: https://www.icmr.gov.in/ethical-guidelines-for-application-of-artificial-intelligence-in-biomedical-research-and-healthcare · NRCeS FHIR IG: https://nrces.in/ndhm/fhir/r4/index.html
- IMDRF GMLP N88: https://imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf · N93 draft: https://www.imdrf.org/sites/default/files/2026-04/IMDRF%20AIML%20WG%20N93%20DRAFT%202026%20-%20Technical%20Framework%20for%20Artificial%20Intelligence%20Life%20Cycle%20Management.pdf
- IHE AIR: https://wiki.ihe.net/index.php/AI_Results
- Rad AI: https://www.radai.com/omni · https://radai.com/blogs/rad-ai-radiology-workflow-focused-ai
- Harrison.ai: https://harrison.ai/harrison-ai-draft-report-generation/ · Qure.ai: https://www.qure.ai/news-press-coverages/qure.ai-nets-six-new-indications-cleared-by-the-fda-taking-the-chest-x-ray-crown-in-the-radiology-ai-race · Lunit CE: https://www.auntminnie.com/clinical-news/digital-x-ray/news/15747118/lunit-lunit-updates-chest-xray-software-secures-ce-mark · Oxipit CE: https://www.einpresswire.com/article/566851839/oxipit-awarded-ce-mark-for-the-first-autonomous-ai-medical-imagingapplication · Sectra marketplace: https://amplifiermarketplace.sectra.com/vendor/oxipit/ · Microsoft CXRReportGen: https://ai.azure.com/catalog/models/CxrReportGen · MAIRA: https://www.microsoft.com/en-us/research/project/project-maira/ · Google CXR Foundation / TB: https://research.google/blog/bringing-foundational-models-to-the-front-lines-of-tb-screening

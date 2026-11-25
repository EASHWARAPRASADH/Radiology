# 08 — Evaluation and Benchmarking Protocol

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

Evaluation runs at four levels:

- **L1:** standalone perception (classification, detection, segmentation).
- **L2:** IR and calibration.
- **L3:** report generation.
- **L4:** clinical (reader studies and prospective silent-mode).

Code lives in `src/radio/eval/metrics.py` (L1–L2) and `src/radio/eval/report_metrics.py` (L3). Reader-study analysis uses validated external tools (§6).

## 0. General rules

- **Unit of analysis.** The image/study for per-finding metrics. CIs come from a **patient-level cluster bootstrap**: resample patients with replacement, 2,000 replicates, percentile or BCa intervals. Multiple studies from one patient are correlated, and ignoring that understates the CI width.
- **Pre-registration.** Each validation study has a statistical analysis plan (SAP) frozen before unblinding. It fixes the primary endpoints, operating points, subgroups and success criteria.
- **Test-set hygiene.** Test sets are locked and accessed only by the evaluation service. Model developers never see test images or labels. Every access is logged.
- **No single headline number.** Results are always reported per finding, per site and per subgroup.
- **Labels.** Claims use radiologist-adjudicated reference standards (07-training-strategy.md, Stage 3 protocol). NLP-mined labels are only for development.

## 1. Image classification (per finding)

| Metric | Definition / notes |
|---|---|
| AUROC | Area under the ROC curve, with patient-cluster bootstrap CI |
| AUPRC | Area under the precision–recall curve. Report it alongside prevalence, since AUPRC depends on prevalence |
| Sensitivity, specificity | At the **locked operating point** (from validation or calibration data, never test), with Wilson CIs plus a cluster-bootstrap check |
| PPV / NPV | At the operating point, at observed prevalence and at a reference prevalence per intended setting (ED, inpatient, outpatient) |
| F1 | At the operating point, as a secondary measure only |
| Calibration | Expected Calibration Error (15 equal-mass bins); reliability diagram; Brier score; calibration slope and intercept (logistic recalibration) |
| Tri-state behaviour | Fraction of cases called `uncertain`; sensitivity and specificity among definite calls; **false-negative rate among `absent` calls** (the safety-critical quantity for "no acute finding" statements) |
| Abstention | Abstention rate overall and by reason; performance on abstained cases (measured offline) to confirm abstention is "doing work" |

### Operating-point selection

- **Default.** Conformal risk control on the calibration split, targeting a per-finding false-negative rate ≤ α_FN via `t_absent` (positives called absent) and a false-positive rate ≤ α_FP via `t_present` (negatives called present). Findings with too few calibration positives (default < 30) or negatives (< 100) are not validated and emitted as `not_assessed`.
- **Proposed α for critical findings** (pneumothorax, tension signs, malpositioned endotracheal tube, free air if added): α_FN = 0.05.
- **Proposed α for other findings:** α_FN = 0.10.
- **Alternatives** (for comparison only): Youden J; sensitivity fixed at 0.90 or 0.95; specificity fixed at 0.90.

## 2. Detection

| Metric | Definition |
|---|---|
| mAP | COCO-style mAP at IoU 0.5, plus mAP@0.25. CXR boxes are fuzzy, so IoU 0.25 is the more clinically meaningful threshold. Both are reported |
| FROC | Lesion-level sensitivity vs mean false positives per image. Report sensitivity at **0.25, 0.5, 1 and 2 FP/image** (clinically relevant FP rates) |
| Lesion localisation accuracy | Fraction of true positives whose predicted box centre falls inside the reference box or mask (pointing game), and laterality accuracy |
| Per-image detection | Sensitivity and specificity of "any box for finding k" vs image-level reference |
| Uncertainty and gating | 95% CI for FROC sensitivity from an **image-level** bootstrap: whole images are resampled so clustered lesions and their false positives move together (box-level binomial CIs are too narrow). A class is validated for localisation only if the **lower** CI bound meets the threshold (prototype: ≥ 0.80 at 1 FP/image for focal findings and at 0.25 FP/image for whole-image findings such as cardiomegaly, which occur at most once per image; ≥ 30 reference boxes); a point estimate above threshold is not enough. Implemented in `eval/detection.py::froc_sensitivity_ci` and `train/train_detector.py::gate_class` |

## 3. Segmentation

| Metric | Use |
|---|---|
| Dice / IoU | Overlap for lungs, heart, pneumothorax, devices |
| HD95, ASSD | Boundary error in mm (pixel spacing from DICOM `ImagerPixelSpacing`/`PixelSpacing`; NIH PNGs lack spacing, so only pixel units are available there) |
| Derived measurement error | CTR absolute error vs radiologist measurement; Bland–Altman plot |

## 4. Report generation (L3)

All report metrics are computed **against the IR and against radiologist reference reports**. Text-overlap metrics (BLEU, ROUGE, METEOR, CIDEr) may be reported for literature comparability but are **never** primary.

| Metric | Definition | Computed from |
|---|---|---|
| Finding-level precision / recall / F1 | Extract (concept, status, laterality) triples from the generated report and the reference: parse the draft's `finding_ids` for ours, use radiologist structured labelling for the reference. Match on concept + status (+ laterality where applicable) | IR + reference |
| Hallucination rate | % of reports containing ≥ 1 finding statement not supported by the reference standard image read (radiologist-adjudicated). Also reported per sentence | Gold QA review |
| IR-faithfulness violations | % of sentences whose content does not match the IR (should be 0 by construction; non-zero means a bug) | Automated validator |
| Contradiction rate | % of reports with internally contradictory statements, or statements contradicting the reference | Validator + gold QA |
| Unsupported-recommendation rate | % of reports containing a recommendation without a satisfied KB guideline | Validator + gold QA |
| Laterality error rate | % of lateralised findings with the wrong side | Gold QA |
| Omission of critical findings | % of reports where a reference-positive critical finding is absent from the draft impression | Gold QA |
| Radiologist edit distance | Normalised word-level Levenshtein distance between draft and signed report; plus **clinically significant edit rate** (edits tagged by the reviewer as changing meaning) | Production / silent-mode logs |
| Report acceptance rate | % of drafts signed with no clinically significant edit | Production / silent-mode logs |
| Time-to-sign | Median reading-plus-reporting time, AI-assisted vs unassisted | Reader study / pilot |
| Learned metrics (secondary, internal only) | RadGraph F1 (MIT code; checkpoint licence to check), RaTEScore (MIT), RadFact (MIT), CheXprompt (MIT). GREEN and CheXbert only in the research environment (licence issues) | Automated |

**Gold QA set.** ~300 studies (estimate) with two-radiologist error tagging using the taxonomy in 07-training-strategy.md, Stage 7. This set is the primary L3 test.

## 5. External validation design

| Dimension | Requirement |
|---|---|
| Institutions | ≥ 3 external institutions not used in training or calibration, including ≥ 1 per launch market (US, EU, India). Indian-population validation is an explicit CDSCO expectation in the 2026 MDSW guidance |
| Equipment | ≥ 3 manufacturers; CR and DX; fixed and portable units |
| Settings | Emergency, inpatient (portable AP heavy), outpatient (PA heavy) |
| Population subgroups | Age bands (18–40, 40–65, 65–80, > 80), sex, BMI where available, race/ethnicity where lawfully recorded, view (PA/AP), presence of support devices, image-quality class |
| Sampling | Consecutive or random sampling of the intended-use population for prevalence-faithful PPV/NPV, plus **enrichment** with positives for rare findings (enrichment reported and corrected by inverse-probability weighting) |
| Analysis | Per-site and pooled metrics; heterogeneity (I², random-effects meta-analysis across sites); subgroup AUROC and sensitivity at the locked operating point with CIs |
| Fairness criterion (proposal) | No subgroup with ≥ 100 positives has sensitivity more than 0.05 below the overall sensitivity with the CI excluding 0 → investigation and documented mitigation before release |
| Drift readiness | The same metrics are wired into post-market monitoring (09-safety-validation.md §6) |

**Research-only datasets as external tests.** MIMIC-CXR, CheXpert, VinDr-CXR and PadChest give multi-site research benchmarks. Their use is internal R&D evidence under their research licences. Whether research-licensed data can appear in a regulatory submission needs counsel review per dataset (unverified), so regulatory external validation is planned on partner data.

## 6. Blinded multi-reader multi-case (MRMC) reader study

**Purpose.** Show that radiologists aided by Radio (detection overlays + draft report) perform at least as well as unaided radiologists, which is the evidence pattern for CADe (FDA's CADe clinical performance guidance). For report drafting, also show non-inferior accuracy and reduced reporting time.

| Element | Design (proposal) |
|---|---|
| Design | Fully crossed, two-session, sequential-or-crossover. **Crossover recommended:** each reader reads each case once unaided and once aided, in random order, separated by a washout |
| Readers | 12–20 radiologists (mix of general and thoracic, varying experience, ≥ 1/3 from non-US sites for EU/India evidence). Sizing by simulation (iMRMC) from pilot variance components |
| Cases | ~300–500 studies, enriched (≈ 40–50% with ≥ 1 target finding), from sites not used in development. Reference standard: ≥ 3-reader adjudicated panel plus follow-up CT/clinical data where available |
| Washout | ≥ 4 weeks between sessions; case order randomised per reader and session |
| Blinding | Readers blinded to the reference standard, prevalence and other readers. The reference panel is independent of the study readers |
| Tasks | Per finding: presence (confidence 0–100), location (mark), laterality. Plus report: produce a signed report (aided session starts from the Radio draft) |
| Primary endpoint | Reader-averaged AUROC (or wAFROC for localisation-aware analysis) difference aided − unaided, for the claimed finding group |
| Secondary | Reader-averaged sensitivity/specificity at the clinical decision threshold; reporting time; clinically significant report error rate; inter-reader variability |
| Statistics | Obuchowski–Rockette (OR) method or Dorfman–Berbaum–Metz (DBM) with Hillis modifications, treating readers and cases as random. Analysis with FDA's **iMRMC** R package (CC0, CRAN) and/or RJafroc for FROC/wAFROC |
| Hypothesis | Superiority for detection AUROC (or non-inferiority margin δ = −0.02 with superiority test second, pre-specified hierarchy) |
| Automation-bias probes | ~5–10% of cases carry a deliberately removed true AI finding or an injected false AI finding (disclosed in consent, not per case). Measures over-reliance. Ethics approval required |

## 7. Prospective silent-mode evaluation (pre-pilot)

- Radio runs on live studies at a pilot site; outputs are hidden from clinicians.
- Signed radiologist reports become the comparison: CheXpert-labeler-style extraction (MIT labeler) plus weekly radiologist adjudication of discordant cases.
- Duration: ≥ 8 weeks or ≥ 5,000 studies (estimate).
- Endpoints: agreement, discordance analysis, abstention rate, latency, OOD rate, critical-finding PPV.

## 8. Acceptance thresholds (proposals, to be fixed in each SAP)

| Gate | Criterion (proposal) |
|---|---|
| Phase 1 exit (research prototype) | Pipeline reproducible; macro-AUROC on NIH CXR14 internal test reported with CIs. No clinical threshold, because NLP labels and a single site make this a feasibility gate only |
| Phase 2 exit (localisation) | FROC sensitivity ≥ 0.80 at 1 FP/image for pneumothorax and effusion on the ChestX-Det/SIIM held-out set; laterality accuracy ≥ 0.98 among true positives |
| Phase 3 exit (reporting) | IR-faithfulness violations = 0 on the 10k synthetic suite; hallucination rate ≤ 2% of reports and laterality error ≤ 0.5% of lateralised findings on the gold QA set; zero unsupported recommendations |
| Phase 4 exit (retrospective clinical) | For each claimed finding on external data: sensitivity lower 95% CI bound ≥ pre-specified target (e.g. 0.85 for pneumothorax) at specificity ≥ 0.85; ECE ≤ 0.05; no fairness flag unresolved; MRMC primary endpoint met |
| Phase 6 pilot continuation | Critical-finding miss rate in drafts not higher than the unaided baseline; clinically significant edit rate trending down; no serious incident attributable to AI |

## Sources

- Guo et al. 2017 (calibration/ECE): https://arxiv.org/abs/1706.04599
- Conformal risk control: https://arxiv.org/abs/2208.02814 · Learn-then-Test: https://arxiv.org/abs/2110.01052
- FDA iMRMC (R package CC0; Java apps frozen at v4.0.3): https://github.com/DIDSR/iMRMC · FDA tool page https://cdrh-rst.fda.gov/node/1301 · CRAN https://cran.r-project.org/package=iMRMC
- RJafroc (FROC/wAFROC analysis): https://github.com/dpc10ster/RJafroc (licence field not reported by the GitHub API; check DESCRIPTION; unverified)
- Dorfman, Berbaum, Metz. *Receiver operating characteristic rating analysis: generalization to the population of readers and patients with the jackknife method.* Invest Radiol 1992;27:723–731 (citation from memory; unverified this session)
- Obuchowski, Rockette. *Hypothesis testing of diagnostic accuracy for multiple readers and multiple tests: an ANOVA approach with dependent observations.* Commun Stat Simul Comput 1995;24:285–308 (citation from memory; unverified this session)
- FDA guidance *Clinical Performance Assessment: Considerations for Computer-Assisted Detection Devices Applied to Radiology Images and Radiology Device Data in Premarket Notification (510(k)) Submissions*, searchable at https://www.fda.gov/regulatory-information/search-fda-guidance-documents (current revision not re-checked this session)
- CDSCO Medical Device Software guidance, 21 Jul 2026 (Indian-population validation expectation), summarised at https://natlawreview.com/article/code-compliance-decoding-indias-medical-device-software-guidance
- Report metrics: RadFact https://github.com/microsoft/RadFact · RaTEScore https://github.com/MAGIC-AI4Med/RaTEScore · CheXprompt https://github.com/microsoft/chexprompt · RadGraph https://github.com/Stanford-AIMI/radgraph · ReXrank https://github.com/rajpurkarlab/ReXrank · GREEN (no licence file) https://github.com/Stanford-AIMI/GREEN · CheXbert (Stanford commercial licence) https://github.com/stanfordmlgroup/CheXbert · CheXpert labeler (MIT) https://github.com/stanfordmlgroup/chexpert-labeler

# 02 — Literature Review

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

Scope: work that directly informs design decisions for a CXR-first, radiologist-in-the-loop system. Every entry gives the citation, what the work showed, and what we take from it ("→ Radio"). We cite only works whose existence, venue and year were confirmed during this session's research or are canonical references. Benchmark numbers are quoted only where a source was checked; otherwise we describe results qualitatively. arXiv IDs are given where known.

---

## 1. CXR classification, datasets and label noise

| Work | Showed | → Radio |
|---|---|---|
| Wang et al., *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks…*, CVPR 2017 | Released NIH CXR14. 112,120 frontal images, 30,805 patients, 14 labels **text-mined from reports**, ~1,000 boxes. | Our main **commercially usable** training set. Its labels are weak supervision, not ground truth. |
| Irvin et al., *CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison*, AAAI 2019 | 224,316 images / 65,240 patients. Rule-based labeler with an explicit **uncertain** class. Studied U-Ignore, U-Ones and U-Zeros policies. Radiologist-labelled val/test sets. | Model "uncertain" as a first-class state, separate from absent. Research-only licence, so benchmark only. |
| Johnson et al., *MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports*, Scientific Data 2019 | 377,110 images / 227,835 studies with reports, single US centre. | De facto benchmark for report generation. PhysioNet licence limits it to research, so **never in shipped weights**. |
| Nguyen et al., *VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations*, Scientific Data 2022 | 18,000 images. 22 local findings with **boxes** from 17 radiologists; test set uses a 5-reader consensus. | Template for our own box-annotation protocol. Research-only licence, so external test only. |
| Bustos et al., *PadChest: A large chest x-ray image dataset with multi-label annotated reports*, Medical Image Analysis 2020 | >160k images, 174 findings plus 104 anatomical locations. 27% manually labelled, the rest by a neural network. | Hierarchical finding + location ontology design (our `ontology/cxr.py`). Research-only licence. |
| Rajpurkar et al., *CheXNet*, arXiv 1711.05225 (2017) | DenseNet-121 trained on CXR14; claimed radiologist-level pneumonia detection. | Historical baseline. Its comparison methodology was later criticised (next row). |
| Oakden-Rayner, *Exploring large-scale public medical image datasets*, Academic Radiology 2020 | Visual audit of CXR14 and others found substantial label errors and that "No Finding" ≠ normal. | Never use NIH labels as test ground truth for claims. Radiologist-adjudicated test sets are required. |
| Zech et al., *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*, PLOS Medicine 2018 | CNNs identified the hospital system; predicting from hospital identity alone gave an AUC of 0.861 on pooled data; external performance was worse. | Site-stratified evaluation, external validation and an OOD gate are mandatory (`08-evaluation-protocol.md`). |
| DeGrave, Janizek & Lee, *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence 2021 | Models learned dataset-source shortcuts (markers, positioning) instead of pathology. | Saliency audits plus cross-source testing. Never mix positives and negatives from different sources in one class. |
| Seyyed-Kalantari et al., *Underdiagnosis bias of AI algorithms applied to chest radiographs in under-served patient populations*, Nature Medicine 2021 | Classifiers trained on MIMIC, CheXpert and CXR14 under-diagnosed under-served groups (race, insurance, sex, age), worst for intersectional subgroups. A Matters Arising (Bernhardt et al., 2022) contested parts of the setup. | Subgroup false-negative rates are a **primary** safety metric, not an afterthought. |
| Gichoya et al., *AI recognition of patient race in medical imaging: a modelling study*, Lancet Digital Health 2022 | Deep models predicted self-reported race from CXR even when the images were degraded. | Protected attributes are encoded implicitly, so fairness must be measured, not assumed. |
| Ridnik et al., *Asymmetric Loss for Multi-Label Classification*, ICCV 2021 | ASL handles positive/negative imbalance in multi-label tasks. | Candidate loss next to masked BCE (`train/losses.py`). |
| Cohen et al., *TorchXRayVision: A library of chest X-ray datasets and models*, MIDL 2022 | Unified loaders for many CXR datasets; pretrained DenseNets. | Reuse loader **code** (Apache-2.0). Do **not** ship its MIMIC/CheXpert/"-all" weights. |

## 2. Self-supervised and foundation image encoders

| Work | Showed | → Radio |
|---|---|---|
| Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024 | Self-supervised ViTs trained on LVD-142M with strong frozen features. Code and weights Apache-2.0. | **Product backbone** with a clean licence chain. Continue SSL on CXR images we have commercial rights to. |
| Pérez-García et al., *Exploring scalable medical image encoders beyond text supervision* (RAD-DINO), Nature Machine Intelligence 2025 | Image-only DINOv2-style pretraining on ~883k CXRs matched or beat text-supervised encoders on classification, segmentation and report generation. | Supports image-only SSL as our pretraining strategy. RAD-DINO itself: MIT weights, but the model card frames them as research use and they were trained on MIMIC/CheXpert/PadChest, so **benchmark only**. |
| Tiu et al., *Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning* (CheXzero), Nature Biomedical Engineering 2022 | CLIP-style image–report training gives zero-shot CXR classification. | Contrastive image–text alignment is powerful but needs paired reports. We don't have them with commercial rights yet (Stage 5 deferred). |
| Zhang et al., *BiomedCLIP*, NEJM AI 2024 | CLIP on 15M PubMed Central figure–caption pairs. | Model card puts deployed use out of scope, so research benchmark only. |
| Xu et al., *ELIXR: Towards a general purpose X-ray AI system through alignment of LLMs and radiology vision encoders*, arXiv 2308.01317 | Google CXR Foundation embeddings aligned to LLMs. Data-efficient classification. | Under HAI-DEF terms; benchmark. Design reference for embedding-based adapters. |
| Ma et al., Ark+ (fully-open foundation model built from many public CXR datasets), Nature 2025 | Accruing heterogeneous expert labels across datasets improves encoders. | Idea of training one model across **heterogeneous label sets** with masking. Weights non-commercial. |
| Sellergren et al., *MedGemma Technical Report*, arXiv 2507.05201 (2025); MedGemma 1.5 report, arXiv 2604.05081 (2026, per model research) | Open medical VLMs (4B/27B) and the MedSigLIP encoder under the HAI-DEF terms. | Strong benchmark and possible future licensed component. HAI-DEF obligations noted in 05. |

## 3. Detection and segmentation

| Work | Showed | → Radio |
|---|---|---|
| Ren et al., *Faster R-CNN*, NeurIPS 2015 | Two-stage detector with region proposals. | Conservative Phase-2 baseline (torchvision, BSD-3). |
| Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet), ICCV 2017 | Focal loss for one-stage detection under extreme class imbalance. | Alternative baseline; focal loss suits rare findings. |
| Carion et al., *End-to-End Object Detection with Transformers* (DETR), ECCV 2020 | Set-prediction detection with no NMS. | Lineage for RT-DETR / D-FINE / RF-DETR (Apache-2.0 options). |
| Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), CVPR 2024 | Real-time DETR variant. | Candidate production detector (Apache-2.0); avoids Ultralytics AGPL. |
| Ronneberger et al., *U-Net*, MICCAI 2015 | Encoder–decoder segmentation with skip connections. | Lung, heart and device masks. |
| Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 2021 | An auto-configured U-Net matched or beat custom methods across many tasks. | Default segmentation engine (Apache-2.0). Use the 2D configuration for CXR. |
| Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications 2022 | 10-task segmentation benchmark; methods generalized across tasks. | CT/MRI expansion (Phase 7). CC-BY-SA 4.0 data. |

## 4. Vision-language models and report generation

| Work | Showed | → Radio |
|---|---|---|
| Chen et al., *Generating Radiology Reports via Memory-driven Transformer* (R2Gen), EMNLP 2020 | Encoder–decoder report generation on IU-Xray and MIMIC. | Baseline. The repo has no licence, so reference only. |
| Endo et al., *Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model* (CXR-RePaiR), ML4H 2021 (PMLR 158) | Retrieves report sentences from a corpus using CLIP-style similarity. | Retrieval as a **style source**: phrase bank for our templates, curated by radiologists. Never as a source of findings. |
| Tanida et al., *Interactive and Explainable Region-guided Radiology Report Generation* (RGRG), CVPR 2023 | Detects anatomical regions, then writes one sentence per salient region. Each sentence is tied to a box, so the report is verifiable and interactive. | **Core design inspiration.** Sentence ↔ finding ↔ box traceability in our IR. Code MIT; trained on MIMIC, so we don't reuse weights. |
| Bannur et al., *MAIRA-2: Grounded Radiology Report Generation*, arXiv 2406.04449 (2024) | Grounded report generation: each finding sentence has optional boxes. Introduced **RadFact**, an LLM-based entailment metric for factuality and grounding at the sentence level. | Evaluation design: sentence-level **logical precision/recall** and grounding precision (RadFact, MIT code). Weights research-only. |
| Chen et al., *CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation*, arXiv 2401.12208 (2024) | Instruction-tuned CXR VLM plus the CheXbench benchmark. | Benchmark only (NC licence). |
| Zambrano Chaves et al., LLaVA-Rad, Nature Communications 16:3108 (2025) | A small (7B) CXR report model trained on 697k pairs; competitive on report metrics. | Shows small specialised models can compete. Research licence; MIMIC data. |
| Tanno et al., *Collaboration between clinicians and vision–language models in radiology report generation* (Flamingo-CXR), Nature Medicine 2025 (arXiv 2311.18260) | An expert panel preferred or rated equivalent the AI reports in 77.7% of in/outpatient cases and 56.1% of ICU cases. Clinically significant errors appeared in **both** AI and human reports. Studied a draft-then-clinician-revise workflow. | Directly supports our **draft + mandatory radiologist review** model. Error taxonomy and preference-study design go into 08. |
| Huang et al., *Generative AI for Chest Radiograph Interpretation in the Emergency Department*, JAMA Network Open 2023;6(10):e2336100 | Image-to-report model for 500 ED CXRs. ED physicians rated it similar to in-house radiologists and better than teleradiology on a Likert scale. | Evidence that end-to-end drafts can be useful, but the study was retrospective at a single site with non-radiologist raters. |
| Wang et al., R2GenGPT, Meta-Radiology 2023 (venue unverified) | A frozen LLM with a visual adapter for report generation. | Possible design for an optional rephrasing LLM. BSD-3 code; Llama-2 terms apply. |
| Microsoft UniRG-CXR, arXiv 2601.17151 (2026) | Claims state of the art on ReXrank. Weights Apache-2.0. | Licence permissive, but training data research-licensed, so counsel must review before any use. |

**Synthesis for Radio.** End-to-end VLMs (Flamingo-CXR, MAIRA-2, LLaVA-Rad) produce fluent reports but can hallucinate, and their strongest weights are not commercially usable. Grounded and region-guided work (RGRG, MAIRA-2) shows the value of tying each sentence to evidence. We take that idea further by making **the IR the only source of content**: detection and classification produce findings, deterministic templates write the sentences, and an optional constrained LLM may only rephrase. A validator then proves every sentence traces back to an IR finding.

## 5. Report-quality evaluation

| Work | Showed | → Radio |
|---|---|---|
| Smit et al., *CheXbert: Combining Automatic Labelers and Expert Annotations…*, EMNLP 2020 | A BERT labeler extracts the 14 CheXpert labels from reports, giving CheXbert-F1. | Common metric. Needs a Stanford commercial licence for product use, so research reporting only. |
| Jain et al., *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports*, NeurIPS Datasets & Benchmarks 2021 | Entity/relation schema and annotated dataset. | RadGraph-F1 (MIT code; weights PhysioNet-credentialed, so internal use). |
| Yu et al., *Evaluating progress in automatic chest X-ray radiology report generation*, Patterns 2023 | Compared metrics against radiologist error counts. Introduced RadGraph F1 and the composite **RadCliQ**, which aligns better with radiologists than BLEU. | BLEU/ROUGE are never primary. RadCliQ/RadGraph go in a secondary panel. |
| Ostmeier et al., *GREEN: Generative Radiology Report Evaluation and Error Notation*, EMNLP 2024 Findings (track unverified) | An LLM judge counts clinically significant and insignificant errors with explanations. | Error-type taxonomy for our radiologist rubric. The repo has no licence, so internal only. |
| Zhao et al., *RaTEScore*, EMNLP 2024 | Entity-aware score robust to synonyms and negation. | Secondary automatic metric (MIT). |
| ReXrank leaderboard (Rajpurkar lab), PMLR v281 2025 | Public leaderboard across ReXGradient, MIMIC, IU and CheXpert Plus. | External comparison point (MIT harness), research data only. |

**For Radio:** because our sentences are generated from the IR, *finding-level* precision and recall against radiologist ground truth and the *hallucination rate* (sentences not traceable to the IR) matter most. Text metrics are secondary (see 08).

## 6. Calibration, conformal prediction and OOD detection

| Work | Showed | → Radio |
|---|---|---|
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (arXiv 1706.04599) | Modern deep nets are overconfident; one-parameter temperature scaling fixes much of it. | Per-label temperature (`calibrate/temperature.py`), fit on a site-matched calibration split. |
| Minderer et al., *Revisiting the Calibration of Modern Neural Networks*, NeurIPS 2021 (arXiv 2106.07998) | Architecture matters; ViTs are often better calibrated than CNNs. | Supports the ViT backbone; still always measure ECE. |
| Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction…*, arXiv 2107.07511 | Distribution-free coverage guarantees. | Framework for thresholds that carry a guarantee. |
| Cauchois et al., *Knowing what you know: valid and validated confidence sets in multiclass and multilabel prediction*, JMLR 2021 (arXiv 2004.10181) | Conformal sets for multi-label problems. | Per-finding sets: {present}, {absent} or {present, absent} = **uncertain**. |
| Angelopoulos et al., *Conformal Risk Control*, ICLR 2024 (arXiv 2208.02814) | Controls expected loss (e.g. false-negative rate) rather than plain coverage. | Set per-finding thresholds to bound FNR for critical findings (`calibrate/conformal.py`). |
| Angelopoulos et al., *Learn then Test*, arXiv 2110.01052 | Calibrates hyperparameters with risk guarantees. | Choosing abstention and OOD thresholds. |
| Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*, ICLR 2017 (arXiv 1610.02136) | Max-softmax baseline for OOD detection. | Baseline. |
| Lee et al., *A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks* (Mahalanobis), NeurIPS 2018 (arXiv 1807.03888) | Class-conditional Gaussian distance in feature space. | Primary OOD score on encoder embeddings (`qc/ood.py`). |
| Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020 (arXiv 2010.03759) | Energy score beats softmax for OOD. | Secondary score. |
| Ren et al., *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection*, arXiv 2106.09022 | Relative Mahalanobis improves near-OOD detection. | Use for near-OOD cases (lateral views, pediatric, unusual devices). |
| Fort et al., *Exploring the Limits of Out-of-Distribution Detection*, NeurIPS 2021 (arXiv 2106.03004) | Mahalanobis on pretrained ViT features gives strong OOD detection. | Our encoder is a ViT, so this fits directly. |
| Kumar & Palepu, arXiv 2210.15805; Hui 2025, arXiv 2505.22496 | Conformal methods applied to CXR classification (zero-shot and multi-label). | CXR-specific evidence that the approach works. |

## 7. Human–AI interaction and automation bias in radiology

| Work | Showed | → Radio |
|---|---|---|
| Seah et al., *Effect of a comprehensive deep-learning model on the accuracy of chest x-ray interpretation by radiologists: a retrospective, multireader multicase study*, Lancet Digital Health 2021;3(8):e496–e506 | 20 radiologists: macro-AUC across 127 findings rose from 0.713 unaided to 0.808 aided; no finding got worse. Industry-funded; retrospective, enriched case set. | Template for our **MRMC** design (08) and for measuring breadth of findings. |
| Yu et al., *Heterogeneity and predictors of the effects of AI assistance on radiologists*, Nature Medicine 2024;30:837–849 | 140 radiologists, 15 CXR tasks. AI help varied widely by radiologist. Experience did not predict benefit. **Inaccurate AI predictions hurt performance.** | Analyse reader studies per reader. Report the harm when AI is wrong. Consider calibrated display and abstention. |
| Dratsch et al., *Automation Bias in Mammography: The Impact of AI BI-RADS Suggestions on Reader Performance*, Radiology 2023;307(4) | Wrong AI suggestions caused large accuracy drops in readers of every experience level, most in inexperienced readers. | UI: show calibrated confidence, never pre-fill a "normal" report as accepted, require an explicit per-finding decision. Safety plan in 09. |
| Gaube et al., *Do as AI say: susceptibility in deployment of clinical decision-aids*, npj Digital Medicine 2021;4:31 | Wrong advice reduced diagnostic accuracy whether labelled as AI or human. Radiologists rated AI-labelled advice lower. | Same mitigations. Measure accuracy when advice is wrong as an explicit endpoint. |
| Plesner et al., *Autonomous Chest Radiograph Reporting Using AI: Estimation of Clinical Impact*, Radiology 2023;307(3):e222268 | Commercial tool: 28% of normal PA films could have been reported autonomously at >99% sensitivity for abnormality, ≈7.8% of all PA films. AI misses were on average more clinically severe. | "Normal" triage is high-value but high-stakes. Radio does **not** report autonomously; every report needs sign-off. |

## 8. Reporting guidelines and regulatory science

| Work | Showed | → Radio |
|---|---|---|
| Wu et al., *How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals*, Nature Medicine 2021;27:582–584 | Of 130 FDA-cleared AI devices (2015–2020), most were evaluated only retrospectively, and few used multi-site evaluation. | Evidence plan: multi-site, prospective where possible, plus post-market surveillance (09, 17). |
| Mongan et al., *Checklist for Artificial Intelligence in Medical Imaging (CLAIM)*, Radiology: AI 2020 | Reporting checklist for imaging-AI studies. | Our validation reports follow CLAIM. |
| Collins et al., *TRIPOD+AI statement*, BMJ 2024 | Reporting guideline for prediction models that use AI. | Model-development reporting. |
| Liu et al., CONSORT-AI and Cruz Rivera et al., SPIRIT-AI, Nature Medicine 2020 | Trial-reporting extensions for AI interventions. | For any prospective pilot (Phase 6). |
| Vasey et al., *DECIDE-AI*, Nature Medicine 2022 | Reporting guideline for early-stage live clinical evaluation of AI decision support. | Phase 6 controlled-pilot reporting. |
| IMDRF, *Good Machine Learning Practice for Medical Device Development: Guiding Principles* (N88, final Jan 2025) | Ten GMLP principles. | Mapped into the QMS and the training strategy (07). |

## 9. Gaps the literature leaves (= Radio IP)

1. **Evidence-grounded reporting with formal guarantees.** Prior work grounds sentences to boxes, but none we found enforces "IR-only content" with a validator and a tri-state-plus-`not_assessed` schema.
2. **Commercially clean CXR foundation weights.** The leading open encoders and VLMs all carry research-data provenance.
3. **Production OOD and quality gating for CXR**, including pediatric, lateral, wrong-body-part and post-processed images.
4. **Per-reader human–AI interaction design** that reduces harm from AI errors (Yu 2024; Dratsch 2023).

## Sources

- ChestX-ray8 (CVPR 2017): https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.1.0/
- VinDr-CXR: https://physionet.org/content/vindr-cxr/1.0.0/
- PadChest: https://bimcv.cipf.es/bimcv-projects/padchest
- Zech 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/
- Seyyed-Kalantari 2021: https://pubmed.ncbi.nlm.nih.gov/34893776/ ; Matters Arising: https://www.nature.com/articles/s41591-022-01846-8
- TorchXRayVision: https://github.com/mlmed/torchxrayvision
- DINOv2: https://github.com/facebookresearch/dinov2
- RAD-DINO: https://huggingface.co/microsoft/rad-dino
- CheXzero: https://github.com/rajpurkarlab/CheXzero
- BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- ELIXR: https://arxiv.org/abs/2308.01317
- Ark+: https://github.com/jlianglab/Ark
- MedGemma model card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
- RGRG: https://openaccess.thecvf.com/content/CVPR2023/html/Tanida_Interactive_and_Explainable_Region-Guided_Radiology_Report_Generation_CVPR_2023_paper.html ; https://github.com/ttanida/rgrg
- MAIRA-2: https://arxiv.org/abs/2406.04449 ; RadFact: https://github.com/microsoft/RadFact
- CheXagent: https://arxiv.org/abs/2401.12208
- LLaVA-Rad: https://github.com/microsoft/LLaVA-Rad
- Flamingo-CXR (Tanno 2025): https://www.nature.com/nature-index/article/10.1038/s41591-024-03302-1 ; https://arxiv.org/abs/2311.18260
- Huang 2023 JAMA Netw Open: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2810195
- CXR-RePaiR: https://github.com/rajpurkarlab/CXR-RePaiR
- R2GenGPT: https://github.com/wang-zhanyu/R2GenGPT
- UniRG-CXR: https://huggingface.co/microsoft/UniRG-CXR
- CheXbert: https://github.com/stanfordmlgroup/CheXbert
- RadGraph: https://github.com/jbdel/RadGraph
- Yu 2023 Patterns (RadCliQ): https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844
- GREEN: https://github.com/Stanford-AIMI/GREEN
- RaTEScore: https://github.com/Angelakeke/RaTEScore
- ReXrank: https://rexrank.ai
- Calibration/OOD arXiv: https://arxiv.org/abs/1706.04599 , https://arxiv.org/abs/2106.07998 , https://arxiv.org/abs/2107.07511 , https://arxiv.org/abs/2004.10181 , https://arxiv.org/abs/2208.02814 , https://arxiv.org/abs/2110.01052 , https://arxiv.org/abs/1610.02136 , https://arxiv.org/abs/1807.03888 , https://arxiv.org/abs/2010.03759 , https://arxiv.org/abs/2106.09022 , https://arxiv.org/abs/2106.03004 , https://arxiv.org/abs/2210.15805 , https://arxiv.org/abs/2505.22496
- Seah 2021 Lancet Digit Health: https://pmc.ncbi.nlm.nih.gov/articles/PMC8689166
- Yu 2024 Nat Med: https://pmc.ncbi.nlm.nih.gov/articles/PMC10957478
- Dratsch 2023 Radiology: https://research.vu.nl/en/publications/automation-bias-in-mammography-the-impact-of-artificial-intellige/
- Gaube 2021 npj Digit Med: https://pmc.ncbi.nlm.nih.gov/articles/PMC7896064
- Plesner 2023 Radiology: https://research.regionh.dk/da/publications/autonomous-chest-radiograph-reporting-using-ai-estimation-of-clin/
- Wu 2021 Nat Med: https://www.nature.com/articles/s41591-021-01312-x
- IMDRF GMLP N88: https://imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf

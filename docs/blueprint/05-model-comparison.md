# 05 — Model Architecture Comparison

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

**Method.** Licences were read from GitHub LICENSE files (via the API), Hugging Face model cards and official pages on 2026-11-25. Code licence and **weights** licence are listed separately because they often differ. Weights also inherit risk from their **training data** (see `04-dataset-licensing-matrix.md`).

We don't reproduce benchmark numbers here. Published results come from different datasets, label sets and splits, so they aren't comparable across rows. "Clinical performance" is a qualitative summary of each paper's own claims, and anything we adopt is re-measured under `08-evaluation-protocol.md`. Sizes and GPU needs are approximate. Values that couldn't be confirmed are marked *unverified*.

> Not legal advice. Licence verdicts are an engineering screen; counsel signs off before anything is promoted to a product model.

Verdicts: **SHIP** = clean for a commercial product. **COND** = usable under conditions or after legal review. **BENCH** = research or benchmark only. **AVOID** = licence trap.

---

## 1. Vision encoders and foundation models

### 1a. Capability

| Model | Type / size | Clinical performance (paper claims) | Localization | Generalization | Latency / GPU | Fine-tune complexity |
|---|---|---|---|---|---|---|
| **DINOv2** (Meta) | SSL ViT; 22M (S) – 1.1B (g) | General-domain; strong frozen features; no CXR claims | Patch features support dense heads and attention maps | Strong across domains | ViT-S/B runs on M4 Pro MPS; ViT-L/g need a cloud GPU | Low (timm/HF); continued SSL on CXR needs multi-GPU |
| **DINOv3** (Meta) | SSL ViT up to 7B | General-domain | Dense features | Strong | Large variants need a cloud GPU | Medium |
| **RAD-DINO** (Microsoft) | DINOv2-style ViT-B, 86.6M, CXR-pretrained (~883k images) | Matched or beat text-supervised encoders on CXR classification, segmentation and report generation (Nature Machine Intelligence 2025) | Good dense features | Multi-source pretraining | MPS-friendly | Low |
| **TorchXRayVision** DenseNet-121s | CNN, ~8M; per-dataset weights | Solid baselines on public CXR benchmarks | CAM only | Per-dataset weights are site-specific | Laptop-friendly | Low |
| **CheXzero** | CLIP (image–report) | Zero-shot CXR classification near radiologist level on some labels (paper) | None | MIMIC-trained | CLIP-size | Medium |
| **BiomedCLIP** | ViT-B/16 + PubMedBERT | Broad biomedical retrieval/classification | None | PMC figures ≠ clinical DICOM | ViT-B | Low |
| **MedSigLIP / CXR Foundation (ELIXR)** (Google) | MedSigLIP 0.9B (400M vision); ELIXR on EfficientNet-L2 | Data-efficient CXR classification (papers) | Limited | Includes non-US (India) private data | Medium GPU | Low (embeddings) |
| **MedImageInsight** (Microsoft) | Size not stated | Multi-modality embeddings | — | Proprietary data mix | — | Low (API) |
| **Ark+** (ASU) | Swin-B/L | Strong across many public CXR sets (Nature 2025) | — | Many datasets | Medium | Medium |
| **CheXOne** (Stanford, 2026) | 4B on a Qwen2.5-VL-3B base | Multi-task CXR | — | 30 public datasets | Cloud GPU | High |
| **ConvNeXt V2 / Swin** | CNN / hierarchical ViT | General | Feature pyramids suit detection | — | Small–medium | Low |
| **MedSAM2** | SAM2.1-based promptable segmenter | CT/MRI/US/video, not CXR | Promptable masks | — | Medium | Medium |

### 1b. Licence and provenance

| Model | Code licence | Weights licence | Training-data provenance | Verdict |
|---|---|---|---|---|
| DINOv2 | Apache-2.0 | Apache-2.0 | LVD-142M web images (curated) | **SHIP** (backbone init) |
| DINOv3 | DINOv3 License (custom) | Same | LVD-1689M web images | **COND** (custom terms: licence pass-through on redistribution, trade controls, no military/ITAR use) |
| RAD-DINO | MIT | MIT, but the card frames intended use as research | MIMIC, CheXpert, PadChest, NIH, BRAX | **BENCH** (data taint; counsel may revisit) |
| TorchXRayVision | Apache-2.0 (library) | Per-model, not stated | Per weight: NIH, RSNA, CheXpert, MIMIC, PadChest, "-all" | Code **SHIP**. NIH/RSNA-only weights **COND**. MIMIC/CheXpert/"-all" weights **BENCH** |
| CheXzero | MIT | Not stated | MIMIC-CXR | **BENCH** |
| BiomedCLIP | MIT | MIT, but card: deployed use, "commercial or otherwise … out of scope" | PMC-15M | **BENCH** |
| MedCLIP | **No LICENSE** | Not stated | Paper: CheXpert/MIMIC (unverified) | **AVOID** |
| MedSigLIP / CXR Foundation | Apache-2.0 | **HAI-DEF terms** | Includes MIMIC-CXR plus licensed/internal data | **COND** (see §6) |
| MedImageInsight | — | Foundry card says MIT; a June 2026 MS blog calls open versions research/academic only; paid "Premium" closed version | Proprietary | **COND** (get written terms from Microsoft) |
| Ark+ | ASU non-commercial licence; patent pending | Request form | CheXpert, NIH, VinDr, Shenzhen, MIMIC, others | **BENCH** |
| CheXOne | Apache-2.0 (framework) | **CC-BY-NC-4.0** | 30 public datasets | **BENCH** |
| ConvNeXt V2 | MIT | **CC-BY-NC** (ImageNet weights) | ImageNet | Architecture **SHIP**, weights **AVOID** |
| Swin | MIT | ImageNet weights; separate licence unverified | ImageNet | Architecture **SHIP**, weights **COND** |
| timm pretrained weights | Apache-2.0 | README: "assume … the original dataset license applies" | Mostly ImageNet | **COND** per weight |
| MedSAM2 | Apache-2.0 | CC-BY-SA-4.0 + "research and education" | CT/MRI/US | **BENCH** |

**Choice: DINOv2 ViT-S/14 (Phase 1, Mac) → ViT-B/14 or ViT-L/14 (cloud), with continued CXR self-supervised pretraining on OK-licensed images.** RAD-DINO and MedSigLIP serve as **benchmarks to beat**. Implemented in `src/radio/models/encoder.py`.

---

## 2. Detection and segmentation

| Tool | Approach | Localization output | Maturity | MPS / GPU | Code licence | Weights | Verdict |
|---|---|---|---|---|---|---|---|
| **torchvision Faster R-CNN / RetinaNet** | Two-stage / one-stage with focal loss | Boxes | Stable, well-known | Runs on MPS (op coverage varies) | BSD-3 | COCO/ImageNet init (caveat) | **SHIP** (Phase-2 baseline) |
| **RT-DETR** (lyuwenyu) | Real-time DETR | Boxes | Active | Cloud GPU recommended | Apache-2.0 | Apache-2.0 | **SHIP** |
| **D-FINE** | DETR refinement | Boxes | Active | Cloud GPU | Apache-2.0 | — | **SHIP** |
| **RF-DETR** (Roboflow) | DINOv2-backbone DETR | Boxes (+seg variants) | Active | Cloud GPU | Apache-2.0 for Nano–Large and Seg sizes; **PML 1.0** for Atto/Femto/Pico/XL/2XL | DINOv2 backbone | **SHIP** (Apache sizes only) |
| DETR | Set prediction | Boxes | Archived | — | Apache-2.0 | — | Reference |
| DINO-DETR (IDEA) | DETR + denoising | Boxes | Stale since 2024 | — | Apache-2.0 | — | Reference |
| MMDetection | Toolbox | Boxes / masks | Last release Jan 2024; push Aug 2024 | CUDA-centric | Apache-2.0 | — | Usable, stale |
| Detectron2 | Toolbox | Boxes / masks | Commits still land; last tag v0.6 (2021) | CUDA-centric | Apache-2.0 | — | Usable |
| **Ultralytics YOLO** | One-stage | Boxes / masks | Very active | Easy | **AGPL-3.0** (Ultralytics says trained models are AGPL too; SaaS triggers it) | AGPL | **AVOID** unless Enterprise licence |
| **nnU-Net** | Self-configuring U-Net | Masks | Mature (Nature Methods 2021) | 2D OK on MPS; 3D needs CUDA (docs) | Apache-2.0 | — | **SHIP** (lung/heart/device/pneumothorax masks) |
| **MONAI** | Toolkit (nets, transforms, metrics) | Masks / boxes / heatmaps | Mature | Plain PyTorch, so 2D usually works on MPS (official support unverified) | Apache-2.0 | — | **SHIP** |
| segmentation_models_pytorch | U-Net/FPN zoo | Masks | Active | MPS OK | MIT | Encoders pull timm/ImageNet weights | **COND** per encoder weight |

**Localization strategy (hybrid).** The brief asks whether to use detection, segmentation, classification, or a hybrid:

| Finding type | Approach | Why |
|---|---|---|
| Focal findings (nodule, mass, focal opacity/consolidation, fracture) | **Box detector** | These are discrete instances; boxes map directly to DICOM SR SCOORD and to the review UI |
| Extent-defined findings (pneumothorax, effusion, cardiomegaly via cardiothoracic ratio, device tip position) | **Segmentation** (nnU-Net 2D) | Supports measurements and laterality |
| Every classifier finding | **Patch-attention / Grad-CAM-style heatmaps** | A weak localization channel, shown only as "supporting evidence" and never as a measurement |
| CT/MRI (Phase 7) | 3D localization through the same `Analyzer` interface | — |

---

## 3. Medical vision-language and report-generation models

| Model | Size | Approach | Grounding | Clinical evidence (paper claims) | GPU | Code / weights licence | Training data | Verdict |
|---|---|---|---|---|---|---|---|---|
| **MAIRA-2** (Microsoft) | 7B (Vicuna-7B + RAD-DINO) | Grounded report generation | **Yes** (per-sentence boxes) | Introduced RadFact; strong grounded reporting | ≥1 large GPU | MSRLA, research only | MIMIC, PadChest, private USMix | **BENCH** (design reference) |
| **CheXagent** (Stanford) | 3B | Instruction-tuned CXR VLM | Limited | CheXbench | 1 GPU | Conflict: HF says MIT, repo says **CC-BY-NC-ND 4.0** | CheXinstruct (includes MIMIC; unverified) | **BENCH** |
| **LLaVA-Rad** (Microsoft) | 7B | Small specialised CXR VLM | No | Competitive report metrics (Nature Communications 2025) | 1 V100 for inference | MSR licence, non-commercial | 697k MIMIC pairs | **BENCH** |
| **MedGemma 4B / 27B, 1.5** (Google) | 4B, 27B | Gemma-3-based medical VLM | No native boxes | CXR report generation plus broad medical QA (tech report) | 4B on Mac (quantized, inference); 27B in cloud | Code Apache-2.0; weights **HAI-DEF** | Includes MIMIC-CXR, Chest ImaGenome | **COND** (§6) |
| **UniRG-CXR** (Microsoft, 2026) | 9B (Qwen3-VL-8B base) | Report generation | — | Claims SOTA on ReXrank | Cloud | **Apache-2.0** weights | MIMIC, CheXpert Plus, ReXGradient, IU-Xray | **COND** (permissive licence over research data; counsel) |
| **CXRReportGen Premium** (Microsoft) | — | Closed, commercial, private preview | — | — | API | Paid commercial | — | **COND** (buy-vs-build option) |
| **NV-Reason-CXR** (NVIDIA) | 3B | Reasoning VLM | — | — | 1 GPU | Code Apache-2.0; weights **NVIDIA OneWay Noncommercial** | — | **AVOID** |
| **RGRG** | Detector + GPT-2 | Region-guided, one sentence per region | **Yes** | CVPR 2023 | 1 GPU | MIT | MIMIC / Chest ImaGenome | Design reference (**BENCH** weights) |
| **R2Gen** | Memory-driven transformer | Encoder–decoder | No | EMNLP 2020 | 1 GPU | **No licence** | MIMIC, IU | **AVOID** code |
| **R2GenGPT** | Swin + frozen Llama-2-7B | Adapter | No | — | 1 GPU | BSD-3 code; Llama-2 terms | MIMIC, IU | **BENCH** |
| **CXR-RePaiR** | CLIP retrieval | Retrieval | No | ML4H 2021 | Small | MIT | MIMIC | Design reference |
| **RadFM** | ~14B (unverified) | Generalist 2D/3D | — | — | Large | Code MIT; weights unclear | PMC-heavy MedMD | **BENCH** |
| **MedRAX** (agent) | Orchestrator | LLM calls CXR tools | Via tools | ICML 2025 demo | — | Apache-2.0 code; pulls NC weights | — | Design reference only |

**Why Radio doesn't ship an end-to-end VLM in Phase 1–4:**

1. Every strong CXR VLM is BENCH or COND on licence.
2. End-to-end generation can state findings the image doesn't support. Flamingo-CXR (Tanno et al., Nature Medicine 2025) found clinically significant errors in AI drafts.
3. FDA has no predicate for image-based report drafting, so a De Novo is likely. A hierarchical design whose report content is produced deterministically from validated findings is easier to verify and to describe in a submission.

Our report engine (`src/radio/report/`) is **IR → deterministic templates**. An optional small LLM may only rephrase, under a grammar constraint, and is checked by the validator.

---

## 4. Text LLMs for constrained rephrasing (optional, Phase 3+)

| Family | Licence (per model research; re-verify at adoption) | Commercial | Fit |
|---|---|---|---|
| Qwen3.5 (0.8B–397B) | Apache-2.0 | Yes | Small sizes run locally; good instruction following |
| Gemma 4 (E2B, E4B, 26B MoE, 31B; Apr 2026) | Apache-2.0 (first Gemma under Apache, per Google OSS blog) | Yes | Small sizes suit on-prem |
| Gemma 3 and earlier | Gemma Terms + Prohibited Use Policy | Conditional | Prefer Gemma 4 |
| Llama 3.x / 4 | Llama Community Licence (700M MAU clause, AUP) — not re-read | Conditional | Avoid unless needed |
| Mistral | Mixed (some Apache-2.0, some research) — check per model | Per model | Per model |

**Constrained decoding:** outlines (Apache-2.0), xgrammar (Apache-2.0; default in vLLM/SGLang), guidance (MIT). The LLM receives only the validated IR (JSON). The output grammar allows only sentence slots keyed to IR `finding_id`s. `report/validator.py` re-parses the output and rejects any sentence whose finding, laterality, location or status isn't in the IR.

---

## 5. Report-quality metrics (for 08)

| Metric | Measures | Licence | Use |
|---|---|---|---|
| Finding-level P/R/F1 vs. radiologist IR | Clinical correctness | Ours | **Primary** |
| Hallucination rate (sentences not traceable to IR / to ground truth) | Safety | Ours | **Primary** |
| RadFact | Sentence-level logical precision/recall and grounding | MIT (archived repo) | Secondary |
| RadGraph F1 / RadCliQ | Entity-relation overlap / composite | MIT code; RadGraph weights PhysioNet-credentialed | Secondary (internal) |
| RaTEScore | Entity-aware, negation-robust | MIT | Secondary |
| CheXbert F1 | 14-label agreement | Stanford commercial licence required | Research reporting only |
| GREEN | LLM-judged clinically significant errors | No repo licence | Internal research only |
| BLEU / ROUGE / METEOR | n-gram overlap | — | Reported, **never primary** |

---

## 6. Conditional licences: what "COND" requires

| Item | Condition to clear |
|---|---|
| **HAI-DEF** (MedGemma, MedSigLIP, CXR Foundation) | Commercial use isn't explicitly banned. Requirements: obtain regulatory clearance "when applicable" before distribution (§3.1); pass restrictions and notices to downstream users (hosted API counts as distribution); don't make Google a device "manufacturer" (§3.2), or Google may terminate; distilled/synthetic-trained models are derivatives; we indemnify Google. → Counsel review plus written confirmation from Google before any product use. |
| **DINOv3** custom licence | Pass the licence through on redistribution; trade-control and military-use exclusions. → Counsel. |
| **MedImageInsight** | Conflicting public statements. → Written terms from Microsoft. |
| **UniRG-CXR** | Apache-2.0 weights trained on research-licensed data. → Counsel opinion on data-taint risk. |
| **CC-BY-SA data** (MSD) | Whether share-alike reaches trained weights. → Counsel. |

## 7. Licence gate (enforced in the model registry)

Every checkpoint in MLflow carries these tags:
- `code_licences[]`
- `weights_init` (source and licence)
- `datasets[]` (each with its licence verdict from 04)
- `research_only` (boolean)

Rules:
- `research_only = true` if **any** initialisation or dataset is BENCH, AVOID or unresolved COND.
- Only `research_only = false` checkpoints can be promoted to the `product` stage.
- `src/radio/train/train_classifier.py` writes these tags. Promotion tooling (Phase 5) enforces the rule.

## 8. Shortlist used by `06-recommended-architecture.md`

| Role | Phase 1 (Mac) | Phase 2–4 (cloud) | Benchmarks to beat (BENCH/COND, not shipped) |
|---|---|---|---|
| Image encoder | DINOv2 ViT-S/14 (Apache-2.0), fine-tuned on NIH CXR14 subset | DINOv2 ViT-B/L + continued CXR SSL on OK data (NIH, MIDRC, partners) | RAD-DINO, MedSigLIP, TorchXRayVision-all |
| Multi-label head | Linear/MLP head, masked BCE / ASL | Same + label-hierarchy constraints | CheXpert-style models |
| Detector | Interface only (`models/detector.py`) | torchvision Faster R-CNN baseline → RT-DETR / RF-DETR (Apache sizes) on ChestX-Det + partner boxes | VinDr-CXR-trained baselines (research) |
| Segmenter | — | nnU-Net 2D (lungs, heart, pneumothorax, effusion, devices) | — |
| Heatmaps | Patch-attention / Grad-CAM from encoder | Same, plus pointing-game evaluation | — |
| Calibration / OOD | Per-label temperature; conformal thresholds; Mahalanobis on embeddings | + energy score, relative Mahalanobis, conformal risk control | — |
| Report text | Deterministic templates from IR | + optional Qwen3.5 / Gemma-4 small (Apache-2.0) rephraser under xgrammar/outlines | MAIRA-2, MedGemma, CheXagent, LLaVA-Rad (comparison only) |
| Report eval | Finding-level P/R, hallucination, contradiction | + RadFact, RaTEScore, RadGraph (internal), radiologist panel | ReXrank harness |

## Sources

- DINOv2: https://github.com/facebookresearch/dinov2 ; DINOv3: https://github.com/facebookresearch/dinov3
- RAD-DINO: https://huggingface.co/microsoft/rad-dino ; rad-dino-maira-2: https://huggingface.co/microsoft/rad-dino-maira-2
- TorchXRayVision: https://github.com/mlmed/torchxrayvision
- CheXzero: https://github.com/rajpurkarlab/CheXzero
- BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- MedCLIP: https://github.com/RyanWangZf/MedCLIP
- CXR Foundation: https://huggingface.co/google/cxr-foundation ; MedSigLIP: https://huggingface.co/google/medsiglip-448
- MedImageInsight: https://ai.azure.com/catalog/models/MedImageInsight ; MS blog: https://techcommunity.microsoft.com/blog/healthcareandlifesciencesblog/a-new-chapter-of-efficient-foundation-models-for-medical-imaging/4526964
- ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2 ; Swin: https://github.com/microsoft/Swin-Transformer ; timm: https://github.com/huggingface/pytorch-image-models
- MedSAM2: https://github.com/bowang-lab/MedSAM2
- Ark+: https://github.com/jlianglab/Ark ; CheXOne: https://huggingface.co/StanfordAIMI/CheXOne
- CheXagent: https://huggingface.co/StanfordAIMI/CheXagent-2-3b , https://github.com/Stanford-AIMI/CheXagent
- MAIRA-2: https://huggingface.co/microsoft/maira-2
- LLaVA-Rad: https://github.com/microsoft/LLaVA-Rad
- MedGemma model card: https://developers.google.com/health-ai-developer-foundations/medgemma/model-card ; HAI-DEF terms: https://developers.google.com/health-ai-developer-foundations/terms
- UniRG-CXR: https://huggingface.co/microsoft/UniRG-CXR
- NV-Reason-CXR: https://github.com/NVIDIA-Medtech/NV-Reason-CXR
- RGRG: https://github.com/ttanida/rgrg ; R2Gen: https://github.com/cuhksz-nlp/R2Gen ; R2GenGPT: https://github.com/wang-zhanyu/R2GenGPT ; CXR-RePaiR: https://github.com/rajpurkarlab/CXR-RePaiR ; RadFM: https://github.com/chaoyi-wu/RadFM ; MedRAX: https://github.com/bowang-lab/MedRAX
- torchvision: https://github.com/pytorch/vision ; RT-DETR: https://github.com/lyuwenyu/RT-DETR ; Ultralytics licence: https://www.ultralytics.com/license
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet ; MONAI: https://github.com/Project-MONAI/MONAI ; smp: https://github.com/qubvel-org/segmentation_models.pytorch
- MMDetection: https://github.com/open-mmlab/mmdetection ; Detectron2: https://github.com/facebookresearch/detectron2
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-2B ; Gemma 4 Apache-2.0: https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html
- outlines: https://github.com/dottxt-ai/outlines ; xgrammar: https://github.com/mlc-ai/xgrammar ; guidance: https://github.com/guidance-ai/guidance
- RadFact: https://github.com/microsoft/RadFact ; RadGraph: https://github.com/jbdel/RadGraph ; RaTEScore: https://github.com/Angelakeke/RaTEScore ; CheXbert: https://github.com/stanfordmlgroup/CheXbert ; GREEN: https://github.com/Stanford-AIMI/GREEN ; ReXrank: https://rexrank.ai
- Flamingo-CXR (Tanno 2025): https://arxiv.org/abs/2311.18260

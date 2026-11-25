# 14 — GPU and Infrastructure Cost Estimates

Status: draft v0.1 — 2026-11-25 · Part of the Radio R&D blueprint (see 00-index.md)

**All figures are estimates.** Prices are list prices checked on **2026-11-25** from the cited pages. They change often and exclude tax, discounts and reserved/committed pricing unless stated. GPU-hour figures are derived from stated FLOP and throughput assumptions and should be replaced with measured numbers from the first cloud runs. Radiologist rates are **not sourced** and appear only as an explicit parameter `R` (USD/hour) for the user to replace with real quotes.

## 1. Reference prices (checked 2026-11-25)

| Provider | Instance / GPU | Price | Source |
|---|---|---|---|
| Lambda (on-demand) | 1× H100 SXM 80 GB | $4.89 /GPU-h | lambda.ai/pricing (fetched) |
| Lambda (on-demand) | 8× H100 SXM 80 GB node | $4.59 /GPU-h ($36.72/node-h) | lambda.ai/pricing (fetched) |
| Lambda (on-demand) | 1× H100 PCIe 80 GB | $3.79 /GPU-h | lambda.ai/pricing (fetched) |
| Lambda (on-demand) | 1× A100 SXM/PCIe 40 GB | $1.99 /GPU-h | lambda.ai/pricing (fetched) |
| Lambda (on-demand) | 8× A100 80 GB node | $3.19 /GPU-h | lambda.ai/pricing (fetched) |
| Lambda (1-Click Cluster, 2 wk–1 yr) | H100, 16 GPUs | $6.16 /GPU-h | lambda.ai/pricing (fetched) |
| RunPod (secure cloud) | H100 SXM 80 GB | $4.39 /h (another listing on the same page shows $4.79) | runpod.io/pricing (fetched) |
| RunPod (secure cloud) | H100 PCIe 80 GB | $2.89 /h | runpod.io/pricing (fetched) |
| RunPod (secure cloud) | A100 SXM 80 GB / A100 PCIe 80 GB | $2.19 /h / $2.09 /h | runpod.io/pricing (fetched) |
| AWS (on-demand, us-east-1) | p5.48xlarge (8× H100) | ≈ $55.04 /instance-h ≈ $6.88 /GPU-h | Third-party trackers (Vantage, Holori); **not verified against the AWS price list** |
| AWS (Capacity Blocks, us-east-1) | p5.48xlarge | $34.608 /instance-h effective (dynamic) | aws.amazon.com/ec2/capacityblocks/pricing (search result; not re-fetched) |

**Planning rate used below.** H100 at **$4.60/GPU-h** (Lambda 8× node). Sensitivity: ×0.65 (RunPod PCIe-class) to ×1.5 (hyperscaler on-demand).

**Compliance note.** Training on **de-identified** data in a research account does not require a HIPAA BAA with the GPU provider. Any PHI-bearing workload (clinical inference in Radio SaaS, partner data before de-identification) must run with a provider and service that will sign a BAA/DPA and meet the data-residency requirements. Hyperscalers typically offer this. Whether the neo-clouds listed above sign BAAs is **unverified**.

## 2. Throughput assumptions

| Model | Input | Forward FLOPs/image (approx.) | Training FLOPs/image (~3× fwd) |
|---|---|---|---|
| ViT-S/14 (22M) | 448 px (1,024 tokens) | ≈ 0.05 TFLOP | ≈ 0.15 TFLOP |
| ViT-B/14 (86M) | 518 px (1,369 tokens) | ≈ 0.30 TFLOP (2·params·tokens + attention) | ≈ 0.9 TFLOP |
| ViT-L/14 (304M) | 518 px | ≈ 1.0 TFLOP | ≈ 3.0 TFLOP |

- **Effective H100 bf16 throughput:** 30–35% MFU of the ~990 TFLOPS dense peak, so ≈ 300–350 TFLOPS. Real CXR pipelines are often limited by data loading (16-bit PNG decode, augmentation), so **every estimate below includes a ×2–3 inefficiency factor**.
- **Experimentation multiplier:** a "final" run is typically preceded by ablations, sweeps and failed runs. Budgets apply **×5** to final-run compute for R&D phases (assumption).

## 3. Phase-by-phase compute estimates

| Phase / workload | Assumptions | Final-run GPU-h (H100) | With ×5 R&D multiplier | Est. cost @ $4.60 |
|---|---|---|---|---|
| **P1** classifier smoke tests | Mac M4 Pro (MPS), NIH subset 2k–10k images, ViT-S/14 | 0 cloud | 0 | $0 (local) |
| **P1** full NIH CXR14 fine-tune | 112,120 images × 20 epochs, ViT-B/14 @ 518 px | ≈ 2 (FLOP-bound) → 5–10 with I/O factor | 25–50 | **$115–230** |
| **P1/P2** CXR self-supervised pretraining (ViT-B) | ≈ 300k licensed images × 100 epochs, DINOv2 multi-crop (≈ 2.7 TFLOP/img/step incl. teacher) | ≈ 65 → 130–200 | 650–1,000 | **$3.0k–4.6k** |
| Same, ViT-L | ≈ 3.5× ViT-B | 450–700 | 2,250–3,500 | **$10k–16k** |
| **P2** detector + nnU-Net 2D anatomy | ≈ 10–20k boxed/masked images, 50–100 epochs @ 1024 px | 20–100 | 100–500 | **$0.5k–2.3k** |
| **P3** image–report alignment (if partner pairs exist) | 200k pairs × 30 epochs @ 448 px, ViT-B + BERT-base | 50–150 | 250–750 | **$1.2k–3.5k** |
| **P3** LoRA rephraser (3–9B Apache LLM) | ≈ 50k pairs × 3 epochs | 10–40 | 50–200 | **$0.2k–0.9k** |
| **P4** evaluation runs | Bootstrap CIs are CPU; GPU for inference on ≈ 50k external images × many model versions | 5–20 | 25–100 | **$0.1k–0.5k** |
| **Total R&D compute, P1–P4** | ViT-B path | — | ≈ 1,100–2,600 | **≈ $5k–12k** |
| **Total R&D compute, P1–P4** | ViT-L path | — | ≈ 2,700–5,100 | **≈ $12k–24k** |

Compute is **not** the dominant cost of this programme. Data licensing, radiologist annotation, the reader study and regulatory work dominate (§5–7).

## 4. Inference / serving (production, estimate)

| Deployment | Hardware assumption | Capacity (estimate) | Cost driver |
|---|---|---|---|
| On-prem appliance | 1 server with 1 inference-class GPU (L4/L40S/A10-class) | A CXR pipeline (ViT-B + detector + segmenter) at a few hundred ms to ~2 s per study. One GPU covers well over a typical department's daily CXR volume; measure in Phase 5 | Hardware capex (quote-based; unsourced) + support |
| Radio SaaS per region | 2+ GPU nodes for HA, autoscaling | Same per-GPU capacity | ≈ GPU-h × utilisation; at $1.99–4.60/GPU-h list (A100–H100) a dedicated always-on GPU costs ≈ $1.5k–3.4k/month per GPU before discounts (730 h/month × rate) |

Smaller or older GPUs are likely sufficient for inference. Price them at deployment time; they are not checked here.

## 5. Storage and transfer

| Item | Volume (estimate) | Price | Note |
|---|---|---|---|
| NIH CXR14 (PNG) | ≈ 42 GB raw download (widely cited; not re-measured) | — | **Exceeds this Mac's ~41 GB free disk.** Phase 1 uses 1–2 batches (≤ 10 GB) locally; the full set lives in cloud object storage |
| Preprocessed 16-bit PNG cache @ 1024 px | ≈ 1–2 MB/image → 112k images ≈ 110–220 GB | Object storage list price ≈ $0.02–0.03/GB-month class (hyperscaler standard tier; **not re-checked this session**) | Lifecycle to infrequent-access after Phase 2 |
| Partner DICOM (original) | ≈ 5–30 MB/image depending on detector and compression → 300k images ≈ 1.5–9 TB | Same | Keep originals in the partner-side store where possible |
| Egress | Mostly intra-region | Hyperscaler egress fees apply cross-region/out-to-internet (not re-checked) | Keep training compute in the same region as data |
| MLflow / artefacts | < 1 TB | Same | — |

**Local Mac recommendation.** Add an external SSD (≥ 1 TB) before downloading more than ~10 GB of data. The internal disk is 96% full (measured 2026-11-25).

## 6. Annotation and reader-study costs (parametric)

`R` = radiologist cost per hour (USD). **Not sourced — replace with real quotes per market.** Rates differ substantially between the US, EU and India.

| Activity | Cases | Readers | Minutes/case (assumption) | Radiologist-hours | Cost |
|---|---|---|---|---|---|
| Label-noise audit | 14 labels × 200 = 2,800 | 1 | 0.5 | ≈ 23 | 23·R |
| Validation gold set | 1,000 per source × 2 sources | 2 | 2 | ≈ 133 | 133·R |
| Calibration set | 500 per site × 3 sites | 2 | 2 | ≈ 100 | 100·R |
| External test sets | 1,500 per site × 3 sites | 3 (+ adjudication ~15%) | 2 | ≈ 450 + 70 | 520·R |
| Box/mask annotation (P2) | 10,000 images | 2 (+ third reader on ~20%) | 3 | ≈ 1,100 | 1,100·R |
| Template authoring + review | — | 2–3 radiologists | — | ≈ 80–150 | 80–150·R |
| Gold QA set (report errors) | 300 | 2 | 6 | 60 | 60·R |
| MRMC reader study | 400 cases × 2 sessions | 15 readers | 2.5 | ≈ 500 | 500·R + per-reader honoraria/CRO fees (unsourced) |
| **Total** | | | | **≈ 2,500–2,600 h** | **≈ 2,550·R** |

Example (illustrative only, not a quote): at R = $150/h the total is ≈ $380k; at R = $50/h it is ≈ $130k. Annotation-platform and project-management overhead (often run through a vendor) is extra and unsourced.

## 7. Other material cost lines (sourced where possible)

| Item | Cost | Source / status |
|---|---|---|
| FDA 510(k) user fee FY2027 | $28,653 (small business $7,163) | Federal Register 30 Jul 2026 |
| FDA De Novo user fee FY2027 | $191,020 (small business $47,755) | Federal Register 30 Jul 2026 |
| EU Notified Body (MDR IIb + AI Act) | Quote-based | Unsourced |
| CDSCO licence fees (MD-14/15 import, Class C) | Per MDR 2017 fee schedule | Not checked this session |
| Stanford AIMI commercial dataset licence (if needed) | $70,000 per dataset per year (FY25 price) | aimi.stanford.edu/datasets-commercial-use |
| MIDRC commercial data agreement | Data access free (agreement required) | midrc.org/faqs |
| ISO 13485 QMS certification, IEC 62304 tooling, pen-testing, legal | Quote-based | Unsourced |

## 8. Phase budget envelope (compute + storage only, estimate)

| Phase | Compute + storage (ViT-B path) |
|---|---|
| 1 Research prototype | $0 local + ≤ $0.5k cloud |
| 2 Localisation | $4k–8k (incl. SSL pretraining) |
| 3 Reporting | $1.5k–4.5k |
| 4 Clinical validation | $0.5k–2k |
| 5–6 Integration + pilot (per site, per year, SaaS) | ≈ $20k–45k (1–1.5 always-on GPU-equivalents + storage; at list rates) |
| 7 Expansion (CT/MRI 3D training) | Higher: 3D segmentation and detection typically need 10× or more of the 2D GPU-hours (assumption); re-estimate when scoped |

## Sources

- Lambda pricing (fetched 2026-11-25): https://lambda.ai/pricing
- RunPod pricing (fetched 2026-11-25): https://www.runpod.io/pricing
- AWS p5.48xlarge on-demand (third-party trackers, not AWS): https://instances.vantage.sh/aws/ec2/p5.48xlarge · https://calculator.holori.com/aws/ec2/p5.48xlarge
- AWS EC2 Capacity Blocks pricing: https://aws.amazon.com/ec2/capacityblocks/pricing
- FDA FY2027 user fees: https://www.federalregister.gov/documents/2026/07/30/2026-15335/medical-device-user-fee-rates-for-fiscal-year-2027 · https://www.raps.org/resource/fda-unveils-fy-2027-user-fee-rates.html
- Stanford AIMI commercial licensing: https://aimi.stanford.edu/datasets-commercial-use
- MIDRC FAQ: https://www.midrc.org/faqs
- DINOv2 training recipe (for SSL compute structure): https://arxiv.org/abs/2304.07193
- H100 peak bf16 dense throughput (~990 TFLOPS) is from NVIDIA's public H100 SXM spec sheet; not re-checked this session.
- Local machine facts (M4 Pro, 24 GB, ~41 GB free disk) were measured on this Mac on 2026-11-25.

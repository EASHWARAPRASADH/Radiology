# Data notices

Datasets used to train or calibrate any Radio model, with their required attribution. Policy:
`docs/blueprint/04-dataset-licensing-matrix.md` §3; machine-readable registry: `src/radio/data/registry.py`.
Raw images are never committed or redistributed (`data/` is git-ignored).

| Dataset | Used for | Licence verdict | Required attribution |
|---|---|---|---|
| ChestX-Det (Deepwise AI Lab) | Phase-2 research detector (train / operating points / test) | OK (annotations Apache-2.0; images are NIH ChestX-ray14) | Annotations © Deepwise AI Lab, Apache License 2.0, https://github.com/Deepwise-AILab/ChestX-Det-Dataset. Lian J. et al., *A Structure-Aware Relation Network for Thoracic Diseases Detection and Segmentation*, IEEE TMI 2021. Images: NIH attribution as below. |
| NIH ChestX-ray14 | Phase-1 research classifier (train / calibrate / test) | OK (terms checked 2026-11-25) | Data provided by the NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC. Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* IEEE CVPR 2017. |

Pretrained weights: torchvision ResNet-50 ImageNet backbone (BSD-3 code; ImageNet-derived weights, registry verdict CONDITIONAL, so the Phase-2 detector is tagged `research_only=true`). DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, Meta AI, Apache-2.0), Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.

No model trained in this repository has been cleared, approved or validated for clinical use.

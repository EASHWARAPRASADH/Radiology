"""Data and weight-initialisation registry with licence verdicts (mirrors
docs/blueprint/04-dataset-licensing-matrix.md and 05-model-comparison.md section 7).

Rules enforced:
* A run with ``purpose: product`` is refused unless every dataset verdict is OK and the weight
  initialisation is OK.
* Any run touching a non-OK dataset or initialisation is tagged ``research_only=true`` (weight taint).
Verdicts are a snapshot of terms as read on ``terms_checked``; re-check before any product build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class Verdict(StrEnum):
    ok = "OK"
    conditional = "CONDITIONAL"
    research_only = "RESEARCH-ONLY"
    unverified = "UNVERIFIED"
    unavailable = "UNAVAILABLE"


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str
    name: str
    verdict: Verdict
    source_url: str
    terms_checked: str
    attribution: str
    redistribution_allowed: bool


@dataclass(frozen=True)
class WeightsEntry:
    weights_id: str
    verdict: Verdict
    licence: str
    trained_on: str
    source_url: str


_D = DatasetEntry
DATASETS: dict[str, DatasetEntry] = {d.dataset_id: d for d in [
    _D("nih_cxr14", "NIH ChestX-ray14", Verdict.ok, "https://nihcc.app.box.com/v/ChestXray-NIHCC", "2026-11-25",
       "Provide a link to the NIH download site, acknowledge the NIH Clinical Center as data provider, and cite "
       "Wang et al., ChestX-ray8, CVPR 2017.", False),
    _D("chestx_det", "ChestX-Det", Verdict.ok, "https://github.com/Deepwise-AILab/ChestX-Det-Dataset", "2026-11-25",
       "Apache-2.0 notice for annotations; NIH attribution for images.", False),
    _D("siim_acr_ptx", "SIIM-ACR Pneumothorax", Verdict.ok,
       "https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/rules", "2026-11-25",
       "Per competition rules; no redistribution to non-participants.", False),
    _D("rsna_pneumonia_2018", "RSNA Pneumonia Detection 2018", Verdict.conditional,
       "https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018", "2026-11-25",
       "RSNA attribution per terms of use; written confirmation of commercial use pending.", False),
    _D("midrc", "MIDRC", Verdict.conditional, "https://www.midrc.org/faqs", "2026-11-25",
       "Per MIDRC commercial DUA (to be signed).", False),
    _D("msd", "Medical Segmentation Decathlon", Verdict.conditional, "http://medicaldecathlon.com/", "2026-11-25",
       "CC-BY-SA 4.0; counsel to confirm share-alike scope for weights.", True),
    _D("brats2021_tcia", "BraTS 2021 (TCIA)", Verdict.ok, "https://wiki.cancerimagingarchive.net/x/sIbECQ",
       "2026-11-25", "CC BY 4.0 attribution + TCIA citation.", True),
    _D("mimic_cxr", "MIMIC-CXR", Verdict.research_only, "https://physionet.org/content/mimic-cxr/2.1.0/",
       "2026-11-25", "Cite PhysioNet and Johnson et al.", False),
    _D("chexpert", "CheXpert", Verdict.research_only, "https://stanfordmlgroup.github.io/competitions/chexpert/",
       "2026-11-25", "Stanford Research Use Agreement.", False),
    _D("vindr_cxr", "VinDr-CXR", Verdict.research_only, "https://physionet.org/content/vindr-cxr/1.0.0/",
       "2026-11-25", "Cite Nguyen et al. 2022.", False),
    _D("padchest", "PadChest", Verdict.research_only, "https://bimcv.cipf.es/bimcv-projects/padchest/",
       "2026-11-25", "PadChest Research Use Agreement.", False),
    _D("synthetic", "Synthetic phantoms (generated in-repo)", Verdict.ok, "radio.ingest.synthetic", "2026-11-25",
       "None.", True),
]}

_W = WeightsEntry
WEIGHTS: dict[str, WeightsEntry] = {w.weights_id: w for w in [
    _W("random", Verdict.ok, "n/a", "nothing", "n/a"),
    _W("vit_small_patch14_dinov2.lvd142m", Verdict.ok, "Apache-2.0", "LVD-142M (web images)",
       "https://github.com/facebookresearch/dinov2"),
    _W("vit_base_patch14_dinov2.lvd142m", Verdict.ok, "Apache-2.0", "LVD-142M (web images)",
       "https://github.com/facebookresearch/dinov2"),
    _W("torchvision_resnet50_imagenet", Verdict.conditional, "BSD-3 code; ImageNet terms may apply to weights",
       "ImageNet-1k", "https://pytorch.org/vision/stable/models.html"),
    _W("timm_imagenet", Verdict.conditional, "Apache-2.0 code; ImageNet terms may apply to weights",
       "ImageNet", "https://github.com/huggingface/pytorch-image-models"),
    _W("rad_dino", Verdict.research_only, "MIT weights; card states research use", "MIMIC, CheXpert, PadChest, NIH, BRAX",
       "https://huggingface.co/microsoft/rad-dino"),
]}


class LicenceGateError(RuntimeError):
    pass


def weights_id_for(encoder: str, pretrained: bool) -> str:
    if not pretrained:
        return "random"
    if encoder in WEIGHTS:
        return encoder
    return "timm_imagenet"


def lineage(dataset_ids: list[str], weights_id: str, purpose: str) -> dict:
    """Compute lineage tags; raise LicenceGateError for disallowed product runs."""
    unknown = [d for d in dataset_ids if d not in DATASETS]
    if unknown:
        raise LicenceGateError(f"datasets not in registry: {unknown}")
    if weights_id not in WEIGHTS:
        raise LicenceGateError(f"weights '{weights_id}' not in registry")
    ds = [DATASETS[d] for d in dataset_ids]
    w = WEIGHTS[weights_id]
    research_only = any(d.verdict != Verdict.ok for d in ds) or w.verdict != Verdict.ok
    if purpose not in ("research", "product"):
        raise LicenceGateError(f"purpose must be 'research' or 'product', got {purpose!r}")
    if purpose == "product" and research_only:
        bad = [f"{d.dataset_id}={d.verdict}" for d in ds if d.verdict != Verdict.ok]
        if w.verdict != Verdict.ok:
            bad.append(f"weights {w.weights_id}={w.verdict}")
        raise LicenceGateError(f"purpose=product refused; non-OK inputs: {bad}")
    return {
        "purpose": purpose,
        "research_only": research_only,
        "datasets": [asdict(d) for d in ds],
        "weights_init": asdict(w),
        "code_licences": ["radio (proprietary)", "timm Apache-2.0", "torch BSD-3"],
    }

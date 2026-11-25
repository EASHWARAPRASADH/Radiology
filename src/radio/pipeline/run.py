"""End-to-end analysis pipeline (stages 1-11) with explicit abstention paths.

    ingest/validate -> quality gate -> preprocess -> encoder+classifier (+activation maps)
    -> OOD gate -> calibration -> tri-state decisions -> normalisation to IR -> critical flags
    -> grounded draft report -> consistency check -> audit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from radio.audit.log import AuditLog
from radio.calibrate.conformal import decide
from radio.ingest.dicom import LoadedImage, UnsupportedInput, load_image
from radio.ontology.cxr import CONCEPTS
from radio.pipeline.bundle import ModelBundle
from radio.preprocess.transforms import Letterbox, prepare
from radio.qc.quality import QualityConfig, assess_quality
from radio.report.generator import GENERATOR_ID, generate_report
from radio.schema.findings import (
    Abstention,
    AnalysisResult,
    BoundingBox,
    ClinicalContext,
    CriticalStatus,
    Finding,
    FindingStatus,
    ImageQuality,
    Localization,
    LocalizationMethod,
    ModelEvidence,
    Provenance,
    QualityStatus,
    StudyInfo,
    UncertaintyStatus,
)


@dataclass
class AnalysisOutput:
    result: AnalysisResult
    heatmaps: dict[str, np.ndarray] = field(default_factory=dict)  # concept -> HxW in source coords [0,1]
    image: LoadedImage | None = None


def _study_info(img: LoadedImage | None) -> StudyInfo:
    if img is None:
        return StudyInfo(modality="unknown", anatomical_region="unknown", source_format="dicom")
    return StudyInfo(
        study_instance_uid=img.study_instance_uid, series_instance_uid=img.series_instance_uid,
        sop_instance_uid=img.sop_instance_uid, modality=img.modality, anatomical_region="chest",
        view_position=img.view_position, source_format=img.source_format,  # type: ignore[arg-type]
        image_rows=img.shape[0], image_columns=img.shape[1],
    )


def _not_assessed_all(reason: str) -> list[Finding]:
    return [Finding(code=k, display=c.display, status=FindingStatus.not_assessed, provenance=Provenance.model_inferred,
                    not_assessed_reason=reason) for k, c in CONCEPTS.items()]


def _bbox_from_map(m: np.ndarray, lb: Letterbox, rel: float = 0.5) -> tuple[BoundingBox | None, np.ndarray]:
    """Upsample a token-grid map to input size, crop padding, threshold, map bbox to source pixels."""
    t = torch.from_numpy(m)[None, None].float()
    up = F.interpolate(t, size=(lb.size, lb.size), mode="bilinear", align_corners=False)[0, 0].numpy()
    h_in, w_in = round(lb.src_h * lb.scale), round(lb.src_w * lb.scale)
    crop = up[lb.pad_y : lb.pad_y + h_in, lb.pad_x : lb.pad_x + w_in]
    pos = np.clip(crop, 0, None)
    norm = pos / pos.max() if pos.max() > 0 else pos
    if pos.max() <= 0:
        return None, norm
    ys, xs = np.nonzero(pos >= rel * pos.max())
    x0, y0 = lb.to_source(xs.min() + lb.pad_x, ys.min() + lb.pad_y)
    x1, y1 = lb.to_source(xs.max() + 1 + lb.pad_x, ys.max() + 1 + lb.pad_y)
    if x1 <= x0 or y1 <= y0:
        return None, norm
    return BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1), norm


class AnalysisPipeline:
    def __init__(self, bundle: ModelBundle, device: torch.device | str = "cpu",
                 quality_cfg: QualityConfig = QualityConfig(), audit: AuditLog | None = None):
        self.bundle = bundle
        self.device = torch.device(device)
        self.quality_cfg = quality_cfg
        self.audit = audit

    def _versions(self) -> dict[str, str]:
        return {self.bundle.model_id: self.bundle.model_version, "report_generator": GENERATOR_ID,
                "pipeline": "radio.pipeline-v0.1"}

    def _abstain(self, img: LoadedImage | None, quality: ImageQuality, reason: str, detail: str | None,
                 ctx: ClinicalContext) -> AnalysisOutput:
        result = AnalysisResult(
            study=_study_info(img), clinical_context=ctx, image_quality=quality,
            findings=_not_assessed_all(f"abstained:{reason}"), critical_status=CriticalStatus.not_assessed,
            uncertainty_status=UncertaintyStatus.high, abstention=Abstention(reason=reason, detail=detail),
            model_versions=self._versions(),
        )
        self._log("analysis_abstained", result)
        return AnalysisOutput(result=result, image=img)

    def _log(self, event: str, result: AnalysisResult) -> None:
        if self.audit is not None:
            self.audit.append(event, actor="system:pipeline", subject=result.result_id,
                              payload=result.model_dump(mode="json"))

    def analyze(self, path: str | Path | None = None, *, image: LoadedImage | None = None,
                context: ClinicalContext | None = None, view_position: str | None = None) -> AnalysisOutput:
        ctx = context or ClinicalContext()
        not_q = ImageQuality(status=QualityStatus.not_assessed)
        # 1. ingest + validate
        if image is None:
            try:
                image = load_image(path, view_position=view_position)  # type: ignore[arg-type]
            except UnsupportedInput as e:
                return self._abstain(None, not_q, e.reason, e.detail, ctx)
        # 2. classical quality gate
        quality = assess_quality(image, self.quality_cfg)
        if quality.status == QualityStatus.non_diagnostic:
            return self._abstain(image, quality, "non_diagnostic_quality", ", ".join(quality.issues), ctx)
        # 3-5. preprocess + model
        x, lb = prepare(image.pixels, self.bundle.config.img_size)
        with torch.no_grad():
            logits, feats, maps = self.bundle.model.forward_with_maps(x[None].to(self.device))
        logits_np = logits[0].float().cpu().numpy()
        feats_np = feats[0].float().cpu().numpy()
        maps_np = maps[0].float().cpu().numpy()
        # 4b. OOD gate on embeddings
        score = float(self.bundle.ood.score(feats_np)[0])
        quality = quality.model_copy(update={"ood_score": score, "ood_threshold": self.bundle.ood.threshold,
                                             "out_of_distribution": score > self.bundle.ood.threshold})
        if quality.out_of_distribution:
            return self._abstain(image, quality, "out_of_distribution",
                                 f"mahalanobis={score:.2f} > {self.bundle.ood.threshold:.2f}", ctx)
        # 7. calibration + tri-state decisions; 8-9. normalise to IR
        findings: list[Finding] = []
        heatmaps: dict[str, np.ndarray] = {}
        labels = self.bundle.labels
        for j, th in enumerate(self.bundle.thresholds):
            code = labels[j]
            c = CONCEPTS[code]
            raw_p = float(1 / (1 + np.exp(-logits_np[j])))
            cal_p = float(1 / (1 + np.exp(-logits_np[j] / th.temperature)))
            status = decide(cal_p, th)
            if status == FindingStatus.not_assessed:
                findings.append(Finding(code=code, display=c.display, status=status,
                                        provenance=Provenance.model_inferred,
                                        not_assessed_reason="insufficient_validation_data"))
                continue
            reasons: list[str] = []
            if status == FindingStatus.uncertain:
                reasons.append("calibrated_probability_between_thresholds")
            if code == "cardiomegaly" and image.view_position == "AP" and status == FindingStatus.present:
                status, reasons = FindingStatus.uncertain, [*reasons, "ap_projection_magnifies_cardiac_silhouette"]
            if quality.status == QualityStatus.limited and status == FindingStatus.uncertain:
                reasons.append("limited_image_quality")
            locs: list[Localization] = []
            if status in (FindingStatus.present, FindingStatus.uncertain):
                bbox, hm = _bbox_from_map(maps_np[j], lb)
                heatmaps[code] = hm
                if bbox is not None:
                    locs.append(Localization(method=LocalizationMethod.activation_map, bbox=bbox,
                                             heatmap_uri=f"heatmap:{code}", validated=False))
            findings.append(Finding(
                code=code, display=c.display, status=status, provenance=Provenance.model_inferred,
                localizations=locs, uncertainty_reasons=reasons,
                evidence=ModelEvidence(model_id=self.bundle.model_id, model_version=self.bundle.model_version,
                                       raw_probability=raw_p, calibrated_probability=cal_p,
                                       threshold_absent=min(th.t_absent, 1.0),
                                       threshold_present=min(th.t_present, 1.0)),
                critical=c.critical_candidate and status in (FindingStatus.present, FindingStatus.uncertain),
            ))
        for k, c in CONCEPTS.items():
            if k not in labels:
                findings.append(Finding(code=k, display=c.display, status=FindingStatus.not_assessed,
                                        provenance=Provenance.model_inferred,
                                        not_assessed_reason="not_in_model_label_space"))
        # 10. critical / uncertainty roll-up
        critical = CriticalStatus.suspected if any(f.critical for f in findings) else CriticalStatus.none_detected
        elevated = quality.status == QualityStatus.limited or any(f.status == FindingStatus.uncertain for f in findings)
        result = AnalysisResult(
            study=_study_info(image), clinical_context=ctx, image_quality=quality, findings=findings,
            critical_status=critical,
            uncertainty_status=UncertaintyStatus.elevated if elevated else UncertaintyStatus.low,
            model_versions=self._versions(),
        )
        # 11-12. grounded report + consistency check
        report = generate_report(result)
        result = result.model_copy(update={
            "report": report,
            "impression": [s.text for s in report.impression],
        })
        result = AnalysisResult.model_validate(result.model_dump())
        self._log("analysis_created", result)
        return AnalysisOutput(result=result, heatmaps=heatmaps, image=image)

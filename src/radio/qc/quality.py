"""Classical image-quality checks (stage 4a). Cheap, explainable, run before any model.

Thresholds are conservative engineering defaults, to be re-tuned on site data and documented in the
validation plan. They gate obvious failures (blank, saturated, tiny, extreme aspect) - they are not a
substitute for radiologist assessment of positioning/inspiration.
"""

from __future__ import annotations

from dataclasses import dataclass

from radio.ingest.dicom import LoadedImage
from radio.schema.findings import ImageQuality, QualityStatus


@dataclass(frozen=True)
class QualityConfig:
    min_side_px: int = 256
    recommended_side_px: int = 1024
    min_std: float = 0.02
    max_saturated_fraction: float = 0.40
    max_aspect_ratio: float = 2.0
    mean_low: float = 0.05
    mean_high: float = 0.95


def assess_quality(img: LoadedImage, cfg: QualityConfig = QualityConfig()) -> ImageQuality:
    px = img.pixels
    h, w = px.shape
    issues: list[str] = []
    fatal = False

    if min(h, w) < cfg.min_side_px:
        issues.append(f"resolution_too_low:{h}x{w}")
        fatal = True
    elif min(h, w) < cfg.recommended_side_px:
        issues.append(f"resolution_below_recommended:{h}x{w}")

    if max(h, w) / max(1, min(h, w)) > cfg.max_aspect_ratio:
        issues.append("extreme_aspect_ratio")
        fatal = True

    std = float(px.std())
    if std < cfg.min_std:
        issues.append("blank_or_uniform_image")
        fatal = True

    sat = float(((px <= 0.005) | (px >= 0.995)).mean())
    if sat > cfg.max_saturated_fraction:
        issues.append(f"saturated_fraction_high:{sat:.2f}")

    mean = float(px.mean())
    if mean < cfg.mean_low:
        issues.append("underexposed_or_dark")
    elif mean > cfg.mean_high:
        issues.append("overexposed_or_bright")

    if img.bits_stored is not None and img.bits_stored < 8:
        issues.append(f"low_bit_depth:{img.bits_stored}")

    if img.view_position == "AP":
        issues.append("ap_projection_cardiac_size_limited")  # AP magnifies the cardiac silhouette

    issues.extend(w_ for w_ in img.warnings if w_ in ("view_position_missing", "body_part_missing"))

    if fatal:
        status = QualityStatus.non_diagnostic
    elif any(i.split(":")[0] in {"resolution_below_recommended", "saturated_fraction_high", "underexposed_or_dark",
                                  "overexposed_or_bright", "low_bit_depth"} for i in issues):
        status = QualityStatus.limited
    else:
        status = QualityStatus.adequate
    return ImageQuality(status=status, issues=issues)

"""Image ingestion: DICOM (CR/DX) and PNG/JPEG -> normalised float image + non-PHI metadata.

Stage 1 (validation/parsing) of the pipeline. Classical/rule-based - no learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut

SUPPORTED_MODALITIES = {"CR", "DX"}
FRONTAL_VIEWS = {"PA", "AP"}

# Direct identifiers we expect to be removed/blank before data reaches the research environment.
PHI_KEYWORDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "ReferringPhysicianName", "InstitutionAddress",
    "AccessionNumber",
)


class UnsupportedInput(Exception):
    """Input cannot be analysed (wrong modality/body part/view, unreadable, ...)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class LoadedImage:
    pixels: np.ndarray  # float32, HxW, [0, 1], higher = brighter (MONOCHROME2 convention)
    source_format: str  # dicom | png | jpeg
    modality: str
    body_part: str | None
    view_position: str | None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    bits_stored: int | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    phi_tags_present: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset: pydicom.Dataset | None = None  # kept for SR export references (never serialised)

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape  # type: ignore[return-value]


def _to_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def load_dicom(path: str | Path, *, require_frontal: bool = True) -> LoadedImage:
    try:
        ds = pydicom.dcmread(str(path))
    except Exception as e:  # noqa: BLE001
        raise UnsupportedInput("unreadable_dicom", str(e)) from e

    modality = str(getattr(ds, "Modality", "") or "").upper()
    if modality not in SUPPORTED_MODALITIES:
        raise UnsupportedInput("unsupported_modality", f"Modality={modality or 'missing'}; supported: CR, DX")
    if "PixelData" not in ds:
        raise UnsupportedInput("no_pixel_data")
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        raise UnsupportedInput("multiframe_not_supported")

    warnings: list[str] = []
    body_part = (str(getattr(ds, "BodyPartExamined", "") or "").upper()) or None
    if body_part is None:
        warnings.append("body_part_missing")
    elif "CHEST" not in body_part and "THORAX" not in body_part:
        raise UnsupportedInput("unsupported_body_part", f"BodyPartExamined={body_part}")

    view = (str(getattr(ds, "ViewPosition", "") or "").upper()) or None
    if view is None:
        warnings.append("view_position_missing")
    elif require_frontal and view not in FRONTAL_VIEWS:
        raise UnsupportedInput("unsupported_view", f"ViewPosition={view}; phase 1 supports PA/AP only")

    try:
        arr = ds.pixel_array
    except Exception as e:  # noqa: BLE001
        raise UnsupportedInput("pixel_decode_failed", str(e)) from e
    if arr.ndim != 2:
        raise UnsupportedInput("unsupported_pixel_layout", f"shape={arr.shape}")

    arr = apply_modality_lut(arr, ds)
    if "WindowCenter" in ds or "VOILUTSequence" in ds:
        arr = apply_voi_lut(arr, ds)
    img = _to_unit(np.asarray(arr))
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        img = 1.0 - img

    phi = [k for k in PHI_KEYWORDS if str(getattr(ds, k, "") or "").strip()]
    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
    return LoadedImage(
        pixels=img,
        source_format="dicom",
        modality=modality,
        body_part=body_part,
        view_position=view,
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", "") or "") or None,
        series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "") or "") or None,
        sop_instance_uid=str(getattr(ds, "SOPInstanceUID", "") or "") or None,
        bits_stored=int(getattr(ds, "BitsStored", 0) or 0) or None,
        pixel_spacing_mm=(float(spacing[0]), float(spacing[1])) if spacing else None,
        phi_tags_present=phi,
        warnings=warnings,
        dataset=ds,
    )


def load_raster(path: str | Path, *, view_position: str | None = None) -> LoadedImage:
    """PNG/JPEG (e.g. NIH ChestX-ray14). Modality/body part are asserted by the caller, not verified."""
    p = Path(path)
    try:
        with Image.open(p) as im:
            mode = im.mode
            if mode in ("I;16", "I;16B", "I"):
                arr = np.asarray(im, dtype=np.float32)
                bits = 16
            else:
                arr = np.asarray(im.convert("L"), dtype=np.float32)
                bits = 8
    except Exception as e:  # noqa: BLE001
        raise UnsupportedInput("unreadable_image", str(e)) from e
    fmt = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
    return LoadedImage(
        pixels=_to_unit(arr),
        source_format=fmt,
        modality="DX",
        body_part="CHEST",
        view_position=view_position.upper() if view_position else None,
        bits_stored=bits,
        warnings=["modality_and_body_part_asserted_not_verified"],
    )


def load_image(path: str | Path, **kw) -> LoadedImage:
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return load_raster(p, view_position=kw.get("view_position"))
    return load_dicom(p, require_frontal=kw.get("require_frontal", True))

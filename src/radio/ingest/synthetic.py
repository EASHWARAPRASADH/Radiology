"""Synthetic DX DICOM writer (PHI-free) for tests and demos."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian, generate_uid


def phantom(size: int = 1024, seed: int = 0) -> np.ndarray:
    """Crude CXR-like phantom, uint16 12-bit. Not anatomically meaningful."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    body = np.exp(-(((xx - 0.5) / 0.42) ** 8 + ((yy - 0.55) / 0.48) ** 8))
    lung_l = np.exp(-(((xx - 0.68) / 0.13) ** 2 + ((yy - 0.48) / 0.25) ** 2))
    lung_r = np.exp(-(((xx - 0.32) / 0.13) ** 2 + ((yy - 0.48) / 0.25) ** 2))
    img = 0.15 + 0.7 * body - 0.45 * (lung_l + lung_r) + 0.02 * rng.standard_normal((size, size))
    img = np.clip(img, 0, 1)
    return (img * 4095).astype(np.uint16)


def write_dx_dicom(
    pixels: np.ndarray,
    path: str | Path,
    *,
    view_position: str = "PA",
    body_part: str = "CHEST",
    modality: str = "DX",
    photometric: str = "MONOCHROME2",
) -> Path:
    path = Path(path)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.ViewPosition = view_position
    ds.PatientName = ""
    ds.PatientID = ""
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.PatientOrientation = ["L", "F"]
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    now = datetime.now()
    ds.StudyDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.StudyTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "Radio synthetic"
    ds.PresentationIntentType = "FOR PRESENTATION"
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.PatientIdentityRemoved = "YES"
    ds.BurnedInAnnotation = "NO"
    ds.PixelSpacing = [0.143, 0.143]
    ds.ImagerPixelSpacing = [0.143, 0.143]

    arr = np.ascontiguousarray(pixels.astype(np.uint16))
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.RescaleType = "US"
    ds.WindowCenter = 2048
    ds.WindowWidth = 4096
    ds.PixelData = arr.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)
    return path

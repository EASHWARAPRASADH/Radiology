"""Export AI findings as a DICOM Comprehensive SR following TID 1500 (Measurement Report), via highdicom.

* Present/uncertain findings with a box -> TID 1410 planar ROI group (closed POLYLINE in source pixels).
* Other assessed findings -> TID 1501 image-level group referencing the source image.
* Qualitative value uses DCM/SCT codes from pydicom's code dictionary: Present (SCT 52101004),
  Absent (SCT 272519000), Indeterminate (SCT 82334004) for "uncertain".
* Not-assessed findings are omitted: absence of a concept in the SR means "not reported", never "absent".
* Finding concepts use the private coding scheme 99RADIO until verified RadLex/SNOMED codes are mapped.
* The SR is flagged incomplete / unverified until a radiologist approves (CompletionFlag/VerificationFlag).
"""

from __future__ import annotations

import highdicom as hd
import numpy as np
import pydicom
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from radio.ontology.cxr import PRIVATE_CODING_SCHEME
from radio.schema.findings import AnalysisResult, FindingStatus, ReviewStatus

_VALUE = {
    FindingStatus.present: codes.SCT.Present,
    FindingStatus.absent: codes.SCT.Absent,
    FindingStatus.uncertain: codes.SCT.Indeterminate,
}


def _concept(code: str, display: str) -> hd.sr.CodedConcept:
    return hd.sr.CodedConcept(value=code[:16], scheme_designator=PRIVATE_CODING_SCHEME, meaning=display)


def build_sr(result: AnalysisResult, source: pydicom.Dataset) -> hd.sr.ComprehensiveSR:
    if result.abstention is not None:
        raise ValueError("no SR for abstained analyses")
    model_id, model_version = next(iter(result.model_versions.items()))
    algo = hd.sr.AlgorithmIdentification(name=model_id, version=model_version[:64])
    observer = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=generate_uid(), name="Radio CXR AI (research)", manufacturer_name="Radio",
            model_name=model_version[:64],
        ),
    )
    groups = []
    for i, f in enumerate(result.findings):
        if f.status == FindingStatus.not_assessed:
            continue
        tracking = hd.sr.TrackingIdentifier(uid=generate_uid(), identifier=f"{f.code}-{i}")
        evals = [hd.sr.QualitativeEvaluation(name=_concept(f.code, f.display), value=_VALUE[f.status])]
        box = next((loc.bbox for loc in f.localizations if loc.bbox is not None), None)
        if box is not None:
            pts = np.array([[box.x_min, box.y_min], [box.x_max, box.y_min], [box.x_max, box.y_max],
                            [box.x_min, box.y_max], [box.x_min, box.y_min]], dtype=np.float32)
            region = hd.sr.ImageRegion(
                graphic_type=hd.sr.GraphicTypeValues.POLYLINE, graphic_data=pts,
                source_image=hd.sr.SourceImageForRegion(source.SOPClassUID, source.SOPInstanceUID),
            )
            groups.append(hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=tracking, referenced_region=region, finding_type=codes.DCM.Finding,
                algorithm_id=algo, qualitative_evaluations=evals,
            ))
        else:
            groups.append(hd.sr.MeasurementsAndQualitativeEvaluations(
                tracking_identifier=tracking, finding_type=codes.DCM.Finding, algorithm_id=algo,
                qualitative_evaluations=evals,
                source_images=[hd.sr.SourceImageForMeasurementGroup.from_source_image(source)],
            ))
    report = hd.sr.MeasurementReport(
        observation_context=hd.sr.ObservationContext(observer_device_context=observer),
        procedure_reported=_concept("cxr", "Chest radiograph"),
        imaging_measurements=groups,
    )
    final = result.review_status in (ReviewStatus.approved, ReviewStatus.amended)
    verified = result.review_status == ReviewStatus.approved
    sr = hd.sr.ComprehensiveSR(
        evidence=[source], content=report[0], series_instance_uid=generate_uid(), series_number=990,
        sop_instance_uid=generate_uid(), instance_number=1, manufacturer="Radio (research prototype)",
        is_complete=final, is_final=final, is_verified=verified,
        # Prototype: reviewer id stands in for the verifying observer's name (production: from the IdP roster).
        verifying_observer_name=f"{result.final_reviewer()}^" if verified else None,  # single-component PN
        verifying_organization="Radio pilot site" if verified else None,
    )
    sr.SeriesDescription = "AI FINDINGS - DRAFT, NOT RADIOLOGIST VERIFIED" if not final else "AI FINDINGS (reviewed)"
    return sr

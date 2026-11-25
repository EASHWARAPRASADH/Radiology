"""Inference + review API (prototype). Synchronous inference; production uses a task queue (doc 10/11).

Safety properties:
* Results are created as ``draft``; only POST /v1/reports/{id}/review by a radiologist changes that.
* The AI draft is immutable - reviews are appended; the final text is stored alongside.
* Every create/view/review/export is written to the hash-chained audit log.

Auth is a stub (``X-User-Id`` / ``X-User-Role`` headers) standing in for OIDC + RBAC.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel

from radio.audit.log import AuditLog
from radio.pipeline.bundle import ModelBundle
from radio.pipeline.run import AnalysisOutput, AnalysisPipeline
from radio.schema.findings import AnalysisResult, ClinicalContext, FindingOverride, Review, ReviewAction


class ReviewRequest(BaseModel):
    action: ReviewAction
    final_report_text: str | None = None
    finding_overrides: list[FindingOverride] = []
    comment: str | None = None


class User(BaseModel):
    id: str
    role: str


def current_user(x_user_id: Annotated[str | None, Header()] = None,
                 x_user_role: Annotated[str | None, Header()] = None) -> User:
    if not x_user_id or not x_user_role:
        raise HTTPException(401, "missing X-User-Id / X-User-Role (prototype auth stub)")
    return User(id=x_user_id, role=x_user_role)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs: dict[str, AnalysisOutput] = {}

    def save(self, out: AnalysisOutput) -> None:
        self.outputs[out.result.result_id] = out
        (self.root / f"{out.result.result_id}.json").write_text(out.result.model_dump_json(indent=2))

    def get(self, rid: str) -> AnalysisOutput:
        if rid not in self.outputs:
            raise HTTPException(404, "unknown result id")
        return self.outputs[rid]


def create_app(bundle_dir: str | Path | None, storage_dir: str | Path = "runs/api_store",
               device: str = "cpu", detector_dir: str | Path | None = None) -> FastAPI:
    """``bundle_dir=None`` builds the app without a model (used for OpenAPI export)."""
    bundle = ModelBundle.load(bundle_dir, device=device) if bundle_dir else None
    store = Store(Path(storage_dir))
    audit = AuditLog(Path(storage_dir) / "audit.jsonl")
    detector = None
    if detector_dir:
        from radio.models.detector import DetectorBundle

        detector = DetectorBundle(detector_dir, device=device)
    pipeline = AnalysisPipeline(bundle, device=device, audit=audit, detector=detector) if bundle else None
    app = FastAPI(title="Radio CXR AI (research prototype)", version="0.1.0",
                  description="AI-generated drafts for radiologist review. Not a medical device.")
    app.state.pipeline = pipeline
    app.state.audit = audit

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def review_page():
        """Upload-and-review page for local testing (uses the same API and auth stub as any client)."""
        return (Path(__file__).parent / "static" / "review.html").read_text()

    @app.get("/v1/health")
    def health():
        return {"status": "ok" if bundle else "no_model", "model_version": bundle.model_version if bundle else None,
                "research_only": bundle.research_only if bundle else None, "detector": detector is not None,
                "audit_chain_valid": audit.verify()}

    @app.get("/v1/studies/{rid}/image.png")
    def get_image(rid: str, user: Annotated[User, Depends(current_user)]):
        """The analysed image (after DICOM VOI LUT / photometric handling) in source pixel coordinates, so
        finding boxes can be overlaid directly."""
        import numpy as np
        from PIL import Image

        out = store.get(rid)
        if out.image is None:
            raise HTTPException(409, "no decoded image for this study (input was rejected before decoding)")
        buf = io.BytesIO()
        Image.fromarray((np.clip(out.image.pixels, 0, 1) * 255).astype("uint8")).save(buf, format="PNG")
        audit.append("image_viewed", actor=user.id, subject=rid)
        return Response(buf.getvalue(), media_type="image/png")

    @app.post("/v1/studies", response_model=AnalysisResult, status_code=201)
    async def create_study(user: Annotated[User, Depends(current_user)], file: UploadFile = File(...),
                           indication: str | None = Form(None), view_position: str | None = Form(None),
                           prior_study_id: str | None = Form(None,
                                                             description="Earlier approved result of the SAME patient "
                                                             "(the caller is responsible for patient matching)"),
                           prior_study_date: str | None = Form(None)):
        if pipeline is None:
            raise HTTPException(503, "no model bundle loaded")
        priors = []
        if prior_study_id:
            from radio.report.comparison import PriorNotVerifiable, prior_from_result

            try:
                priors.append(prior_from_result(store.get(prior_study_id).result, prior_study_date))
            except PriorNotVerifiable as e:
                raise HTTPException(409, f"prior cannot be used for comparison: {e}") from e
        suffix = Path(file.filename or "upload.dcm").suffix or ".dcm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            out = pipeline.analyze(tmp.name, context=ClinicalContext(indication=indication, priors=priors),
                                   view_position=view_position)
        store.save(out)
        audit.append("study_submitted", actor=user.id, subject=out.result.result_id,
                     payload={"prior_study_id": prior_study_id} if prior_study_id else None)
        return out.result

    @app.get("/v1/studies/{rid}", response_model=AnalysisResult)
    def get_study(rid: str, user: Annotated[User, Depends(current_user)]):
        out = store.get(rid)
        audit.append("result_viewed", actor=user.id, subject=rid)
        return out.result

    @app.get("/v1/studies/{rid}/report", response_class=PlainTextResponse)
    def get_report(rid: str, user: Annotated[User, Depends(current_user)]):
        r = store.get(rid).result
        audit.append("report_viewed", actor=user.id, subject=rid)
        authored = r.radiologist_text()
        if authored:
            author, text = authored
            return (f"[{r.review_status.value.upper()} by {r.final_reviewer()}; text by {author}]\n\n{text}")
        if r.report is None:
            return f"AI abstained ({r.abstention.reason if r.abstention else 'unknown'}). No draft available."
        by = f" by {r.final_reviewer()}" if r.reviews else ""
        return f"[{r.review_status.value.upper()}{by}]\n\n" + r.report.to_text()

    @app.post("/v1/reports/{rid}/review", response_model=AnalysisResult)
    def review(rid: str, body: ReviewRequest, user: Annotated[User, Depends(current_user)]):
        if user.role != "radiologist":
            raise HTTPException(403, "only radiologists may review/approve reports")
        out = store.get(rid)
        try:
            rv = Review(reviewer_id=user.id, action=body.action, final_report_text=body.final_report_text,
                        finding_overrides=body.finding_overrides, comment=body.comment)
            updated = out.result.apply_review(rv)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        out.result = updated
        store.save(out)
        audit.append(f"report_{body.action.value}", actor=user.id, subject=rid,
                     payload=json.loads(rv.model_dump_json()))
        return updated

    @app.get("/v1/studies/{rid}/sr")
    def get_sr(rid: str, user: Annotated[User, Depends(current_user)]):
        from radio.export.dicom_sr import build_sr

        out = store.get(rid)
        if out.image is None or out.image.dataset is None or out.result.abstention is not None:
            raise HTTPException(409, "DICOM SR is only available for analysed DICOM inputs")
        sr = build_sr(out.result, out.image.dataset)
        buf = io.BytesIO()
        sr.save_as(buf)
        audit.append("sr_exported", actor=user.id, subject=rid)
        return Response(buf.getvalue(), media_type="application/dicom")

    @app.get("/v1/studies/{rid}/fhir")
    def get_fhir(rid: str, user: Annotated[User, Depends(current_user)]):
        from radio.export.fhir import build_diagnostic_report

        out = store.get(rid)
        audit.append("fhir_exported", actor=user.id, subject=rid)
        return json.loads(build_diagnostic_report(out.result).model_dump_json())

    return app

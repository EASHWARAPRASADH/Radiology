"""Radio command-line interface: ``uv run radio --help``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Radio CXR AI research prototype (not a medical device).")


@app.command("export-schema")
def export_schema(out: Path = Path("docs/blueprint/schemas/analysis-result.schema.json")):
    """Write the AnalysisResult JSON Schema (single source of truth: radio.schema.findings)."""
    from radio.schema.findings import json_schema

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_schema(), indent=2) + "\n")
    typer.echo(out)


@app.command("export-openapi")
def export_openapi(out: Path = Path("docs/blueprint/schemas/openapi.json")):
    """Write the API's OpenAPI document."""
    import tempfile

    from radio.api.app import create_app

    with tempfile.TemporaryDirectory() as d:
        spec = create_app(None, storage_dir=d).openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    typer.echo(out)


@app.command("make-synthetic")
def make_synthetic(out_dir: Path, n: int = 3):
    """Write PHI-free synthetic DX DICOM phantoms (plumbing tests only)."""
    from radio.ingest.synthetic import phantom, write_dx_dicom

    for i in range(n):
        typer.echo(write_dx_dicom(phantom(seed=i), out_dir / f"phantom_{i}.dcm"))


@app.command()
def train(config: Path, set_: list[str] = typer.Option([], "--set", help="Override, e.g. --set optim.epochs=1")):
    """Train the classifier from a YAML config."""
    import yaml

    from radio.train.train_classifier import TrainConfig
    from radio.train.train_classifier import train as run

    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in set_)}
    run(TrainConfig.from_yaml(config, overrides))


@app.command()
def calibrate(run_dir: Path, alpha_fn: float = 0.05, alpha_fp: float = 0.05, min_positives: int = 30,
              min_negatives: int = 100, ood_quantile: float = 0.995):
    """Fit temperatures, conformal thresholds and the OOD gate on the validation split -> run_dir/bundle."""
    from radio.pipeline.bundle import build_bundle

    out = build_bundle(run_dir, alpha_fn=alpha_fn, alpha_fp=alpha_fp, min_positives=min_positives,
                       min_negatives=min_negatives, ood_quantile=ood_quantile)
    cal = json.loads((out / "calibration.json").read_text())
    for t in cal["thresholds"]:
        flag = "validated" if t["validated"] else "NOT VALIDATED -> not_assessed"
        typer.echo(f"{t['label']:<20} T={t['temperature']:.2f} absent<{t['t_absent']:.3f} "
                   f"present>={min(t['t_present'], 1):.3f} pos={t['n_pos']:<4} {flag}")
    typer.echo(out)


@app.command()
def evaluate(run_dir: Path, split: str = "test", n_boot: int = 200):
    """Per-label metrics with patient-level bootstrap CIs on a held-out split."""
    import numpy as np

    from radio.calibrate.conformal import thresholds_from_json
    from radio.calibrate.temperature import apply_temperature
    from radio.eval.metrics import evaluate as ev

    p = np.load(run_dir / f"preds_{split}.npz", allow_pickle=True)
    cal = json.loads((run_dir / "bundle" / "calibration.json").read_text())
    ths = thresholds_from_json(cal["thresholds"])
    temps = np.array([t.temperature for t in ths])
    labels = [str(x) for x in p["labels"]]
    res = ev(apply_temperature(p["logits"], temps), p["y"], p["mask"], labels, p["patient_id"], ths, n_boot=n_boot)
    (run_dir / f"eval_{split}.json").write_text(json.dumps(res, indent=2))
    typer.echo(f"{'label':<20}{'n_pos':>6}{'AUROC':>8}{'95% CI':>17}{'AUPRC':>8}{'ECE':>7}{'defer':>7}{'sens*':>7}")
    for lab in labels:
        r = res[lab]
        ci = r.get("auroc_ci95")
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "n/a"
        defer = r.get("deferral_rate", float("nan"))
        sens = r.get("sensitivity_present_or_uncertain", float("nan"))
        typer.echo(f"{lab:<20}{int(r['n_pos']):>6}{r['auroc']:>8.3f}{ci_s:>17}{r['auprc']:>8.3f}{r['ece']:>7.3f}"
                   f"{defer:>7.2f}{sens:>7.2f}")
    s = res["_summary"]
    typer.echo(f"mean AUROC {s['mean_auroc']:.3f} | images {s['n_images']} | patients {s['n_patients']}")
    typer.echo("sens* = sensitivity counting 'uncertain' (deferred to radiologist) as flagged. Not a clinical claim.")


@app.command()
def infer(bundle: Path, image: Path, indication: str = typer.Option(None), view_position: str = typer.Option(None),
          json_out: Path = typer.Option(None, "--json"), heatmap_dir: Path = typer.Option(None)):
    """Analyse one image and print the draft report."""
    from radio.pipeline.bundle import ModelBundle
    from radio.pipeline.run import AnalysisPipeline
    from radio.schema.findings import ClinicalContext

    pipe = AnalysisPipeline(ModelBundle.load(bundle))
    out = pipe.analyze(image, context=ClinicalContext(indication=indication), view_position=view_position)
    r = out.result
    if json_out:
        json_out.write_text(r.model_dump_json(indent=2))
    if heatmap_dir and out.heatmaps:
        from PIL import Image

        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for code, hm in out.heatmaps.items():
            Image.fromarray((hm * 255).astype("uint8")).save(heatmap_dir / f"{code}.png")
    if r.abstention:
        typer.echo(f"ABSTAINED: {r.abstention.reason} {r.abstention.detail or ''}")
        return
    typer.echo(r.report.to_text())
    typer.echo(f"\nvalidation_passed={r.report.validation_passed} critical={r.critical_status.value} "
               f"uncertainty={r.uncertainty_status.value} ood={r.image_quality.ood_score:.2f}"
               f"/{r.image_quality.ood_threshold:.2f}")


@app.command()
def serve(bundle: Path, host: str = "127.0.0.1", port: int = 8000, storage: Path = Path("runs/api_store")):
    """Run the inference/review API."""
    import uvicorn

    from radio.api.app import create_app

    uvicorn.run(create_app(bundle, storage_dir=storage), host=host, port=port)


if __name__ == "__main__":
    app()

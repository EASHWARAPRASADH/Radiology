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


@app.command("train-detector")
def train_detector_cmd(config: Path, set_: list[str] = typer.Option([], "--set", help="e.g. --set epochs=1")):
    """Phase 2: train the Faster R-CNN detector on ChestX-Det, pick operating points, gate per class."""
    import yaml

    from radio.train.train_detector import DetTrainConfig, train_detector

    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in set_)}
    train_detector(DetTrainConfig.from_yaml(config, overrides))


@app.command("dedupe-chestx-det")
def dedupe_chestx_det(det_root: Path = Path("data/chestx_det"), nih_images: Path = Path("data/nih/images"),
                      out: Path = Path("data/chestx_det/nih_matches.json")):
    """Find ChestX-Det images that are copies of local NIH CXR14 images (fingerprint match)."""
    from radio.data.chestx_det import fingerprint_matches

    det = sorted(det_root.glob("*/*.png"))
    nih = sorted(nih_images.rglob("*.png"))
    m = fingerprint_matches(det, nih)
    out.write_text(json.dumps(m, indent=2))
    typer.echo(f"{len(m)} of {len(det)} ChestX-Det images match one of {len(nih)} local NIH images -> {out}")


@app.command("leak-check-chestx-det")
def leak_check(det_root: Path = Path("data/chestx_det"), matches: Path = Path("data/chestx_det/nih_matches.json"),
               nih_csv: Path = Path("data/nih/Data_Entry_2017.csv"),
               out: Path = Path("data/chestx_det/test_patient_leak.json")):
    """List ChestX-Det test images whose NIH patient also appears in ChestX-Det train."""
    from radio.data.chestx_det import patient_leak_report

    rep = patient_leak_report(det_root, json.loads(matches.read_text()), nih_csv)
    out.write_text(json.dumps(rep, indent=2))
    typer.echo({k: v for k, v in rep.items() if k != "leaky_test_images"})


@app.command("regate-detector")
def regate(run_dir: Path, leak_file: Path = Path("data/chestx_det/test_patient_leak.json")):
    """Re-score the detector on the test split minus known patient-leaky images and re-apply the Phase-2 gate."""
    from radio.train.train_detector import regate_detector

    exclude = set(json.loads(leak_file.read_text())["leaky_test_images"]) if leak_file.exists() else set()
    res = regate_detector(run_dir, exclude)
    full = json.loads((run_dir / "eval_test.json").read_text())
    typer.echo(f"{'class':<20}{'boxes':>6}{'AP25 all':>9}{'AP25 clean':>11}{'S@1fp all':>10}{'S@1fp clean':>12}"
               f"{'S@1fp leaky':>12}{'gate FP/img':>12}{'S@gate':>7}{'95% CI':>11}  gate")
    for c, v in res["leak_reduced"].items():
        if c.startswith("_"):
            continue
        lk = res["leaky_only"][c]["sens@1.0fp"] if res["leaky_only"] else float("nan")
        g = res["gates"][c]
        ci = f"{g['test_sens_ci_lower']:.2f}-{g['test_sens_ci_upper']:.2f}"
        typer.echo(f"{c:<20}{int(v['n_gt_boxes']):>6}{full[c]['ap25']:>9.3f}{v['ap25']:>11.3f}"
                   f"{full[c]['sens@1.0fp']:>10.2f}{v['sens@1.0fp']:>12.2f}{lk:>12.2f}{g['gate_fppi']:>12}"
                   f"{g['test_sens_at_gate_fppi']:>7.2f}{ci:>11}  {'pass' if c in res['validated'] else 'fail'}")
    typer.echo(f"summary all={full['_summary']} clean={res['leak_reduced']['_summary']}")
    g0 = next(iter(res["gates"].values()), {})
    typer.echo(f"gate: {g0.get('gate_statistic')} of sens at the class's FP/image >= {g0.get('min_sensitivity')} "
               f"({g0.get('ci_method')}, level {g0.get('ci_level')}), >= {g0.get('min_lesions')} boxes")
    typer.echo(f"validated after re-gate (excluded {len(exclude)} leaky images): {res['validated']}")


@app.command("validator-suite")
def validator_suite(n: int = 2000, seed: int = 0, out: Path = typer.Option(None)):
    """WP 3.2: inject known report errors into template drafts of random IRs; report validator detection rates."""
    import random

    from radio.report.perturb import random_result, run_suite

    rng = random.Random(seed)
    res = run_suite([random_result(rng) for _ in range(n)], seed=seed)
    typer.echo(f"{n} random IRs; unperturbed drafts failing validation: {res['unperturbed_failures']}")
    typer.echo(f"{'probe':<32}{'applied':>8}{'any error':>11}{'expected':>10}")
    for k, v in res["probes"].items():
        typer.echo(f"{k:<32}{v['applied']:>8}{v['detected_any_error']:>11.3f}{v['detected_expected_code']:>10.3f}")
    rp = res["rephraser_synonym_injection"]
    typer.echo(f"hostile rephraser: {rp['synonyms_in_final_draft']} of {rp['sentences_offered']} sentences "
               "kept an injected synonym")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))


def _eval_set(det_root: Path, eval_set: str, matches: Path, nih_test_list: Path, leak_file: Path,
              include_file: Path | None):
    from radio.data.chestx_det import eval_records

    m = json.loads(matches.read_text()) if matches.exists() else {}
    nih_test = set(nih_test_list.read_text().split()) if nih_test_list.exists() else set()
    leaky = set(json.loads(leak_file.read_text())["leaky_test_images"]) if leak_file.exists() else set()
    include = set(json.loads(include_file.read_text())) if include_file else None
    return eval_records(det_root, eval_set, m, nih_test, leaky, include)


@app.command("report-eval")
def report_eval(bundle: Path, detector: Path = typer.Option(None), det_root: Path = Path("data/chestx_det"),
                split: str = "test", out: Path = Path("runs/report_eval"),
                matches: Path = Path("data/chestx_det/nih_matches.json"),
                nih_test_list: Path = Path("data/nih/test_list.txt"),
                leak_file: Path = Path("data/chestx_det/test_patient_leak.json"),
                eval_set: str = typer.Option("chestx_test", help="chestx_test | nih_test_patients"),
                include_file: Path = typer.Option(None, help="JSON list of file names to restrict the set to"),
                view_position: str = "PA", limit: int = typer.Option(None)):
    """WP 3.4: run the full pipeline on ChestX-Det images and score the draft reports against radiologist labels.

    ``chestx_test`` excludes ChestX-Det copies of NIH images the classifier trained or calibrated on, and test
    images that share a patient with ChestX-Det train (the detector's training set). ``nih_test_patients`` is
    patient-clean for the classifier only, so it refuses ``--detector``."""
    from radio.data.chestx_det import DET_CONCEPTS
    from radio.eval.report_eval import evaluate_reports
    from radio.pipeline.bundle import ModelBundle
    from radio.pipeline.run import AnalysisPipeline

    if split != "test":
        raise typer.BadParameter("only the test split is supported; use --eval-set")
    if detector and eval_set == "nih_test_patients":
        raise typer.BadParameter("nih_test_patients includes the detector's training images; drop --detector")
    recs, n_excluded = _eval_set(det_root, eval_set, matches, nih_test_list, leak_file, include_file)
    recs = recs[:limit]
    det = None
    if detector:
        from radio.models.detector import DetectorBundle

        det = DetectorBundle(detector)
    pipe = AnalysisPipeline(ModelBundle.load(bundle), detector=det)
    out.mkdir(parents=True, exist_ok=True)
    results_dir = out / "results"
    results_dir.mkdir(exist_ok=True)

    def items():
        for i, rec in enumerate(recs):
            r = pipe.analyze(rec.path, view_position=view_position).result
            (results_dir / f"{rec.file_name}.json").write_text(r.model_dump_json())
            boxes = {c: rec.boxes[[lab == c for lab in rec.labels]] for c in DET_CONCEPTS}
            if (i + 1) % 50 == 0:
                typer.echo(f"  {i + 1}/{len(recs)}")
            yield rec.file_name, r, rec.image_labels, boxes

    res = evaluate_reports(items(), DET_CONCEPTS)
    res["excluded_images"] = n_excluded
    res["detector"] = str(detector) if detector else None
    (out / "report_eval.json").write_text(json.dumps(res, indent=2))
    s = res["summary"]
    typer.echo(f"{s['images']} images ({n_excluded} {split} images excluded), abstained {s['abstained']}, "
               f"validation pass {s['validation_pass_rate']:.3f}, false-normal reports {s['false_normal_reports']}")
    typer.echo(f"finding precision {s['finding_precision']:.3f} recall {s['finding_recall']:.3f} "
               f"hallucination {s['hallucination_rate']:.3f}; mean asserted {s['mean_asserted_findings']:.2f}, "
               f"uncertain {s['mean_uncertain_findings']:.2f}; reports with a false 'absent' "
               f"{s['reports_with_false_absent']}")
    typer.echo(f"{'concept':<20}{'ref+':>5}{'flagged':>9}{'false abs':>10}{'n/a':>6}{'PPV pres':>9}"
               f"{'loc claims':>11}{'loc hit':>8}")
    for c, v in res["per_concept"].items():
        p = v["ref_positive"]
        typer.echo(f"{c:<20}{p['n']:>5}{v['sensitivity_flagged']:>9.2f}{v['false_absent_rate']:>10.2f}"
                   f"{p['not_assessed']:>6.2f}{v['ppv_present']:>9.2f}{v['localised_claims']:>11}"
                   f"{v['localised_claim_hit_rate']:>8.2f}")
    typer.echo(out / "report_eval.json")


@app.command("eval-pointing")
def eval_pointing(bundle: Path, det_root: Path = Path("data/chestx_det"), run_dir: Path = typer.Option(None),
                  matches: Path = Path("data/chestx_det/nih_matches.json"), split: str = "test",
                  nih_test_list: Path = Path("data/nih/test_list.txt"),
                  leak_file: Path = Path("data/chestx_det/test_patient_leak.json"),
                  eval_set: str = typer.Option("chestx_test", help="chestx_test | nih_test_patients"),
                  include_file: Path = typer.Option(None, help="JSON list of file names to restrict the set to"),
                  tag: str = typer.Option("", help="suffix for the output file name")):
    """WP 2.6: classifier activation-map pointing game + image AUROC vs ChestX-Det radiologist labels.

    ``chestx_test`` excludes ChestX-Det copies of NIH images the classifier trained on, and test images that share
    a patient with ChestX-Det train. ``nih_test_patients`` uses ChestX-Det images of NIH official-test patients
    from both splits (patient-clean for the classifier)."""
    from radio.eval.pointing import pointing_game
    from radio.pipeline.bundle import ModelBundle

    if split != "test":
        raise typer.BadParameter("only the test split is supported; use --eval-set")
    b = ModelBundle.load(bundle)
    recs, _ = _eval_set(det_root, eval_set, matches, nih_test_list, leak_file, include_file)
    res = pointing_game(b, recs)
    name = f"pointing_{eval_set}{'_' + tag if tag else ''}.json"
    target = (run_dir or bundle.parent) / name
    target.write_text(json.dumps(res, indent=2))
    typer.echo(f"{'label':<20}{'lesion imgs':>12}{'hit rate':>10}{'chance':>9}{'AUROC(rad)':>12}{'pos':>6}")
    for k, v in res.items():
        if k.startswith("_"):
            continue
        typer.echo(f"{k:<20}{v['n_lesion_images']:>12}{v['pointing_hit_rate']:>10.2f}{v['chance_hit_rate']:>9.2f}"
                   f"{v['image_auroc_vs_radiologist']:>12.3f}{v['n_pos_images']:>6}")
    typer.echo(json.dumps(res["_summary"]))
    typer.echo(target)


@app.command()
def infer(bundle: Path, image: Path, indication: str = typer.Option(None), view_position: str = typer.Option(None),
          json_out: Path = typer.Option(None, "--json"), heatmap_dir: Path = typer.Option(None),
          detector: Path = typer.Option(None, help="Detector bundle dir (Phase 2)"),
          prior: Path = typer.Option(None, help="Approved AnalysisResult JSON of an earlier study, same patient"),
          prior_date: str = typer.Option(None)):
    """Analyse one image and print the draft report."""
    from radio.pipeline.bundle import ModelBundle
    from radio.pipeline.run import AnalysisPipeline
    from radio.schema.findings import ClinicalContext

    det = None
    if detector:
        from radio.models.detector import DetectorBundle

        det = DetectorBundle(detector)
    priors = []
    if prior:
        from radio.report.comparison import PriorNotVerifiable, prior_from_result
        from radio.schema.findings import AnalysisResult

        try:
            priors.append(prior_from_result(AnalysisResult.model_validate_json(prior.read_text()), prior_date))
        except PriorNotVerifiable as e:
            typer.echo(f"[prior not used for comparison: {e}]")
    pipe = AnalysisPipeline(ModelBundle.load(bundle), detector=det)
    out = pipe.analyze(image, context=ClinicalContext(indication=indication, priors=priors),
                       view_position=view_position)
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
def monitor(run_dir: Path, log: Path = typer.Option(None, help="Training stdout log to parse"),
            host: str = "127.0.0.1", port: int = 8766):
    """Live training dashboard (epoch curves, ETA, prediction previews) for a run directory."""
    import uvicorn

    from radio.monitor import create_monitor

    uvicorn.run(create_monitor(run_dir, log), host=host, port=port, log_level="warning")


@app.command()
def serve(bundle: Path, host: str = "127.0.0.1", port: int = 8000, storage: Path = Path("runs/api_store"),
          detector: Path = typer.Option(None, help="Detector bundle dir (Phase 2)")):
    """Run the inference/review API."""
    import uvicorn

    from radio.api.app import create_app

    uvicorn.run(create_app(bundle, storage_dir=storage, detector_dir=detector), host=host, port=port)


if __name__ == "__main__":
    app()

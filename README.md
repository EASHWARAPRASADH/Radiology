# Radio: radiologist-in-the-loop chest X-ray AI (research prototype)

> **Not a medical device.** Research prototype. Not cleared, approved or validated for clinical use. All AI output is a draft for review by a qualified radiologist.

Radio analyses a chest radiograph and produces three things:
- a **structured findings representation (IR)** where every concept is `present | absent | uncertain | not_assessed`
- an **evidence-grounded draft report** in which every sentence cites the findings it comes from
- **DICOM SR (TID 1500) and FHIR `DiagnosticReport`** exports that stay *preliminary* until a radiologist approves them

The design is hierarchical: perception models produce the IR, and the text layer only verbalises that IR. A rule-based validator rejects any sentence that isn't grounded in a finding.

- **R&D blueprint (17 deliverables):** [`docs/blueprint/00-index.md`](docs/blueprint/00-index.md)

## Quick start

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). On Apple Silicon, training uses MPS automatically.

```bash
uv sync                      # install
uv run pytest                # 49 tests: schema, report grounding, pipeline, API, exports, audit, detection
uv run radio --help
```

### Run the Phase-1 feasibility experiment on NIH ChestX-ray14

1. Download one image batch (1.9 GB) plus the metadata from the official NIH Box share into `data/nih`:

   ```bash
   scripts/fetch_nih_subset.sh 1
   ```

2. Train the DINOv2 ViT-S/14 classifier:

   ```bash
   uv run radio train configs/classifier_nih_smoke.yaml
   ```

   For the longer run, use `configs/classifier_nih.yaml`.

3. Calibrate. This fits temperatures, conformal thresholds and the out-of-distribution gate, and writes `runs/<run>/bundle`:

   ```bash
   uv run radio calibrate runs/nih_dinov2s_smoke
   ```

4. Evaluate. This reports AUROC, AUPRC, ECE and deferral rates with patient-level bootstrap confidence intervals:

   ```bash
   uv run radio evaluate runs/nih_dinov2s_smoke
   ```

5. Analyse a single image:

   ```bash
   uv run radio infer runs/nih_dinov2s_smoke/bundle data/nih/images/images_001/images/00000001_000.png \
       --view-position PA --heatmap-dir runs/heatmaps
   ```

   A synthetic phantom DICOM, which the out-of-distribution gate should reject:

   ```bash
   uv run radio make-synthetic runs/synthetic --n 1
   uv run radio infer runs/nih_dinov2s_smoke/bundle runs/synthetic/phantom_0.dcm
   ```

6. Serve the review API (OpenAPI docs at http://127.0.0.1:8000/docs):

   ```bash
   uv run radio serve runs/nih_dinov2s_smoke/bundle
   ```

### Phase 2: localisation (ChestX-Det)

1. Download ChestX-Det (1.4 GB). The script validates the archives before extracting them:

   ```bash
   scripts/fetch_chestx_det.sh
   ```

2. Find ChestX-Det copies of your local NIH images, and test images that share an NIH patient with training:

   ```bash
   uv run radio dedupe-chestx-det
   uv run radio leak-check-chestx-det
   ```

3. Train the detector. Faster R-CNN at 512 px takes about 14 min per epoch on an M4 Pro:

   ```bash
   uv run radio train-detector configs/detector_chestxdet.yaml
   ```

4. Re-score on the test set minus the known-leaky images, and gate each class on the lower 95% confidence bound of its sensitivity:

   ```bash
   uv run radio regate-detector runs/chestxdet_frcnn_512
   ```

5. Run the activation-map pointing game and score the classifier against radiologist labels:

   ```bash
   uv run radio eval-pointing runs/nih_dinov2s_224/bundle
   ```

6. Infer with the classifier and detector together:

   ```bash
   uv run radio infer runs/nih_dinov2s_224/bundle IMAGE --detector runs/chestxdet_frcnn_512/det_bundle
   ```

## Layout

```
src/radio/
  schema/findings.py     IR + JSON Schema (single source of truth)
  ontology/cxr.py        finding vocabulary, dataset label maps
  ingest/                DICOM/PNG ingestion, synthetic DICOM writer
  preprocess/            letterbox resize (no flips; laterality matters)
  qc/                    classical quality gate + Mahalanobis OOD gate
  data/                  NIH loader (patient-level splits), data/weights licence registry
  models/                DINOv2 encoder, classifier with activation maps, detector interface (Phase 2)
  train/                 training loop (MPS/CUDA/CPU), masked BCE/ASL
  calibrate/             temperature scaling, conformal present/uncertain/absent thresholds
  eval/                  classification + report-level metrics
  report/                template generator, consistency validator, approved-guideline registry
  pipeline/              bundle, end-to-end pipeline with abstention, Analyzer plug-in interface
  export/                DICOM SR (highdicom), FHIR R4B DiagnosticReport
  audit/                 hash-chained append-only audit log
  api/                   FastAPI inference + radiologist review
docs/blueprint/          research and design deliverables, generated schemas
configs/                 training configs
```

## Data and licences

- Phase-1 training uses only datasets with a commercial-use verdict of OK. See [`NOTICE-DATA.md`](NOTICE-DATA.md) and `src/radio/data/registry.py`.
- A run with `purpose: product` is refused if any dataset or weight initialisation isn't cleared.
- Research-only datasets (MIMIC-CXR, CheXpert, VinDr-CXR, PadChest, …) may be used for internal benchmarking only.
- Medical images are never committed: `data/` is git-ignored.

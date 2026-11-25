"""Train the finding detector on ChestX-Det (Phase 2), choose operating points on validation, gate per class
on the official test split, and write a detector bundle.

Run directory (runs/<run_name>/): config.yaml, model.pt, history.json, preds_{val,test}.json,
eval_test.json, det_bundle/.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from radio.data.chestx_det import DET_CONCEPTS, DetectionDataset, DetRecord, collate, load_split, split_train_val
from radio.eval.detection import (
    ImageGT,
    ImagePreds,
    class_tp_arrays,
    evaluate_detections,
    froc_sensitivity_ci,
    score_threshold_at_fppi,
)
from radio.models.detector import ClassOperatingPoint, DetectorConfig, build_detector, write_detector_bundle
from radio.train.train_classifier import pick_device


@dataclass
class DetTrainConfig:
    run_name: str = "chestxdet_frcnn"
    out_dir: str = "runs"
    purpose: str = "research"
    root: str = "data/chestx_det"
    dataset_ids: list[str] = field(default_factory=lambda: ["chestx_det"])
    val_frac: float = 0.15
    max_train_images: int | None = None
    min_size: int = 512
    pretrained_backbone: bool = True
    trainable_backbone_layers: int = 3
    epochs: int = 10
    batch_size: int = 4
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 200
    num_workers: int = 4
    device: str = "auto"
    seed: int = 0
    target_fppi: float = 1.0  # operating point per class, chosen on validation
    gate_fppi: float = 1.0  # Phase-2 gate: FROC sensitivity at this FP/image on test...
    gate_min_sensitivity: float = 0.80  # ...must reach this
    gate_min_lesions: int = 30  # and the test set must contain at least this many reference boxes
    gate_statistic: str = "lower_bound"  # "lower_bound" (image-level bootstrap CI) or "point" (Phase-2 original)
    gate_ci_level: float = 0.95  # two-sided level; the gate uses the lower end
    gate_bootstrap: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> DetTrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update(overrides or {})
        return cls(**raw)


def _gt(records: list[DetRecord]) -> list[ImageGT]:
    return [ImageGT(boxes=r.boxes, labels=list(r.labels)) for r in records]


def gate_class(cfg: DetTrainConfig, preds: list[ImagePreds], gts: list[ImageGT], concept: str,
               te: dict) -> tuple[bool, dict]:
    """Apply the per-class localisation gate. With ``gate_statistic="lower_bound"`` the class passes only if
    the lower end of the image-bootstrap CI for sensitivity at ``gate_fppi`` reaches ``gate_min_sensitivity``."""
    if cfg.gate_statistic not in ("lower_bound", "point"):
        raise ValueError(f"gate_statistic must be 'lower_bound' or 'point', got {cfg.gate_statistic!r}")
    ci = froc_sensitivity_ci(preds, gts, concept, fppi=cfg.gate_fppi, n_boot=cfg.gate_bootstrap,
                             level=cfg.gate_ci_level, seed=cfg.seed)
    sens = te.get(f"sens@{cfg.gate_fppi}fp", float("nan"))
    stat = ci["lower"] if cfg.gate_statistic == "lower_bound" else sens
    validated = bool(te["n_gt_boxes"] >= cfg.gate_min_lesions and not np.isnan(stat)
                     and stat >= cfg.gate_min_sensitivity)
    gate = {"test_sens_at_gate_fppi": sens, "test_sens_ci_lower": ci["lower"], "test_sens_ci_upper": ci["upper"],
            "ci_level": cfg.gate_ci_level, "ci_method": f"image_bootstrap_percentile_n{ci['n_boot']}",
            "gate_statistic": cfg.gate_statistic, "gate_fppi": cfg.gate_fppi,
            "test_n_gt_boxes": te["n_gt_boxes"], "test_ap25": te["ap25"],
            "min_sensitivity": cfg.gate_min_sensitivity, "min_lesions": cfg.gate_min_lesions}
    return validated, gate


@torch.no_grad()
def predict(model, records: list[DetRecord], device, batch_size: int = 4) -> list[ImagePreds]:
    model.eval()
    ds = DetectionDataset(records, DET_CONCEPTS)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    out: list[ImagePreds] = []
    for imgs, _ in dl:
        res = model([im.to(device) for im in imgs])
        for o in res:
            out.append(ImagePreds(boxes=o["boxes"].cpu().numpy(), scores=o["scores"].cpu().numpy(),
                                  labels=[DET_CONCEPTS[int(i) - 1] for i in o["labels"].cpu().numpy()]))
    return out


def _dump_preds(path: Path, records, preds):
    path.write_text(json.dumps([
        {"file_name": r.file_name, "boxes": p.boxes.round(1).tolist(), "scores": p.scores.round(4).tolist(),
         "labels": p.labels} for r, p in zip(records, preds, strict=True)]))


def train_detector(cfg: DetTrainConfig) -> Path:
    from radio.data.registry import lineage

    weights_id = "torchvision_resnet50_imagenet" if cfg.pretrained_backbone else "random"
    lin = lineage(cfg.dataset_ids, weights_id, cfg.purpose)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    train_all = load_split(cfg.root, "train")
    test = load_split(cfg.root, "test")
    train, val = split_train_val(train_all, cfg.val_frac)
    if cfg.max_train_images:
        train = train[: cfg.max_train_images]
    print(f"[data] train={len(train)} val={len(val)} test={len(test)} device={device}", flush=True)

    dcfg = DetectorConfig(concepts=tuple(DET_CONCEPTS), pretrained_backbone=cfg.pretrained_backbone,
                          min_size=cfg.min_size, max_size=cfg.min_size,
                          trainable_backbone_layers=cfg.trainable_backbone_layers)
    model = build_detector(dcfg).to(device)
    dl = DataLoader(DetectionDataset(train, DET_CONCEPTS, augment=True), batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=collate, persistent_workers=cfg.num_workers > 0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup_iters) * 0.5 * (
        1 + math.cos(math.pi * s / max(1, total))))

    history, best, step = [], -1.0, 0
    for epoch in range(cfg.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for imgs, targets in dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {losses}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            run += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{total} loss {loss.item():.3f}", flush=True)
        vp = predict(model, val, device)
        ev = evaluate_detections(vp, _gt(val), DET_CONCEPTS)["_summary"]
        rec = {"epoch": epoch, "train_loss": run / max(1, len(dl)), "val_map25": ev["map25"],
               "val_mean_sens@1fp": ev["mean_sens@1.0fp"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[epoch {epoch}] {rec}", flush=True)
        if rec["val_map25"] > best:
            best = rec["val_map25"]
            torch.save(model.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    vp, tp_ = predict(model, val, device), predict(model, test, device)
    _dump_preds(out / "preds_val.json", val, vp)
    _dump_preds(out / "preds_test.json", test, tp_)
    test_gt = _gt(test)
    test_eval = evaluate_detections(tp_, test_gt, DET_CONCEPTS)
    (out / "eval_test.json").write_text(json.dumps(test_eval, indent=2))

    ops = []
    for c in DET_CONCEPTS:
        s, t, _ = class_tp_arrays(vp, _gt(val), c)
        thr = score_threshold_at_fppi(s, t, len(val), cfg.target_fppi)
        validated, gate = gate_class(cfg, tp_, test_gt, c, test_eval[c])
        ops.append(ClassOperatingPoint(concept=c, box_score_threshold=thr, target_fppi=cfg.target_fppi,
                                       validated=validated, gate={**gate, "test_split": "official"}))
    bundle = write_detector_bundle(out / "det_bundle", model.state_dict(), dcfg, ops, lin)
    print(f"[done] best val mAP@0.25={best:.3f} test mAP@0.25={test_eval['_summary']['map25']:.3f} "
          f"validated={[o.concept for o in ops if o.validated]} research_only={lin['research_only']} -> {bundle}")
    return out


def regate_detector(run_dir: str | Path, exclude: set[str], cfg: DetTrainConfig | None = None) -> dict:
    """Re-score saved test predictions without ``exclude`` (e.g. patient-leaky images), rewrite per-class
    ``validated`` flags in the detector bundle, and save both evaluations for comparison."""
    run = Path(run_dir)
    cfg = cfg or DetTrainConfig.from_yaml(run / "config.yaml")
    recs = {r.file_name: r for r in load_split(cfg.root, "test")}
    preds = json.loads((run / "preds_test.json").read_text())

    def subset(keep):
        ps, gs = [], []
        for p in preds:
            if keep(p["file_name"]):
                ps.append(ImagePreds(np.asarray(p["boxes"], dtype=np.float32).reshape(-1, 4),
                                     np.asarray(p["scores"], dtype=np.float32), p["labels"]))
                r = recs[p["file_name"]]
                gs.append(ImageGT(r.boxes, list(r.labels)))
        return ps, gs, evaluate_detections(ps, gs, DET_CONCEPTS)

    clean_p, clean_g, clean = subset(lambda f: f not in exclude)
    leaky = subset(lambda f: f in exclude)[2] if exclude else None
    (run / "eval_test_leak_reduced.json").write_text(json.dumps(clean, indent=2))
    if leaky is not None:
        (run / "eval_test_leaky_only.json").write_text(json.dumps(leaky, indent=2))
    ops_path = run / "det_bundle" / "operating_points.json"
    ops = json.loads(ops_path.read_text())
    for o in ops:
        o["validated"], gate = gate_class(cfg, clean_p, clean_g, o["concept"], clean[o["concept"]])
        o["gate"] = {**gate, "test_split": "official_minus_known_patient_leak" if exclude else "official",
                     "excluded_images": len(exclude)}
    ops_path.write_text(json.dumps(ops, indent=2))
    return {"leak_reduced": clean, "leaky_only": leaky, "validated": [o["concept"] for o in ops if o["validated"]],
            "gates": {o["concept"]: o["gate"] for o in ops}}

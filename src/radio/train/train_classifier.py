"""Train the multi-label classifier (stage 3 of the training strategy) and dump predictions/features
for calibration, OOD fitting and evaluation.

Run directory layout (runs/<run_name>/):
    config.yaml, model.pt, model_config.json, history.json,
    preds_{val,test}.npz  (logits, y, mask, patient_id, feats),
    feats_train.npz       (in-distribution features for the OOD gate)
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from radio.data.nih import NIHDataset, load_nih
from radio.models.classifier import ClassifierConfig, build_classifier
from radio.train.losses import masked_asl, masked_bce


@dataclass
class DataCfg:
    dataset_ids: list[str] = field(default_factory=lambda: ["nih_cxr14"])
    csv: str = "data/nih/Data_Entry_2017.csv"
    images: str = "data/nih/images"
    split_dir: str | None = "data/nih"
    max_images: int | None = None
    val_frac: float = 0.1
    test_frac: float = 0.2


@dataclass
class ModelCfg:
    encoder: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    img_size: int = 448
    dropout: float = 0.1


@dataclass
class OptimCfg:
    epochs: int = 5
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_encoder: float = 2e-5
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    loss: str = "bce"  # bce | asl
    num_workers: int = 4
    amp: bool = True
    device: str = "auto"
    seed: int = 0
    max_ood_train_feats: int = 5000


@dataclass
class TrainConfig:
    run_name: str = "nih_dinov2s"
    out_dir: str = "runs"
    purpose: str = "research"  # research | product (product enforces the licence gate)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in (overrides or {}).items():
            sect, key = k.split(".", 1) if "." in k else (None, k)
            (raw.setdefault(sect, {}) if sect else raw)[key] = v
        return cls(run_name=raw.get("run_name", cls.run_name), out_dir=raw.get("out_dir", "runs"),
                   purpose=raw.get("purpose", "research"),
                   data=DataCfg(**raw.get("data", {})), model=ModelCfg(**raw.get("model", {})),
                   optim=OptimCfg(**raw.get("optim", {})))


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _mean_auroc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    logits, feats, ys, masks, idx = [], [], [], [], []
    for x, y, m, i in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type != "cpu"):
            lo, fe = model(x)
        logits.append(lo.float().cpu().numpy())
        feats.append(fe.float().cpu().numpy())
        ys.append(y.numpy())
        masks.append(m.numpy())
        idx.append(i.numpy())
    return {"logits": np.concatenate(logits), "feats": np.concatenate(feats), "y": np.concatenate(ys),
            "mask": np.concatenate(masks), "idx": np.concatenate(idx)}


def _log_mlflow(cfg: TrainConfig, lin: dict, history: list[dict], out: Path) -> None:
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve()}")
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.set_tags({"purpose": lin["purpose"], "research_only": str(lin["research_only"]).lower(),
                         "datasets": ",".join(d["dataset_id"] for d in lin["datasets"]),
                         "weights_init": lin["weights_init"]["weights_id"]})
        mlflow.log_params({"encoder": cfg.model.encoder, "img_size": cfg.model.img_size,
                           "epochs": cfg.optim.epochs, "max_images": cfg.data.max_images})
        for h in history:
            mlflow.log_metric("val_mean_auroc", h["val_mean_auroc"], step=h["epoch"])
        mlflow.log_artifacts(str(out), artifact_path="run", )


def train(cfg: TrainConfig) -> Path:
    from radio.data.registry import lineage, weights_id_for

    lin = lineage(cfg.data.dataset_ids, weights_id_for(cfg.model.encoder, cfg.model.pretrained), cfg.purpose)
    _seed(cfg.optim.seed)
    device = pick_device(cfg.optim.device)
    out = Path(cfg.out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "lineage.json").write_text(json.dumps(lin, indent=2))

    splits = load_nih(cfg.data.csv, cfg.data.images, split_dir=cfg.data.split_dir, val_frac=cfg.data.val_frac,
                      test_frac=cfg.data.test_frac, max_images=cfg.data.max_images, seed=cfg.optim.seed)
    labels = splits.labels
    print(f"[data] train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
          f"patients(train)={splits.train.patient_id.nunique()} device={device}")
    assert not (set(splits.train.patient_id) & set(splits.test.patient_id)), "patient leakage train/test"
    assert not (set(splits.train.patient_id) & set(splits.val.patient_id)), "patient leakage train/val"

    mcfg = ClassifierConfig(encoder=cfg.model.encoder, pretrained=cfg.model.pretrained, img_size=cfg.model.img_size,
                            labels=tuple(labels), dropout=cfg.model.dropout)
    model = build_classifier(mcfg).to(device)

    o = cfg.optim
    dl_kw = dict(batch_size=o.batch_size, num_workers=o.num_workers, persistent_workers=o.num_workers > 0)
    tr = DataLoader(NIHDataset(splits.train, labels, mcfg.img_size, augment=True), shuffle=True, drop_last=True,
                    **dl_kw)
    va = DataLoader(NIHDataset(splits.val, labels, mcfg.img_size), shuffle=False, **dl_kw)
    te = DataLoader(NIHDataset(splits.test, labels, mcfg.img_size), shuffle=False, **dl_kw)

    params = [
        {"params": model.encoder.parameters(), "lr": o.lr_encoder},
        {"params": model.head.parameters(), "lr": o.lr_head},
    ]
    opt = torch.optim.AdamW(params, weight_decay=o.weight_decay)
    total = max(1, o.epochs * len(tr))
    warm = max(1, int(o.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )
    loss_fn = masked_asl if o.loss == "asl" else masked_bce
    amp = o.amp and device.type != "cpu"

    history, best = [], -1.0
    step = 0
    for epoch in range(o.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y, m, _ in tr:
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(x)
            loss = loss_fn(logits.float(), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{total} loss {loss.item():.4f}", flush=True)
        pv = predict(model, va, device, amp)
        auroc = _mean_auroc(pv["y"], 1 / (1 + np.exp(-pv["logits"])))
        rec = {"epoch": epoch, "train_loss": run_loss / max(1, len(tr)), "val_mean_auroc": auroc,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[epoch {epoch}] {rec}", flush=True)
        if not np.isnan(auroc) and auroc > best or best < 0:
            best = auroc
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    (out / "model_config.json").write_text(json.dumps(mcfg.to_dict(), indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "config.yaml").write_text(yaml.safe_dump({"run_name": cfg.run_name, "out_dir": cfg.out_dir,
                                                     "purpose": cfg.purpose,
                                                     "data": vars(cfg.data), "model": vars(cfg.model),
                                                     "optim": vars(cfg.optim)}))

    for name, loader, frame in (("val", va, splits.val), ("test", te, splits.test)):
        p = predict(model, loader, device, amp)
        np.savez(out / f"preds_{name}.npz", **p, patient_id=frame["patient_id"].to_numpy()[p["idx"]],
                 image=frame["image"].to_numpy()[p["idx"]], labels=np.array(labels))
    n_ood = min(len(splits.train), o.max_ood_train_feats)
    sub = Subset(NIHDataset(splits.train, labels, mcfg.img_size), list(range(n_ood)))
    ptr = predict(model, DataLoader(sub, shuffle=False, **dl_kw), device, amp)
    np.savez(out / "feats_train.npz", feats=ptr["feats"])
    _log_mlflow(cfg, lin, history, out)
    print(f"[done] best val mean AUROC={best:.4f} research_only={lin['research_only']} -> {out}")
    return out

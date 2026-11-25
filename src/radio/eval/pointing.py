"""Work package 2.6: do classifier activation maps point at radiologist-annotated lesions?

For every ChestX-Det test image and every concept that is both in the classifier's label space and
annotated with boxes in that image, take the activation-map peak (source pixels) and count a hit if it falls
inside any reference box of that concept (pointing game). The chance baseline is the fraction of the image
covered by those boxes (the hit rate of a uniformly random point).

Also reports image-level AUROC of the classifier against ChestX-Det's *radiologist* image labels - an
independent check of the Phase-1 classifier, whose own training labels were NLP-mined.
Images that fingerprint-match NIH images used to train/calibrate the classifier are excluded.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from radio.data.chestx_det import DetRecord, read_gray
from radio.pipeline.bundle import ModelBundle
from radio.preprocess.transforms import prepare


def _union_area_fraction(boxes: np.ndarray, h: int, w: int) -> float:
    m = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return float(m.mean())


@torch.no_grad()
def pointing_game(bundle: ModelBundle, records: list[DetRecord], exclude: set[str] | None = None,
                  device: str = "cpu") -> dict[str, dict]:
    exclude = exclude or set()
    labels = bundle.labels
    model = bundle.model.to(device).eval()
    hits: dict[str, list[float]] = {c: [] for c in labels}
    chance: dict[str, list[float]] = {c: [] for c in labels}
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    used = 0
    for r in records:
        if r.file_name in exclude:
            continue
        used += 1
        img = read_gray(r.path)
        h, w = img.shape
        x, lb = prepare(img, bundle.config.img_size)
        logits, _, maps = model.forward_with_maps(x[None].to(device))
        temps = np.array([t.temperature for t in bundle.thresholds])
        probs.append(1 / (1 + np.exp(-logits[0].float().cpu().numpy() / temps)))
        targets.append(np.array([float(c in r.image_labels) for c in labels]))
        up = F.interpolate(maps.float().cpu(), size=(lb.size, lb.size), mode="bilinear", align_corners=False)[0]
        h_in, w_in = round(h * lb.scale), round(w * lb.scale)
        for j, c in enumerate(labels):
            gb = r.boxes[[k for k, lab in enumerate(r.labels) if lab == c]]
            if len(gb) == 0:
                continue
            crop = up[j, lb.pad_y:lb.pad_y + h_in, lb.pad_x:lb.pad_x + w_in].numpy()
            yy, xx = np.unravel_index(int(crop.argmax()), crop.shape)
            sx, sy = lb.to_source(xx + lb.pad_x + 0.5, yy + lb.pad_y + 0.5)
            hit = any(b[0] <= sx <= b[2] and b[1] <= sy <= b[3] for b in gb)
            hits[c].append(float(hit))
            chance[c].append(_union_area_fraction(gb, h, w))
    P, Y = np.stack(probs), np.stack(targets)
    out: dict[str, dict] = {}
    for j, c in enumerate(labels):
        y = Y[:, j]
        auroc = float(roc_auc_score(y, P[:, j])) if 0 < y.sum() < len(y) else float("nan")
        out[c] = {
            "n_lesion_images": len(hits[c]),
            "pointing_hit_rate": float(np.mean(hits[c])) if hits[c] else float("nan"),
            "chance_hit_rate": float(np.mean(chance[c])) if chance[c] else float("nan"),
            "image_auroc_vs_radiologist": auroc,
            "n_pos_images": int(y.sum()),
        }
    rated = [v for v in out.values() if v["n_lesion_images"] >= 10]
    out["_summary"] = {
        "images_used": used,
        "images_excluded_overlap": len(exclude),
        "mean_hit_rate(n>=10)": float(np.mean([v["pointing_hit_rate"] for v in rated])) if rated else float("nan"),
        "mean_chance(n>=10)": float(np.mean([v["chance_hit_rate"] for v in rated])) if rated else float("nan"),
    }
    return out

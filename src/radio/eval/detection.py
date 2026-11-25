"""Detection metrics per docs/blueprint/08-evaluation-protocol.md section 2.

* mAP (VOC-style all-point interpolated AP per class) at IoU 0.5 and 0.25.
* FROC: lesion-level sensitivity at 0.25 / 0.5 / 1 / 2 mean false positives per image. A prediction is a
  hit if IoU >= ``hit_iou`` (default 0.25, the clinically meaningful threshold for fuzzy CXR boxes) with an
  unmatched reference box of the same class; each reference box is matched at most once.
* Pointing accuracy: fraction of true-positive predictions whose box centre lies inside the reference box.
* Per-image detection: sensitivity/specificity of "any box of class k with score >= t".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FP_RATES = (0.25, 0.5, 1.0, 2.0)


@dataclass
class ImagePreds:
    boxes: np.ndarray  # [M,4]
    scores: np.ndarray  # [M]
    labels: list[str]


@dataclass
class ImageGT:
    boxes: np.ndarray  # [N,4]
    labels: list[str]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _match_class(preds: list[ImagePreds], gts: list[ImageGT], cls: str, iou_thr: float):
    """Greedy score-ordered matching. Returns (scores, is_tp, centre_inside, n_gt)."""
    recs = []
    n_gt = 0
    for img_idx, (p, g) in enumerate(zip(preds, gts, strict=True)):
        gmask = np.array([lab == cls for lab in g.labels], dtype=bool)
        n_gt += int(gmask.sum())
        pmask = np.array([lab == cls for lab in p.labels], dtype=bool)
        for k in np.flatnonzero(pmask):
            recs.append((float(p.scores[k]), img_idx, k))
    recs.sort(key=lambda r: -r[0])
    used: dict[int, set[int]] = {}
    scores, tps, inside = [], [], []
    for s, img_idx, k in recs:
        g = gts[img_idx]
        gidx = np.flatnonzero(np.array([lab == cls for lab in g.labels], dtype=bool))
        hit, centre_in = False, False
        if len(gidx):
            ious = iou_matrix(preds[img_idx].boxes[k : k + 1], g.boxes[gidx])[0]
            taken = used.setdefault(img_idx, set())
            order = np.argsort(-ious)
            for o in order:
                if ious[o] >= iou_thr and gidx[o] not in taken:
                    taken.add(int(gidx[o]))
                    hit = True
                    b = preds[img_idx].boxes[k]
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    gb = g.boxes[gidx[o]]
                    centre_in = bool(gb[0] <= cx <= gb[2] and gb[1] <= cy <= gb[3])
                    break
        scores.append(s)
        tps.append(hit)
        inside.append(centre_in)
    return np.array(scores), np.array(tps, dtype=bool), np.array(inside, dtype=bool), n_gt


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def froc(scores: np.ndarray, tp: np.ndarray, n_gt: int, n_images: int,
         fp_rates=FP_RATES) -> dict[str, float]:
    out = {}
    if n_gt == 0:
        return {f"sens@{r}fp": float("nan") for r in fp_rates}
    order = np.argsort(-scores)
    tp_o = tp[order]
    sens = np.cumsum(tp_o) / n_gt
    fppi = np.cumsum(~tp_o) / max(n_images, 1)
    for r in fp_rates:
        ok = fppi <= r
        out[f"sens@{r}fp"] = float(sens[ok].max()) if ok.any() else 0.0
    return out


def score_threshold_at_fppi(scores: np.ndarray, tp: np.ndarray, n_images: int, target: float) -> float:
    """Lowest score threshold whose mean FP/image stays <= target (for choosing an operating point)."""
    if len(scores) == 0:
        return 1.0
    order = np.argsort(-scores)
    s, t = scores[order], tp[order]
    fppi = np.cumsum(~t) / max(n_images, 1)
    ok = np.flatnonzero(fppi <= target)
    return float(s[ok[-1]]) if len(ok) else float(s[0]) + 1e-6


def evaluate_detections(preds: list[ImagePreds], gts: list[ImageGT], classes: list[str],
                        hit_iou: float = 0.25) -> dict[str, dict]:
    n_img = len(gts)
    res: dict[str, dict] = {}
    for c in classes:
        r: dict[str, float] = {}
        s50, tp50, _, n_gt = _match_class(preds, gts, c, 0.5)
        s25, tp25, inside, _ = _match_class(preds, gts, c, hit_iou)
        r["n_gt_boxes"] = float(n_gt)
        r["n_images_with_class"] = float(sum(1 for g in gts if c in g.labels))
        r["ap50"] = average_precision(tp50[np.argsort(-s50)] if len(s50) else tp50, n_gt)
        r[f"ap{int(hit_iou * 100)}"] = average_precision(tp25[np.argsort(-s25)] if len(s25) else tp25, n_gt)
        r.update(froc(s25, tp25, n_gt, n_img))
        r["pointing_accuracy_tp"] = float(inside[tp25].mean()) if tp25.any() else float("nan")
        res[c] = r
    valid = [v for v in res.values() if v["n_gt_boxes"] > 0]
    res["_summary"] = {
        "n_images": n_img,
        "map50": float(np.nanmean([v["ap50"] for v in valid])) if valid else float("nan"),
        f"map{int(hit_iou * 100)}": float(np.nanmean([v[f"ap{int(hit_iou * 100)}"] for v in valid])) if valid
        else float("nan"),
        "mean_sens@1.0fp": float(np.nanmean([v["sens@1.0fp"] for v in valid])) if valid else float("nan"),
        "hit_iou": hit_iou,
    }
    return res


def class_tp_arrays(preds, gts, cls, hit_iou=0.25):
    """Expose (scores, tp, n_gt) for operating-point selection."""
    s, tp, _, n_gt = _match_class(preds, gts, cls, hit_iou)
    return s, tp, n_gt

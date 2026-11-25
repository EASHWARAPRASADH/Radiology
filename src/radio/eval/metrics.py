"""Classification metrics with patient-level bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from radio.calibrate.conformal import LabelThreshold, decide
from radio.schema.findings import FindingStatus


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(err)


def _point(y: np.ndarray, p: np.ndarray, th: LabelThreshold | None) -> dict[str, float]:
    r: dict[str, float] = {"n": float(len(y)), "n_pos": float(y.sum()), "prevalence": float(y.mean())}
    both = 0 < y.sum() < len(y)
    r["auroc"] = float(roc_auc_score(y, p)) if both else float("nan")
    r["auprc"] = float(average_precision_score(y, p)) if both else float("nan")
    r["ece"] = ece(y, p)
    r["brier"] = float(np.mean((p - y) ** 2))
    if th is not None and th.validated:
        st = np.array([decide(float(v), th).value for v in p])
        pres, unc, absn = st == FindingStatus.present, st == FindingStatus.uncertain, st == FindingStatus.absent
        pos, neg = y > 0.5, y <= 0.5
        r["deferral_rate"] = float(unc.mean())
        # Sensitivity counting deferred cases as "flagged for review" (the clinically relevant miss rate).
        r["sensitivity_present_or_uncertain"] = float((pres | unc)[pos].mean()) if pos.any() else float("nan")
        r["fnr_called_absent"] = float(absn[pos].mean()) if pos.any() else float("nan")
        r["fpr_called_present"] = float(pres[neg].mean()) if neg.any() else float("nan")
        r["specificity_called_absent"] = float(absn[neg].mean()) if neg.any() else float("nan")
        r["ppv_present"] = float(y[pres].mean()) if pres.any() else float("nan")
        tp, fp, fn = float((pres & pos).sum()), float((pres & neg).sum()), float((~pres & pos).sum())
        r["f1_present"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return r


def evaluate(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
    patient_ids: np.ndarray,
    thresholds: list[LabelThreshold] | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    pats = np.unique(patient_ids)
    rows_by_pat = {p: np.flatnonzero(patient_ids == p) for p in pats}
    th_by = {t.label: t for t in thresholds} if thresholds else {}
    out: dict[str, dict] = {}
    for j, lab in enumerate(labels):
        m = mask[:, j] > 0
        point = _point(y[m, j], probs[m, j], th_by.get(lab))
        boots: dict[str, list[float]] = {k: [] for k in ("auroc", "auprc", "sensitivity_present_or_uncertain")}
        for _ in range(n_boot):
            samp = rng.choice(pats, size=len(pats), replace=True)
            idx = np.concatenate([rows_by_pat[p] for p in samp])
            idx = idx[m[idx]]
            yy, pp = y[idx, j], probs[idx, j]
            if 0 < yy.sum() < len(yy):
                boots["auroc"].append(roc_auc_score(yy, pp))
                boots["auprc"].append(average_precision_score(yy, pp))
            th = th_by.get(lab)
            if th is not None and th.validated and yy.sum() > 0:
                st = np.array([decide(float(v), th) for v in pp[yy > 0.5]])
                boots["sensitivity_present_or_uncertain"].append(float(np.isin(st, [FindingStatus.present,
                                                                                    FindingStatus.uncertain]).mean()))
        for k, v in boots.items():
            if len(v) >= 20:
                point[f"{k}_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[lab] = point
    valid = [r["auroc"] for r in out.values() if not np.isnan(r["auroc"])]
    out["_summary"] = {"mean_auroc": float(np.mean(valid)) if valid else float("nan"),
                       "n_images": int(len(y)), "n_patients": int(len(pats)), "n_boot": n_boot}
    return out

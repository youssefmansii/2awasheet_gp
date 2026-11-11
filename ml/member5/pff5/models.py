"""
Thin LightGBM layer: match-grouped cross-validation, out-of-fold predictions,
final models, and the metrics used in the report.

All folds are split BY MATCH (never by row) - frames and events from one match
are highly correlated, so row-level splits would leak and inflate every score.
The match -> fold assignment is a fixed function of the match id, so every task
uses the same folds and out-of-fold predictions can safely be chained
(e.g. predicted participants feeding the challenge-type model).
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

N_FOLDS = 5
BASE_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, max_bin=255,
)
SMALL_PARAMS = dict(BASE_PARAMS, num_leaves=15, min_data_in_leaf=25, learning_rate=0.04, feature_fraction=0.7)


def fold_of(gids, n_folds: int = N_FOLDS, seed: int = 2022) -> np.ndarray:
    """Deterministic fold id per match id, balanced over the matches present."""
    gids = np.asarray(gids)
    uniq = np.unique(gids)
    rng = np.random.RandomState(seed)
    perm = uniq[np.argsort(rng.rand(int(uniq.max()) + 1)[uniq])]  # depends only on the id values
    k = min(n_folds, len(uniq))
    lut = {g: i % max(k, 1) for i, g in enumerate(perm)}
    return np.array([lut[g] for g in gids], np.int64)


def class_weights(y: np.ndarray, n_class: int, power: float = 0.5) -> np.ndarray:
    """Per-row weights ~ (1 / class frequency) ** power.  0 = none, 1 = fully balanced."""
    cnt = np.bincount(y, minlength=n_class).astype(float)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** power
    w = w / (w[y] * 1.0).mean()
    return w[y]


def lgb_cv(X: pd.DataFrame, y, groups, objective="binary", num_class=None, weight=None, params=None,
           rounds=3000, early=100, fit_final=True, verbose=False):
    """Grouped CV. Returns dict(oof, final, best_iter, folds, features)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    folds = fold_of(groups)
    single = len(np.unique(groups)) == 1
    if single:  # debugging on one match only: random 5-fold over rows (scores are optimistic!)
        folds = np.random.RandomState(0).randint(0, N_FOLDS, len(y))
    p = dict(BASE_PARAMS if params is None else params)
    p["objective"] = objective
    if objective == "multiclass":
        p["num_class"] = num_class
        p["metric"] = "multi_logloss"
        oof = np.zeros((len(y), num_class), np.float32)
    else:
        p["metric"] = "binary_logloss"
        oof = np.zeros(len(y), np.float32)
    feats = list(X.columns)
    Xv = X.to_numpy(np.float32)
    iters = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if objective == "multiclass" and len(np.unique(y[tr])) < 2:
            continue
        dtr = lgb.Dataset(Xv[tr], y[tr], weight=None if weight is None else weight[tr], feature_name=feats,
                          free_raw_data=False)
        dva = lgb.Dataset(Xv[va], y[va], weight=None if weight is None else weight[va], reference=dtr)
        bst = lgb.train(p, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(early, verbose=False)] + ([lgb.log_evaluation(200)] if verbose else []))
        oof[va] = bst.predict(Xv[va], num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
    best = int(np.median(iters)) if iters else 200
    final = None
    if fit_final:
        dall = lgb.Dataset(Xv, y, weight=weight, feature_name=feats)
        final = lgb.train(p, dall, max(int(best * 1.1), 20))
    return dict(oof=oof, final=final, best_iter=best, folds=folds, features=feats, single_match=single)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def clf_report(y_true, proba, labels, mask=None) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the majority-class baseline."""
    y_true = np.asarray(y_true)
    if mask is not None:
        y_true, proba = y_true[mask], proba[mask]
    pred = proba.argmax(1) if proba.ndim == 2 else (proba > 0.5).astype(int)
    n = len(y_true)
    out = dict(n=int(n), accuracy=float((pred == y_true).mean()) if n else None)
    per, f1s = {}, []
    cnt = np.bincount(y_true, minlength=len(labels))
    out["majority_baseline_accuracy"] = float(cnt.max() / max(n, 1))
    for i, lab in enumerate(labels):
        tp = int(((pred == i) & (y_true == i)).sum())
        fp = int(((pred == i) & (y_true != i)).sum())
        fn = int(((pred != i) & (y_true == i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[str(lab)] = dict(support=int(cnt[i]), precision=round(pr, 4), recall=round(rc, 4), f1=round(f1, 4))
        if cnt[i] > 0:
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
    out["per_class"] = per
    conf = np.zeros((len(labels), len(labels)), int)
    np.add.at(conf, (y_true, pred), 1)
    out["confusion_rows_true_cols_pred"] = conf.tolist()
    return out


def binary_report(y_true, score, threshold=None) -> dict:
    """AUC, average precision, best-F1 threshold."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    n, npos = len(y_true), int(y_true.sum())
    out = dict(n=int(n), positives=npos, base_rate=float(npos / max(n, 1)))
    if npos == 0 or npos == n:
        return out
    order = np.argsort(-score, kind="stable")
    yt = y_true[order]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, n + 1)
    rec = tp / npos
    out["average_precision"] = float((prec * yt).sum() / npos)
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    out["auc"] = float((ranks[y_true == 1].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(f1.argmax())
    out.update(best_f1=float(f1[k]), best_f1_threshold=float(score[order][k]),
               precision_at_best=float(prec[k]), recall_at_best=float(rec[k]))
    if threshold is not None:
        pred = score >= threshold
        tp_ = int((pred & (y_true == 1)).sum())
        out["at_threshold"] = dict(threshold=float(threshold), precision=tp_ / max(int(pred.sum()), 1),
                                   recall=tp_ / npos)
    return out


def pick_report(keys, score, y, k_list=(1, 2, 3)) -> dict:
    """Top-k accuracy of "which candidate" over groups that contain a correct candidate,
    plus coverage (how often the correct one was among the candidates at all)."""
    d = pd.DataFrame(dict(key=np.asarray(keys), s=np.asarray(score, float), y=np.asarray(y).astype(bool)))
    d["r"] = d.groupby("key")["s"].rank(ascending=False, method="first")
    g = d.groupby("key")
    has = g["y"].any()
    best_rank = d[d["y"]].groupby("key")["r"].min()
    out = dict(groups=int(len(has)), groups_with_answer=int(has.sum()))
    for k in k_list:
        out[f"top{k}_accuracy"] = float((best_rank <= k).sum() / max(int(has.sum()), 1))
    return out


def save_booster(bst, path, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bst.save_model(path)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)


def load_booster(path):
    with open(path + ".json") as f:
        meta = json.load(f)
    return lgb.Booster(model_file=path), meta

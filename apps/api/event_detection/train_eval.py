"""
train_eval.py - Step 3: shot / not-shot, leave-one-match-out on the TRAINING matches
====================================================================================

* builds (and caches) features for every match
* labels: for each PFF SH event the nearest kick within +-15 frames is the positive (answer key)
* rule baseline vs gradient boosting; the boosting threshold is chosen by an INNER leave-one-match-out
  on the 5 training matches only (the held-out match never influences its own threshold)
* scoring: a predicted shot counts as correct if a PFF shot is within +-15 frames (0.5 s), one-to-one

Run (from apps/api):  python -m event_detection.train_eval --match 3816 3828 3829 3830 3831 10514
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .answer_key import load_shots
from .context import CTX_VERSION, context_features
from .features import FEAT_VERSION, FEATURES, build_features
from .shot_detector import ShotModel, non_max_suppression, rule_baseline
from .tracking_reader import cache_dir, load_tracking

TOL = 15
UNSEEN = set(json.load(open(cache_dir().parents[1] / "split_member4.json"))["unseen"])


def match_table(mid: str, final: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(mid) in UNSEEN and not final:
        raise SystemExit(f"{mid} is a locked UNSEEN match (use --final only for the last test)")
    path = cache_dir() / f"features_{mid}_v{FEAT_VERSION}.pkl"
    tr = load_tracking(mid)
    if path.exists():
        F = pd.read_pickle(path)
    else:
        F, _ = build_features(tr)
        F.to_pickle(path)
    cpath = cache_dir() / f"ctx_{mid}_v{CTX_VERSION}.pkl"
    if cpath.exists():
        C = pd.read_pickle(cpath)
    else:
        C = context_features(tr, F); C.to_pickle(cpath)
    F = pd.concat([F.reset_index(drop=True), C.reset_index(drop=True)], axis=1)
    sh = load_shots(mid, tr); sh = sh[sh.is_sh].reset_index(drop=True)
    F = F.copy(); F["y"] = 0; F["shot_row"] = -1
    for r, s in sh.iterrows():
        d = np.abs(F.idx.values - int(s.idx))
        k = int(np.argmin(d))
        if d[k] <= TOL:
            F.loc[k, "y"] = 1; F.loc[k, "shot_row"] = r
    sh["match_id"] = mid
    return F, sh


def score(pred_idx: dict, shots: pd.DataFrame) -> dict:
    tp = 0; n_pred = sum(len(v) for v in pred_idx.values()); offs = []
    for mid, g in shots.groupby("match_id"):
        p = sorted(pred_idx.get(mid, []))
        used = set()
        for s in g.idx.values:
            best = None
            for k, q in enumerate(p):
                if k not in used and abs(q - s) <= TOL and (best is None or abs(q - s) < abs(p[best] - s)):
                    best = k
            if best is not None:
                used.add(best); tp += 1; offs.append(p[best] - s)
    n_true = len(shots)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_true if n_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pred": n_pred, "true": n_true, "tp": tp, "precision": prec, "recall": rec, "f1": f1,
            "median_abs_offset_frames": float(np.median(np.abs(offs))) if offs else np.nan}


def preds_to_idx(F, keep) -> dict:
    out = {}
    for mid, i in zip(F.match_id.values[keep], F.idx.values[keep]):
        out.setdefault(mid, []).append(int(i))
    return out


def best_threshold(F: pd.DataFrame, shots: pd.DataFrame, oof: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 46)
    f1s = [score(preds_to_idx(F, non_max_suppression(F, oof, t)), shots)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def match_folds(mids, k: int, seed: int = 20261110) -> list[list[str]]:
    """Split match ids into k groups (whole matches only).  k >= number of matches = leave-one-match-out."""
    mids = sorted(mids)
    if k >= len(mids):
        return [[m] for m in mids]
    order = np.random.default_rng(seed).permutation(mids).tolist()
    return [sorted(order[i::k]) for i in range(k)]


def lomo(Fall: pd.DataFrame, Sall: pd.DataFrame, nested: bool = True, k: int | None = None) -> tuple[np.ndarray, dict]:
    """Out-of-fold shot scores with whole matches held out.  Up to 8 matches: leave-one-match-out;
    more matches: k (default 6) groups of matches.  thr_of maps every match to the threshold chosen
    without its own fold (only when nested=True)."""
    mids = sorted(Fall.match_id.unique())
    k = k or (len(mids) if len(mids) <= 8 else 6)
    oof = np.zeros(len(Fall)); thr_of = {}
    for fold in match_folds(mids, k):
        te = Fall.match_id.isin(fold).values
        Ftr = Fall[~te]
        oof[te] = ShotModel().fit(Ftr, Ftr.y.values).proba(Fall[te])
        if nested:
            Fi = Ftr.reset_index(drop=True)
            inner = np.zeros(len(Fi))
            for f2 in match_folds(sorted(Fi.match_id.unique()), min(k, 5)):
                kk = Fi.match_id.isin(f2).values
                inner[kk] = ShotModel().fit(Fi[~kk], Fi.y[~kk].values).proba(Fi[kk])
            thr = best_threshold(Fi, Sall[~Sall.match_id.isin(fold)], inner)
            for m in fold:
                thr_of[m] = thr
    return oof, thr_of


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    a = ap.parse_args()
    Fs, Ss = zip(*(match_table(m) for m in a.match))
    Fall = pd.concat(Fs, ignore_index=True); Sall = pd.concat(Ss, ignore_index=True)
    print(f"candidates {len(Fall)} | positives {int(Fall.y.sum())} of {len(Sall)} PFF shots (ceiling)")
    # rule baseline
    rb = rule_baseline(Fall)
    print("RULES        ", {k: round(v, 3) for k, v in score(preds_to_idx(Fall, non_max_suppression(Fall, rb, 0.5)), Sall).items()})
    oof, thr_of = lomo(Fall, Sall)
    keep = np.zeros(len(Fall), bool)
    per = []
    for m, thr in thr_of.items():
        k = (Fall.match_id == m).values
        kk = non_max_suppression(Fall[k].reset_index(drop=True), oof[k], thr)
        keep[np.flatnonzero(k)[kk]] = True
        s = score(preds_to_idx(Fall[k].reset_index(drop=True), kk), Sall[Sall.match_id == m])
        per.append({"match": m, "threshold": round(thr, 2), **{x: round(y, 3) for x, y in s.items()}})
    print("BOOSTING LOMO", {k: round(v, 3) for k, v in score(preds_to_idx(Fall, keep), Sall).items()})
    print(pd.DataFrame(per).to_string(index=False))
    # training-fit score (for the overfitting gap)
    full = ShotModel().fit(Fall, Fall.y.values)
    pin = full.proba(Fall)
    thr_all = float(np.median(list(thr_of.values())))
    print("BOOSTING on its own training data (thr %.2f)" % thr_all,
          {k: round(v, 3) for k, v in score(preds_to_idx(Fall, non_max_suppression(Fall, pin, thr_all)), Sall).items()})
    Fall.assign(p_oof=oof, picked=keep).to_pickle(cache_dir() / "step3_oof.pkl")

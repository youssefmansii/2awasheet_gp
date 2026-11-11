"""
cv48.py - honest evaluation on the 48 TRAINING matches with 6 folds of whole matches
====================================================================================
Every match is predicted by models that never saw it (8 matches held out per fold).
  python -m event_detection.cv48 prep            # gather features + PFF shots (answer key) of the 48 matches
  python -m event_detection.cv48 fold <0..5>     # shot model + threshold (inner 4-fold, training part only)
  python -m event_detection.cv48 fields <0..5>   # fields for the detected shots of that fold
  python -m event_detection.cv48 report          # detection + field accuracy, per match and overall
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from .eval_fields import FIELDS, _norm, oracle_table
from .field_models import GROUP, FieldModels
from .field_table import shot_rows
from .run_member4 import PFF_NAMES, end_to_end, field_report
from .shot_detector import ShotModel, non_max_suppression
from .tracking_reader import cache_dir, load_tracking
from .train_eval import best_threshold, match_folds, match_table, preds_to_idx, score

D = cache_dir()
SPLIT = json.load(open(D.parents[1] / "split_member4.json"))
TRAIN = [str(m) for m in sorted(SPLIT["train"])]
FOLDS = match_folds(TRAIN, 6)


def prep():
    Fs, Ss = zip(*(match_table(m) for m in TRAIN))
    pd.concat(Fs, ignore_index=True).to_pickle(D / "cv48_F.pkl")
    pd.concat(Ss, ignore_index=True).to_pickle(D / "cv48_S.pkl")
    json.dump(FOLDS, open(D / "cv48_folds.json", "w"))
    print("folds", FOLDS)


def fold(i: int):
    F = pd.read_pickle(D / "cv48_F.pkl"); S = pd.read_pickle(D / "cv48_S.pkl")
    te = F.match_id.isin(FOLDS[i]).values
    Ftr = F[~te].reset_index(drop=True); Str = S[~S.match_id.isin(FOLDS[i])]
    p = ShotModel().fit(Ftr, Ftr.y.values).proba(F[te])
    inner = np.zeros(len(Ftr))
    for f2 in match_folds(sorted(Ftr.match_id.unique()), 3):
        k = Ftr.match_id.isin(f2).values
        inner[k] = ShotModel().fit(Ftr[~k], Ftr.y[~k].values).proba(Ftr[k])
    thr = best_threshold(Ftr, Str, inner)
    pickle.dump({"idx": np.flatnonzero(te), "p": p, "thr": thr}, open(D / f"cv48_fold{i}.pkl", "wb"))
    print("fold", i, "threshold", round(thr, 3))


def fields(i: int):
    from .field_models2 import FieldModels2, enrich
    from .fields_cv import load_all
    F = pd.read_pickle(D / "cv48_F.pkl")
    r = pickle.load(open(D / f"cv48_fold{i}.pkl", "rb"))
    others = [m for m in TRAIN if m not in FOLDS[i]]
    fm = FieldModels2().fit(*load_all(others))
    out = {}
    for m in FOLDS[i]:
        sel = (F.match_id.values[r["idx"]] == m)
        Fm = F.iloc[r["idx"][sel]].reset_index(drop=True)
        keep = non_max_suppression(Fm, r["p"][sel], r["thr"])
        D_ = Fm[keep].sort_values("idx")
        if len(D_):
            T, C = enrich(m, Fm, shot_rows(m, Fm, D_.idx.tolist()))
            P = fm.predict(T, C)
            pred = pd.DataFrame({"kick_idx": D_.idx.values})
            for k, v in PFF_NAMES.items():
                pred[v] = P[k].values
        else:
            pred = pd.DataFrame(columns=["kick_idx"] + list(PFF_NAMES.values()))
        out[m] = pred
    pickle.dump(out, open(D / f"cv48_fields{i}.pkl", "wb"))
    print("fields fold", i, {m: len(v) for m, v in out.items()})


def report():
    det, pairs, per = [], [], []
    for i in range(6):
        preds = pickle.load(open(D / f"cv48_fields{i}.pkl", "rb"))
        for m, pred in preds.items():
            d, pp = end_to_end(pred, m)
            det.append(d); pairs.append(pp)
            per.append({"match": m, "fold": i, **{k: round(v, 3) for k, v in d.items()}})
    tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
    P, R = tp / npred, tp / ntrue
    s = {"matches": len(det), "pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / (P + R)}
    per = pd.DataFrame(per); fr = field_report(pd.concat(pairs, ignore_index=True))
    print("DETECTION (6-fold by match, 48 training matches):", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
    print("per-match F1: median %.2f, 10%%-90%% range %.2f-%.2f" % (per.f1.median(), per.f1.quantile(.1), per.f1.quantile(.9)))
    print(fr.round(3).to_string(index=False))
    per.to_csv(D / "cv48_detection.csv", index=False); fr.to_csv(D / "cv48_fields.csv", index=False)
    json.dump(s, open(D / "cv48_summary.json", "w"), indent=1)
    pd.concat(pairs, ignore_index=True).to_pickle(D / "cv48_pairs.pkl")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prep": prep, "report": report}.get(cmd, lambda: None)()
    if cmd == "fold":
        fold(int(sys.argv[2]))
    if cmd == "fields":
        fields(int(sys.argv[2]))

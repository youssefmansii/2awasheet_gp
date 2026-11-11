"""
run_member4.py - the whole Member-4 pipeline: tracking file  ->  PFF-style shot (SH) events
===========================================================================================

  python -m event_detection.run_member4 lomo  --match <training ids>      # honest test inside training
  python -m event_detection.run_member4 train --match <training ids>      # fit + save the final models
  python -m event_detection.run_member4 predict --match 3830              # write predicted shots
  python -m event_detection.run_member4 final --match <unseen ids>        # ONE-TIME test on unseen

Output per match: data/interim/m4/pred_shots_<id>.csv (+ .json), one row per detected shot with the
PFF column names (shooterPlayerId, shotOutcomeType, keeperPlayerId, ...).
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from .eval_fields import FIELDS, _norm, oracle_table
from .field_models import GROUP, FieldModels
from .field_table import shot_rows
from .shot_detector import ShotModel, non_max_suppression
from .tracking_reader import cache_dir, load_tracking
from .train_eval import TOL, best_threshold, lomo, match_table, preds_to_idx, score

PFF_NAMES = {"shooter_id": "shooterPlayerId", "outcome": "shotOutcomeType", "keeper_id": "keeperPlayerId",
             "blocker_id": "blockerPlayerId", "failed_blocker_id": "failedBlockerPlayerId",
             "gl_clearer_id": "glClearerPlayerId", "body_movement": "bodyMovementType", "ball_moving": "ballMoving",
             "shot_height": "shotInitialHeightType", "save_height": "saveHeightType",
             "save_rebound": "saveReboundType", "shot_nature": "shotNatureType"}
MODEL_PATH = cache_dir() / "member4_models.pkl"
SPLIT = json.load(open(cache_dir().parents[1] / "split_member4.json"))


class Pipeline:
    def fit(self, matches):
        Fs, Ss = zip(*(match_table(m) for m in matches))
        F = pd.concat(Fs, ignore_index=True); S = pd.concat(Ss, ignore_index=True)
        self.shot = ShotModel().fit(F, F.y.values)
        oof, _ = lomo(F, S, nested=False, k=min(len(matches), 6))   # threshold from out-of-fold scores (training only)
        self.threshold = best_threshold(F, S, oof)
        self.fields = FieldModels().fit(oracle_table(matches))
        self.train_matches = list(matches)
        return self

    def predict(self, mid: str, final: bool = False) -> pd.DataFrame:
        F, _ = match_table(mid, final=final)
        p = self.shot.proba(F)
        keep = non_max_suppression(F, p, self.threshold)
        D = F[keep].sort_values("idx").reset_index(drop=True)
        if not len(D):
            return pd.DataFrame()
        R = shot_rows(mid, F, D.idx.tolist())
        if hasattr(self.fields, "rank"):            # version 2 field models
            from .field_models2 import enrich
            R, C = enrich(mid, F, R)
            P = self.fields.predict(R, C)
        else:
            P = self.fields.predict(R)
        tr = load_tracking(mid)
        out = pd.DataFrame({"match_id": mid, "frame": tr.frame[D.idx.values], "period": tr.period[D.idx.values],
                            "video_time_s": tr.t[D.idx.values].round(3), "shot_probability": p[keep][np.argsort(F.idx[keep].values)].round(3),
                            "kick_idx": D.idx.values})
        for k, v in PFF_NAMES.items():
            out[v] = P[k].values
        return out


def end_to_end(pred: pd.DataFrame, mid: str, final: bool = False) -> tuple[dict, pd.DataFrame]:
    """Detection P/R/F1 + field accuracy on the shots we found (matched to PFF within +-TOL frames)."""
    _, S = match_table(mid, final=final)
    d = score({mid: pred.kick_idx.tolist()} if len(pred) else {}, S)
    pairs = []
    used = set()
    for r, s in S.iterrows():
        if not len(pred):
            break
        dd = np.abs(pred.kick_idx.values - int(s.idx))
        cand = [k for k in np.argsort(dd) if dd[k] <= TOL and k not in used]
        if cand:
            k = cand[0]; used.add(k)
            row = {"match_id": mid, "pff_row": r}
            for f in FIELDS:
                row["true_" + f] = _norm(s[f]); row["pred_" + f] = _norm(pred[PFF_NAMES[f]].iloc[k])
            pairs.append(row)
    return d, pd.DataFrame(pairs)


def field_report(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FIELDS:
        t, p = pairs["true_" + f].values, pairs["pred_" + f].values
        has = np.array([x is not None for x in t])
        rows.append({"field": f, "n_with_pff_value": int(has.sum()),
                     "accuracy": float((t[has] == p[has]).mean()) if has.any() else np.nan})
    rows.append({"field": "outcome (4 groups)", "n_with_pff_value": len(pairs),
                 "accuracy": float((pairs.true_outcome.map(GROUP) == pairs.pred_outcome.map(GROUP)).mean()) if len(pairs) else np.nan})
    return pd.DataFrame(rows)


def run_lomo(matches):
    det, allpairs, per = [], [], []
    for m in matches:
        others = [x for x in matches if x != m]
        pipe = Pipeline().fit(others)
        pred = pipe.predict(m)
        d, pairs = end_to_end(pred, m)
        per.append({"match": m, "threshold": round(pipe.threshold, 2), **{k: round(v, 3) for k, v in d.items()}})
        det.append(d); allpairs.append(pairs)
        print(per[-1], flush=True)
    tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
    P, R = tp / max(npred, 1), tp / max(ntrue, 1)
    summary = {"pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9)}
    pairs = pd.concat(allpairs, ignore_index=True)
    return summary, pd.DataFrame(per), field_report(pairs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lomo", "lomo1", "lomo_summary", "train", "predict", "final", "trainfit"])
    ap.add_argument("--match", nargs="+", required=True)
    ap.add_argument("--heldout", default=None)
    a = ap.parse_args()
    unseen = set(map(str, SPLIT["unseen"]))
    if a.mode != "final":
        bad = [m for m in a.match if m in unseen]
        assert not bad, f"unseen matches {bad} can only be used with 'final'"
    if a.mode == "lomo":
        s, per, fr = run_lomo(a.match)
        print("END-TO-END (leave-one-match-out):", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
        print(fr.round(3).to_string(index=False))
        per.to_csv(cache_dir() / "lomo_detection.csv", index=False); fr.to_csv(cache_dir() / "lomo_fields.csv", index=False)
        json.dump(s, open(cache_dir() / "lomo_summary.json", "w"), indent=1)
    elif a.mode == "lomo1":   # one fold (keeps every call short); results collected by lomo_summary
        m = a.heldout
        pipe = Pipeline().fit([x for x in a.match if x != m])
        pred = pipe.predict(m)
        d, pairs = end_to_end(pred, m)
        pickle.dump({"d": d, "pairs": pairs, "threshold": pipe.threshold}, open(cache_dir() / f"lomo_fold_{m}.pkl", "wb"))
        print(m, {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}, "thr", round(pipe.threshold, 2))
    elif a.mode == "lomo_summary":
        det, pairs, per = [], [], []
        for m in a.match:
            r = pickle.load(open(cache_dir() / f"lomo_fold_{m}.pkl", "rb"))
            det.append(r["d"]); pairs.append(r["pairs"])
            per.append({"match": m, "threshold": round(r["threshold"], 2), **{k: round(v, 3) for k, v in r["d"].items()}})
        tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
        P, R = tp / max(npred, 1), tp / max(ntrue, 1)
        s = {"pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9)}
        fr = field_report(pd.concat(pairs, ignore_index=True))
        print(pd.DataFrame(per).to_string(index=False))
        print("END-TO-END (leave-one-match-out):", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
        print(fr.round(3).to_string(index=False))
        pd.DataFrame(per).to_csv(cache_dir() / "lomo_detection.csv", index=False); fr.to_csv(cache_dir() / "lomo_fields.csv", index=False)
        json.dump(s, open(cache_dir() / "lomo_summary.json", "w"), indent=1)
    elif a.mode == "train":
        pipe = Pipeline().fit(a.match)
        pickle.dump(pipe, open(MODEL_PATH, "wb"))
        print("saved", MODEL_PATH, "threshold", round(pipe.threshold, 3), "trained on", a.match)
    elif a.mode == "trainfit":   # score of the final model on its OWN training matches (overfitting gap)
        pipe = pickle.load(open(MODEL_PATH, "rb"))
        det, pairs = [], []
        for m in a.match:
            d, pp = end_to_end(pipe.predict(m), m); det.append(d); pairs.append(pp)
        tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
        P, R = tp / max(npred, 1), tp / max(ntrue, 1)
        s = {"pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9)}
        fr = field_report(pd.concat(pairs, ignore_index=True))
        print("TRAINING-DATA FIT:", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
        print(fr.round(3).to_string(index=False))
        json.dump(s, open(cache_dir() / "trainfit_summary.json", "w"), indent=1); fr.to_csv(cache_dir() / "trainfit_fields.csv", index=False)
    else:
        pipe = pickle.load(open(MODEL_PATH, "rb"))
        final = a.mode == "final"
        det, pairs = [], []
        for m in a.match:
            pred = pipe.predict(m, final=final)
            pred.to_csv(cache_dir() / f"pred_shots_{m}.csv", index=False)
            pred.to_json(cache_dir() / f"pred_shots_{m}.json", orient="records", indent=1)
            d, pp = end_to_end(pred, m, final=final); det.append(d); pairs.append(pp)
            print(m, {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}, flush=True)
        if final:
            tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
            P, R = tp / max(npred, 1), tp / max(ntrue, 1)
            s = {"pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9)}
            fr = field_report(pd.concat(pairs, ignore_index=True))
            print("UNSEEN TEST:", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
            print(fr.round(3).to_string(index=False))
            json.dump(s, open(cache_dir() / "unseen_summary.json", "w"), indent=1); fr.to_csv(cache_dir() / "unseen_fields.csv", index=False)

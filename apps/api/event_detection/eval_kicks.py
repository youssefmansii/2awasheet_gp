"""
eval_kicks.py - Step 2 check: how many PFF shots get a kick candidate nearby?  (uses the answer key)
Run (from apps/api):  python -m event_detection.eval_kicks --match 3816 3828 3829 3830 3831 10514
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .answer_key import load_shots
from .ball_physics import BallParams, clean_ball, kick_candidates
from .tracking_reader import cache_dir, load_tracking

UNSEEN = set(json.load(open(cache_dir().parents[1] / "split_member4.json"))["unseen"])


def run(matches, p: BallParams, tol_frames=(5, 15), quiet=False):
    rows, ncand, minutes = [], 0, 0.0
    for mid in matches:
        assert int(mid) not in UNSEEN, "unseen match!"
        tr = load_tracking(mid)
        ball = clean_ball(tr, p)
        c = kick_candidates(tr, ball, p)
        ncand += len(c); minutes += tr.n_frames / tr.fps / 60
        sh = load_shots(mid, tr); sh = sh[sh.is_sh]
        ci = c.idx.values
        for _, r in sh.iterrows():
            d = ci - int(r.idx)
            k = int(np.argmin(np.abs(d))) if len(d) else None
            rows.append({"match_id": mid, "clock": r.game_clock, "outcome": r.outcome, "body": r.body_type,
                         "setpiece": r.setpiece_type,
                         "offset": int(d[k]) if k is not None else None,
                         "kind": c.kind.iloc[k] if k is not None else None,
                         "dv": float(c.dv.iloc[k]) if k is not None else np.nan,
                         "speed_after": float(c.speed_after.iloc[k]) if k is not None else np.nan})
    R = pd.DataFrame(rows)
    out = {"candidates": ncand, "per_minute": ncand / minutes, "shots": len(R)}
    for tol in tol_frames:
        out[f"recall_pm{tol}"] = float((R.offset.abs() <= tol).mean())
    if not quiet:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
    return out, R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    a = ap.parse_args()
    out, R = run(a.match, BallParams())
    R.to_csv(cache_dir() / "step2_kick_recall.csv", index=False)
    miss = R[R.offset.abs() > 15]
    print("missed (no candidate within +-15 frames):", len(miss))
    print(miss.to_string(index=False))
    print("offset distribution of found:", R[R.offset.abs() <= 15].offset.describe().round(2).to_dict())

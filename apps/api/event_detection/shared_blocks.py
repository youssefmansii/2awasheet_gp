"""
shared_blocks.py - Member 4's shared part: WHO blocked a pass / cross / clearance (PFF blockerPlayerId)
=======================================================================================================
Member 2 detects the pass / cross and Member 3 the clearance.  Given such an event (its moment and the team
on the ball), this module names the blocker from tracking with the same candidate-ranking model as for shots.
Evaluated here at PFF's event moments (answer key used for the moment, the team and the scoring only).
  python -m event_detection.shared_blocks build <shard>/<n>      python -m event_detection.shared_blocks eval
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from .answer_key import _id, load_events, nearest_idx
from .ball_physics import clean_ball
from .features import FEAT_VERSION
from .player_rank import CAND_FEATS, _clf, candidates
from .tracking_reader import cache_dir, load_tracking

D = cache_dir(); folds = json.load(open(D / "cv48_folds.json")); TRAIN = sum(folds, [])
OUTC = {"PA": "passOutcomeType", "CR": "crossOutcomeType", "CL": "clearanceOutcomeType"}

if sys.argv[1] == "build":
    t0 = time.time(); shard, n = map(int, sys.argv[2].split("/")); built = 0
    for m in TRAIN[shard::n]:
        p = D / f"cand_blocks_{m}.pkl"
        if p.exists(): continue
        if time.time() - t0 > 140: break
        tr = load_tracking(m); ci = pd.read_pickle(D / f"features_{m}_v{FEAT_VERSION}.pkl").idx.values
        rows = []
        for e in load_events(m):
            pe, ge = e.get("possessionEvents") or {}, e.get("gameEvents") or {}
            ty = pe.get("possessionEventType")
            if ty in OUTC and pe.get(OUTC[ty]) == "B" and pe.get("blockerPlayerId") is not None and e.get("eventTime") is not None:
                i = nearest_idx(tr, e["eventTime"], ge.get("period"))
                if abs(tr.t[i] - e["eventTime"]) > 1: continue
                d = np.abs(ci - i); k = int(np.argmin(d))
                rows.append({"type": ty, "kick_idx": int(ci[k]) if d[k] <= 15 else i, "blocker": _id(pe["blockerPlayerId"]),
                             "side": "home" if ge.get("homeTeam") else "away"})
        E = pd.DataFrame(rows).drop_duplicates("kick_idx")
        C = candidates(tr, clean_ball(tr), ci, E.kick_idx.tolist(), att_sides=E.side.tolist())
        C = C.merge(E, on="kick_idx"); C["y"] = (C.player_id == C.blocker).astype(int)
        C.to_pickle(p); built += 1
    print("built", built)
else:
    C = pd.concat([pd.read_pickle(D / f"cand_blocks_{m}.pkl") for m in TRAIN], ignore_index=True)
    C = C[(~C.is_att) & (C.is_gk < 0.5)].reset_index(drop=True)
    X = C[CAND_FEATS].astype(float).values; p = np.zeros(len(C))
    for f in folds:
        te = C.match_id.isin(f).values
        p[te] = _clf().fit(X[~te], C.y.values[~te]).predict_proba(X[te])[:, 1]
    C["p"] = p; g = C.groupby(["match_id", "kick_idx"], sort=False)
    top = C.loc[g.p.idxmax()]; near = C.loc[g.dpath.idxmin().dropna()]
    for ty in ["PA", "CR", "CL", None]:
        a = top if ty is None else top[top.type == ty]; b = near if ty is None else near[near.type == ty]
        print(f"{ty or 'ALL':4s} blocked events {len(a):4d} | nearest-on-path rule {b.y.mean():.1%} | ranking model {a.y.mean():.1%}")

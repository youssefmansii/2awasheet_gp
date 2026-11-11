"""
context.py - trajectory CONTEXT around every kick (tracking only), added to the kick features
==============================================================================================
Published work on event detection (sequence models; FOOTPASS / TAAD+DST, ELASTIC) shows that most false
alarms disappear when the model sees what happens before and after the action, not one instant.  We give
the tree models that view as samples of the trajectory at fixed time offsets around the kick, in the
frame of the nearest goal (goal at x = +L):
  ball: distance to goal line, |y|, height z, speed          at -0.5 ... +1.2 s  (8 offsets)
  nearest attacker / nearest defender / defending keeper distance to the ball   (5 offsets)
  share of frames with a tracked ball before and after the kick
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ball_physics import clean_ball
from .tracking_reader import Tracking

CTX_VERSION = "1"
B_OFF = [-15, -9, -4, 4, 9, 15, 24, 36]
P_OFF = [-9, -4, 0, 6, 15]


def context_features(tr: Tracking, F: pd.DataFrame, ball: dict | None = None) -> pd.DataFrame:
    ball = ball or clean_ball(tr)
    pos, sp, ok = ball["pos"], ball["speed2d"], ball["ok"]
    T = tr.n_frames
    L = float(tr.meta.get("pitch_length") or 105.0) / 2
    idx = F.idx.values.astype(int)
    sg = np.where(F.bx.values >= 0, 1.0, -1.0)
    per = tr.period[np.clip(idx, 0, T - 1)]
    side_home = tr.players.side.values == "home"
    is_gk = (tr.players.position == "GK").values
    home_sign = np.array([tr.attack_sign("home", int(p)) for p in per])
    att_home = home_sign == sg                       # attacking team is home?
    att = np.where(att_home[:, None], side_home[None, :], ~side_home[None, :])
    out = {}
    for o in B_OFF:
        j = np.clip(idx + o, 0, T - 1)
        same = tr.period[j] == per
        x = np.where(same, sg * pos[j, 0], np.nan); y = np.where(same, pos[j, 1], np.nan)
        out[f"c_bdx_{o}"] = L - x; out[f"c_bay_{o}"] = np.abs(y)
        out[f"c_bz_{o}"] = np.where(same, pos[j, 2], np.nan); out[f"c_bs_{o}"] = np.where(same, sp[j], np.nan)
    for o in P_OFF:
        j = np.clip(idx + o, 0, T - 1)
        d = np.hypot(tr.px[j] - pos[j, 0:1], tr.py[j] - pos[j, 1:2])
        with np.errstate(all="ignore"):
            out[f"c_datt_{o}"] = np.nanmin(np.where(att, d, np.nan), axis=1)
            out[f"c_ddef_{o}"] = np.nanmin(np.where(~att & ~is_gk[None, :], d, np.nan), axis=1)
            out[f"c_dgk_{o}"] = np.nanmin(np.where(~att & is_gk[None, :], d, np.nan), axis=1)
    c = np.concatenate([[0], np.cumsum(ok)])
    out["c_ok_before"] = (c[np.clip(idx, 0, T)] - c[np.clip(idx - 15, 0, T)]) / 15.0
    out["c_ok_after"] = (c[np.clip(idx + 30, 0, T)] - c[np.clip(idx, 0, T)]) / 30.0
    return pd.DataFrame(out, index=F.index)


CTX_FEATURES = ([f"c_{k}_{o}" for o in B_OFF for k in ("bdx", "bay", "bz", "bs")]
                + [f"c_{k}_{o}" for o in P_OFF for k in ("datt", "ddef", "dgk")] + ["c_ok_before", "c_ok_after"])

"""
field_table.py - one row per shot with all field features (tracking only) + PFF truth (answer key)
=================================================================================================
Used to train / score the field models.  `kick_idx` comes either from PFF's shot moments ("oracle", to
score the fields on their own) or from our detector (end-to-end).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ball_physics import clean_ball
from .shot_fields import ShotContext, rule_outcome
from .tracking_reader import load_tracking

TRUTH_COLS = ["shooter_id", "outcome", "keeper_id", "blocker_id", "failed_blocker_id", "failed_blocker2_id",
              "gl_clearer_id", "body_movement", "ball_moving", "shot_height", "save_height", "save_rebound",
              "shot_nature", "body_type", "setpiece_type"]


def shot_rows(mid: str, F: pd.DataFrame, kick_idx: list[int]) -> pd.DataFrame:
    tr = load_tracking(mid)
    ball = clean_ball(tr)
    ctx = ShotContext(tr, ball, F)
    rows = []
    for i in kick_idx:
        i = int(i)
        sg, att = ctx.frame_of(i)
        s, s_d, s_k = ctx.shooter(i, sg, att)
        f = ctx.outcome_features(i)
        f.update(ctx.group_b(i, s, sg))
        f.update({"match_id": mid, "kick_idx": i, "sign": sg, "att_side": att,
                  "pred_shooter": ctx.pid[s] if s is not None else None, "shooter_ball_d": s_d,
                  "rule_outcome": rule_outcome(f)})
        gk = ctx.defending_gk(i, sg, att)
        f["pred_keeper_any"] = ctx.pid[gk] if gk is not None else None
        ci = ctx.c.idx.values
        nxt = ci[(ci > i + 2) & (ci < i + int(3 * tr.fps))]
        t1 = int(nxt[0]) if nxt.size else None
        dfn = "away" if att == "home" else "home"
        dmask = (ctx.side == dfn) & ~ctx.is_gk & ctx.on_pitch(i)

        def nearest_def(a, b, exclude=None):
            X, Y, _ = ctx.bxyz(a, max(a + 1, b), sg); PX, PY = ctx.players_xy(a, max(a + 1, b), sg)
            m = dmask.copy()
            if exclude is not None:
                m[exclude] = False
            d = np.where(m[None, :], np.hypot(PX - X[:, None], PY - Y[:, None]), np.nan)
            if not np.isfinite(d).any():
                return None, np.nan
            best = np.nanmin(d, axis=0); k = int(np.nanargmin(best))
            return k, float(best[k])
        # blocker: defender closest to the ball's path from the kick until just after the first touch
        b_end = (t1 + 4) if t1 is not None else i + int(tr.fps)
        b, bd = nearest_def(i + 2, b_end)
        f["pred_blocker_any"] = ctx.pid[b] if b is not None else None
        f["blocker_d"] = bd
        # failed blocker candidate: closest defender to the path in the first 0.5 s (blocker excluded
        # only if the ball really changed direction near him)
        fb, fbd = nearest_def(i + 2, i + int(0.5 * tr.fps))
        f["fb1"] = ctx.pid[fb] if fb is not None else None; f["fb1_d"] = fbd
        fb2, fbd2 = nearest_def(i + 2, i + int(0.5 * tr.fps), exclude=fb)
        f["fb2"] = ctx.pid[fb2] if fb2 is not None else None; f["fb2_d"] = fbd2
        f["fb_is_blocker"] = float(fb is not None and fb == b)
        # ball height at the first touch (save height) and where the ball is 0.7 s later (rebound zone)
        if t1 is not None:
            zz = ball["pos"][max(0, t1 - 2):t1 + 3, 2]
            f["t1_z_med"] = float(np.nanmedian(zz)) if np.isfinite(zz).any() else np.nan
            j = min(tr.n_frames - 1, t1 + int(0.7 * tr.fps))
            okj = np.flatnonzero(ball["ok"][t1 + 1:j + 1])
            if okj.size:
                p = ball["pos"][t1 + 1 + okj[-1]]
                f["reb_x"], f["reb_y"], f["reb_z"] = sg * p[0], sg * p[1], p[2]
            else:
                f["reb_x"] = f["reb_y"] = f["reb_z"] = np.nan
        else:
            f["t1_z_med"] = f["reb_x"] = f["reb_y"] = f["reb_z"] = np.nan
        rows.append(f)
    return pd.DataFrame(rows)

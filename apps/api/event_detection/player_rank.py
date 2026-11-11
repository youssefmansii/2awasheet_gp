"""
player_rank.py - learned "who did it?" models (tracking only)
=============================================================
The first version picked the nearest player.  That fails when the tracking is rough (shooter off-screen,
crowded box).  Here every player on the pitch becomes a CANDIDATE with features describing how he relates
to the ball around the kick and along its path; a classifier scores each candidate and the best one wins
(a learning-to-rank setup, the same idea ELASTIC uses with hand-set weights, but learned from data).
  shooter         candidates = attacking team      label = PFF shooter
  blocker         candidates = defending outfield  label = PFF blocker        (trained on blocked shots)
  failed blocker  candidates = defending outfield  label = PFF failed blocker 1/2 (written if score > thr)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .tracking_reader import Tracking

OFFS = [-12, -8, -4, -2, 0, 2, 4]
CAND_FEATS = ([f"d_{o}" for o in OFFS] + ["dmin_kick", "rank_kick", "approach", "vis", "is_gk", "dpath", "dpath_t", "rank_path",
              "d_t1", "perp_line", "proj_line", "d_kickpoint", "between", "kick_dist", "t1_dt", "ball_speed", "ball_z",
              "gap_to_best", "p_dist_goal"])


def candidates(tr: Tracking, ball: dict, cand_idx: np.ndarray, kicks, att_sides=None) -> pd.DataFrame:
    """att_sides: optional list ('home'/'away') giving the team on the ball for each kick (used for passes,
    where the acting team is not the team attacking the nearest goal)."""
    pos, sp = ball["pos"], ball["speed2d"]
    T = tr.n_frames; fps = tr.fps
    L = float(tr.meta.get("pitch_length") or 105.0) / 2
    side = tr.players.side.values; is_gk = (tr.players.position == "GK").values; pid = tr.players.player_id.values
    okf = np.flatnonzero(ball["ok"])
    rows = []
    for q, i in enumerate(kicks):
        i = int(i)
        j0 = i if np.isfinite(pos[i, 0]) else int(okf[np.clip(np.searchsorted(okf, i), 0, len(okf) - 1)])
        k0 = pos[j0]
        sg = 1 if k0[0] >= 0 else -1
        att = "home" if tr.attack_sign("home", int(tr.period[i])) == sg else "away"
        if att_sides is not None:
            att = att_sides[q]; sg = tr.attack_sign(att, int(tr.period[i]))
        onp = np.isfinite(tr.px[max(0, i - 60):min(T, i + 60)]).any(axis=0)
        P = np.flatnonzero(onp)
        D = {}
        for o in OFFS:
            j = int(np.clip(i + o, 0, T - 1))
            D[o] = np.hypot(tr.px[j, P] - pos[j, 0], tr.py[j, P] - pos[j, 1])
        a, b = max(0, i - 8), min(T, i + 4)
        with np.errstate(all="ignore"):
            dk = np.nanmin(np.hypot(tr.px[a:b][:, P] - pos[a:b, 0:1], tr.py[a:b][:, P] - pos[a:b, 1:2]), axis=0)
        nxt = cand_idx[(cand_idx > i + 2) & (cand_idx < i + 90)]
        t1 = int(nxt[0]) if nxt.size else min(T - 1, i + 30)
        a2, b2 = min(T - 1, i + 2), min(T, t1 + 5)
        dp = np.hypot(tr.px[a2:b2][:, P] - pos[a2:b2, 0:1], tr.py[a2:b2][:, P] - pos[a2:b2, 1:2])
        with np.errstate(all="ignore"):
            dpath = np.nanmin(dp, axis=0)
            dpath_t = np.where(np.isfinite(dpath), np.nanargmin(np.where(np.isfinite(dp), dp, np.inf), axis=0), np.nan) / fps
        d_t1 = np.hypot(tr.px[t1, P] - pos[t1, 0], tr.py[t1, P] - pos[t1, 1])
        gx, gy = sg * L - k0[0], -k0[1]; gl = max(np.hypot(gx, gy), 1e-6)
        rx, ry = tr.px[i, P] - k0[0], tr.py[i, P] - k0[1]
        proj = (rx * gx + ry * gy) / gl ** 2
        perp = np.abs(rx * gy - ry * gx) / gl
        is_att = side[P] == att
        for grp, key in ((is_att, dk), (~is_att, dpath)):
            pass
        rk = np.full(len(P), np.nan); rp = np.full(len(P), np.nan)
        for mask, src, dst in ((is_att, dk, rk), (~is_att & ~is_gk[P], dpath, rp)):
            v = np.where(mask & np.isfinite(src), src, np.inf)
            dst[mask] = np.argsort(np.argsort(v))[mask]
        best_att = np.nanmin(np.where(is_att, dk, np.nan)) if is_att.any() and np.isfinite(dk[is_att]).any() else np.nan
        best_def = np.nanmin(np.where(~is_att & ~is_gk[P], dpath, np.nan)) if np.isfinite(dpath[~is_att & ~is_gk[P]]).any() else np.nan
        for n, s in enumerate(P):
            r = {"kick_idx": i, "slot": int(s), "player_id": pid[s], "is_att": bool(is_att[n]), "is_gk": float(is_gk[s]),
                 "dmin_kick": dk[n], "rank_kick": rk[n], "approach": (D[-8][n] - D[0][n]) * fps / 8.0,
                 "vis": float(tr.pvis[i, s]), "dpath": dpath[n], "dpath_t": dpath_t[n], "rank_path": rp[n], "d_t1": d_t1[n],
                 "perp_line": perp[n], "proj_line": proj[n], "d_kickpoint": float(np.hypot(rx[n], ry[n])),
                 "between": float(0 < proj[n] < 1 and perp[n] < 2.0), "kick_dist": gl, "t1_dt": (t1 - i) / fps,
                 "ball_speed": float(np.nanmean(sp[i + 2:i + 7])) if np.isfinite(sp[i + 2:i + 7]).any() else np.nan,
                 "ball_z": float(k0[2]), "gap_to_best": (dk[n] - best_att) if is_att[n] else (dpath[n] - best_def),
                 "p_dist_goal": float(np.hypot(sg * L - tr.px[i, s], tr.py[i, s]))}
            for o in OFFS:
                r[f"d_{o}"] = D[o][n]
            rows.append(r)
    C = pd.DataFrame(rows)
    C["match_id"] = tr.match_id
    return C


def _clf(seed=20261110):
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=seed)


class PlayerRankers:
    def fit(self, C: pd.DataFrame):
        """C: candidate rows of the training shots with label columns y_shooter, y_blocker, y_fb, has_blocker."""
        X = C[CAND_FEATS].astype(float).values
        a = C.is_att.values
        self.shooter = _clf().fit(X[a], C.y_shooter.values[a])
        d = (~a) & (C.is_gk.values < 0.5)
        hb = d & C.has_blocker.values
        self.blocker = _clf().fit(X[hb], C.y_blocker.values[hb])
        self.fb = _clf().fit(X[d], C.y_fb.values[d])
        return self

    def score(self, C: pd.DataFrame) -> pd.DataFrame:
        X = C[CAND_FEATS].astype(float).values
        a = C.is_att.values; d = (~a) & (C.is_gk.values < 0.5)
        out = C[["match_id", "kick_idx", "player_id"]].copy()
        out["p_shooter"] = np.where(a, self.shooter.predict_proba(X)[:, 1], -1.0)
        out["p_blocker"] = np.where(d, self.blocker.predict_proba(X)[:, 1], -1.0)
        out["p_fb"] = np.where(d, self.fb.predict_proba(X)[:, 1], -1.0)
        return out

    @staticmethod
    def pick(S: pd.DataFrame) -> pd.DataFrame:
        """One row per shot: best shooter, best blocker, best + second failed blocker (with their scores)."""
        rows = []
        for (m, k), g in S.groupby(["match_id", "kick_idx"], sort=False):
            sh = g.loc[g.p_shooter.idxmax()]; bl = g.loc[g.p_blocker.idxmax()]
            f = g.sort_values("p_fb", ascending=False)
            rows.append({"match_id": m, "kick_idx": k, "rk_shooter": sh.player_id if sh.p_shooter >= 0 else None,
                         "rk_blocker": bl.player_id if bl.p_blocker >= 0 else None, "rk_blocker_p": bl.p_blocker,
                         "rk_fb1": f.player_id.iloc[0], "rk_fb1_p": f.p_fb.iloc[0],
                         "rk_fb2": f.player_id.iloc[1] if len(f) > 1 else None, "rk_fb2_p": f.p_fb.iloc[1] if len(f) > 1 else -1})
        return pd.DataFrame(rows)


def label(C: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Attach PFF truth (answer key) to candidate rows.  T: oracle table with true_* columns."""
    t = T.set_index(["match_id", "kick_idx"])
    t = t[~t.index.duplicated()]
    key = pd.MultiIndex.from_arrays([C.match_id, C.kick_idx])
    g = lambda c: t[c].reindex(key).values
    pid = C.player_id.values
    C = C.copy()
    C["y_shooter"] = (pid == g("true_shooter_id")).astype(int)
    C["y_blocker"] = (pid == g("true_blocker_id")).astype(int)
    C["has_blocker"] = pd.notna(g("true_blocker_id"))
    C["y_fb"] = ((pid == g("true_failed_blocker_id")) | (pid == g("true_failed_blocker2_id"))).astype(int)
    return C

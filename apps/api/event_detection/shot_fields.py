"""
shot_fields.py - fill the Member-4 fields of a detected shot, from TRACKING ONLY
================================================================================

Given the kick frame i of a shot:
  shooter        attacking player closest to the ball around the kick
  outcome_feats  what the ball does next: first touch (by keeper / defender / attacker), crossing of the
                 goal line (inside or outside the goal frame, height), how the game restarts afterwards
                 (kick-off = goal, goal kick = off target, corner = touched by the defence) ...
  rule_outcome   football rules on those features ->  G / O / S / F / B / C / L
  keeper         defending goalkeeper on the pitch (filled for S, F, G, like PFF)
  blocker        defending outfield player at the block (B / C);  gl_clearer for L
  failed blocker defender who came close to the ball's path but did not touch it
  group B        body movement, ball moving, shot height, save height, rebound zone, power/placement
All coordinates are turned so the target goal is at x = +L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tracking_reader import Tracking

GOAL_HALF, BAR = 3.66, 2.44
G = 9.81


class ShotContext:
    """Everything about the match needed to describe shots (built once per match)."""

    def __init__(self, tr: Tracking, ball: dict, cands: pd.DataFrame):
        self.tr, self.ball, self.c = tr, ball, cands.reset_index(drop=True)
        self.L = float(tr.meta.get("pitch_length") or 105.0) / 2
        self.W2 = float(tr.meta.get("pitch_width") or 68.0) / 2
        self.fps = tr.fps
        self.side = tr.players.side.values
        self.is_gk = (tr.players.position == "GK").values
        self.pid = tr.players.player_id.values
        sp = ball["speed2d"]
        static = ball["ok"] & (sp < 0.6)
        # runs of a still ball lasting >= 1 s (restarts: kick-off, goal kick, corner, free kick, throw-in)
        d = np.diff(np.concatenate([[0], static.astype(np.int8), [0]]))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        keep = (en - st) >= int(self.fps)
        self.still = list(zip(st[keep], en[keep]))
        self.still_start = np.array([a for a, _ in self.still]) if self.still else np.zeros(0, int)
        # kick-off FORMATION (works even when the ball is not tracked, e.g. after replays):
        # nearly every player in his own half and somebody standing in the centre circle
        home_sign = np.array([tr.attack_sign("home", int(p)) for p in tr.period])
        sgn = np.where(self.side[None, :] == "home", home_sign[:, None], -home_sign[:, None])
        fin = np.isfinite(tr.px)
        own = (tr.px * sgn <= 1.0) & fin
        tot = fin.sum(axis=1)
        centre = (np.hypot(tr.px, tr.py) < 4.0).any(axis=1)
        self.kickoff_form = (tot >= 18) & (own.sum(axis=1) >= tot - 1) & centre

    # ---- helpers ---------------------------------------------------------------------------
    def frame_of(self, i: int):
        tr = self.tr
        bx = self.ball["pos"][i, 0] if np.isfinite(self.ball["pos"][i, 0]) else self._near_ball(i)[0]
        sg = 1 if bx >= 0 else -1
        att = "home" if tr.attack_sign("home", int(tr.period[i])) == sg else "away"
        return sg, att

    def _near_ball(self, i: int):
        ok = np.flatnonzero(self.ball["ok"])
        j = ok[np.clip(np.searchsorted(ok, i), 0, len(ok) - 1)]
        return self.ball["pos"][j]

    def bxyz(self, a: int, b: int, sg: int):
        p = self.ball["pos"][a:b]
        return sg * p[:, 0], sg * p[:, 1], p[:, 2]

    def players_xy(self, a: int, b: int, sg: int):
        return sg * self.tr.px[a:b], sg * self.tr.py[a:b]

    def on_pitch(self, i: int) -> np.ndarray:
        a, b = max(0, i - 60), min(self.tr.n_frames, i + 60)
        return np.isfinite(self.tr.px[a:b]).any(axis=0)

    # ---- shooter -----------------------------------------------------------------------------
    def shooter(self, i: int, sg: int, att: str):
        a, b = max(0, i - 8), min(self.tr.n_frames, i + 4)
        X, Y, _ = self.bxyz(a, b, sg)
        if not np.isfinite(X).any():
            p = self._near_ball(i); X, Y = np.array([sg * p[0]]), np.array([sg * p[1]]); a, b = i, i + 1
        PX, PY = self.players_xy(a, a + len(X), sg)
        d = np.hypot(PX - X[:, None], PY - Y[:, None])
        mask = (self.side == att)
        dm = np.where(mask[None, :], d, np.nan)
        if not np.isfinite(dm).any():
            return None, np.nan, None
        best = np.nanmin(dm, axis=0)
        s = int(np.nanargmin(best))
        k = int(np.nanargmin(dm[:, s]))
        return s, float(best[s]), a + k

    # ---- outcome features ----------------------------------------------------------------------
    def outcome_features(self, i: int) -> dict:
        tr, fps, L = self.tr, self.fps, self.L
        sg, att = self.frame_of(i)
        dfn = "away" if att == "home" else "home"
        T = tr.n_frames
        H = min(T, i + int(3.0 * fps))
        X, Y, Z = self.bxyz(i + 1, H, sg)
        PX, PY = self.players_xy(i + 1, H, sg)
        onp = self.on_pitch(i)
        defm = (self.side == dfn) & onp
        gkm = defm & self.is_gk
        attm = (self.side == att) & onp
        kick = self.bxyz(i, i + 1, sg)
        k0 = np.array([kick[0][0], kick[1][0], kick[2][0]], float)
        if not np.isfinite(k0[0]):
            p = self._near_ball(i); k0 = np.array([sg * p[0], sg * p[1], p[2]])
        vv = self.ball["vel"][i + 2:min(T, i + 7)]
        v0 = (np.nanmean(vv, axis=0) if np.isfinite(vv).any() else np.full(3, np.nan)) * np.array([sg, sg, 1])
        f = {"kick_x": k0[0], "kick_y": k0[1], "kick_dist": float(np.hypot(L - k0[0], k0[1]))}
        # straight + gravity path from the kick to the goal line
        f.update(self._path_to_line(k0, v0, "k_"))
        # distances of players to the ball after the kick
        d = np.hypot(PX - X[:, None], PY - Y[:, None])
        dgk = np.where(gkm[None, :], d, np.nan); ddef = np.where((defm & ~self.is_gk)[None, :], d, np.nan)
        datt = np.where(attm[None, :], d, np.nan)
        # first touch after the kick (next candidate)
        ci = self.c.idx.values
        nxt = ci[(ci > i + 2) & (ci < H)]
        t1 = int(nxt[0]) if nxt.size else None
        f["has_touch"] = float(t1 is not None)
        if t1 is not None:
            r = t1 - (i + 1)
            lo, hi = max(0, r - 3), min(len(X), r + 4)
            g_d = np.nanmin(dgk[lo:hi]) if np.isfinite(dgk[lo:hi]).any() else np.nan
            o_d = np.nanmin(ddef[lo:hi]) if np.isfinite(ddef[lo:hi]).any() else np.nan
            a_d = np.nanmin(datt[lo:hi]) if np.isfinite(datt[lo:hi]).any() else np.nan
            xt = self.bxyz(t1, t1 + 1, sg)
            f.update({"t1_dt": (t1 - i) / fps, "t1_gk_d": g_d, "t1_def_d": o_d, "t1_att_d": a_d,
                      "t1_x": float(xt[0][0]), "t1_absy": float(abs(xt[1][0])), "t1_z": float(xt[2][0]),
                      "t1_dist_from_kick": float(np.hypot(xt[0][0] - k0[0], xt[1][0] - k0[1]))})
            # path of the ball just before the touch, extended to the line (for on/off target)
            vb = self.ball["vel"][max(i + 1, t1 - 4):t1].mean(axis=0) * np.array([sg, sg, 1])
            pb = np.array([xt[0][0], xt[1][0], xt[2][0]])
            f.update(self._path_to_line(pb, vb, "t1_"))
            f["t1_speed_drop"] = float(self.c.speed_before.values[ci == t1][0] - self.c.speed_after.values[ci == t1][0])
            f["t1_turn"] = float(self.c.turn_deg.values[ci == t1][0])
            # did the ball go past the keeper before the touch?
            gx = PX[r, gkm] if gkm.any() else np.array([np.nan])
            f["t1_behind_gk"] = float(np.nanmax(gx) < xt[0][0]) if np.isfinite(gx).any() else np.nan
        else:
            for k in ["t1_dt", "t1_gk_d", "t1_def_d", "t1_att_d", "t1_x", "t1_absy", "t1_z", "t1_dist_from_kick",
                      "t1_y_line", "t1_z_line", "t1_on_target", "t1_speed_drop", "t1_turn", "t1_behind_gk", "t1_t_line"]:
                f[k] = np.nan
        f["min_gk_d"] = float(np.nanmin(dgk)) if np.isfinite(dgk).any() else np.nan
        f["min_def_d"] = float(np.nanmin(ddef[: int(0.8 * fps)])) if np.isfinite(ddef[: int(0.8 * fps)]).any() else np.nan
        # goal line crossing seen in the data
        ok = np.isfinite(X)
        cross = np.flatnonzero(ok & (X >= L - 0.1))
        if cross.size:
            c0 = cross[0]
            f.update({"line_seen": 1.0, "line_y": float(abs(Y[c0])), "line_z": float(Z[c0]),
                      "line_before_touch": float(t1 is None or i + 1 + c0 <= t1 + 2)})
        else:
            f.update({"line_seen": 0.0, "line_y": np.nan, "line_z": np.nan, "line_before_touch": np.nan})
        f["max_x_3s"] = float(np.nanmax(X)) if ok.any() else np.nan
        last = np.flatnonzero(ok)
        f["last_x_3s"] = float(X[last[-1]]) if last.size else np.nan
        f["last_absy_3s"] = float(abs(Y[last[-1]])) if last.size else np.nan
        f["lost_share_3s"] = float(1 - ok.mean()) if len(ok) else 1.0
        f.update(self.restart(i, sg))
        return f

    def _path_to_line(self, p0, v, pre):
        L = self.L
        if np.isfinite(v).all() and v[0] > 0.5 and np.isfinite(p0).all():
            t = (L - p0[0]) / v[0]
            y = p0[1] + v[1] * t
            z = max(0.0, p0[2] + v[2] * t - 0.5 * G * t * t) if p0[2] > 0.3 or v[2] > 1 else 0.0
            on = float(abs(y) <= GOAL_HALF + 0.25 and z <= BAR + 0.15)
            return {pre + "y_line": float(abs(y)), pre + "z_line": float(z), pre + "on_target": on, pre + "t_line": float(t)}
        return {pre + "y_line": np.nan, pre + "z_line": np.nan, pre + "on_target": 0.0, pre + "t_line": np.nan}

    def restart(self, i: int, sg: int) -> dict:
        """How play restarts: kick-off formation within 150 s, and the first still ball (>= 1 s) within 100 s."""
        tr, L, W2 = self.tr, self.L, self.W2
        out = {"rs_dt": np.nan, "rs_kickoff": 0.0, "rs_goalkick": 0.0, "rs_corner": 0.0, "rs_other": 0.0, "rs_none": 1.0}
        a0, a1 = i + int(2 * self.fps), min(tr.n_frames, i + int(150 * self.fps))
        same = tr.period[a0:a1] == tr.period[i] if a0 < a1 else np.zeros(0, bool)
        ko = np.flatnonzero(self.kickoff_form[a0:a1] & same)
        # a kick-off formation must hold for >= 1 s (30 frames within a 2 s window)
        ko_dt = np.nan
        for k in ko:
            if self.kickoff_form[a0 + k:a0 + k + int(2 * self.fps)].sum() >= self.fps:
                ko_dt = (a0 + k - i) / self.fps
                break
        out["ko_dt"] = ko_dt
        # other restarts between the shot and the kick-off are not considered (kick-off wins)
        if np.isfinite(ko_dt):
            out.update({"rs_dt": ko_dt, "rs_kickoff": 1.0, "rs_none": 0.0})
            return out
        if not len(self.still_start):
            return out
        k = np.searchsorted(self.still_start, i + int(1.0 * self.fps))
        if k >= len(self.still_start):
            return out
        a, b = self.still[k]
        if a - i > 100 * self.fps or tr.period[a] != tr.period[i]:
            return out
        x, y = sg * np.nanmedian(self.ball["pos"][a:b, 0]), sg * np.nanmedian(self.ball["pos"][a:b, 1])
        out.update({"rs_dt": (a - i) / self.fps, "rs_none": 0.0})
        if abs(x) < 3 and abs(y) < 3:
            # kick-off check: almost every player in his own half
            m = (a + b) // 2
            per = int(tr.period[m])
            own = 0; tot = 0
            for s in range(len(self.side)):
                px = tr.px[m, s]
                if np.isfinite(px):
                    tot += 1
                    own += int(px * tr.attack_sign(self.side[s], per) <= 1.5)
            out["rs_kickoff"] = float(tot > 0 and own >= tot - 2)
            out["rs_other"] = 1.0 - out["rs_kickoff"]
        elif x > L - 6.5 and abs(y) < 10.5:
            out["rs_goalkick"] = 1.0
        elif x > L - 3 and abs(y) > W2 - 3:
            out["rs_corner"] = 1.0
        else:
            out["rs_other"] = 1.0
        return out

    # ---- players linked to the outcome ---------------------------------------------------------
    def defender_at_touch(self, i: int, t1: int, sg: int, att: str, want_gk: bool):
        dfn = "away" if att == "home" else "home"
        m = (self.side == dfn) & self.on_pitch(i) & (self.is_gk if want_gk else ~self.is_gk)
        lo, hi = max(0, t1 - 3), min(self.tr.n_frames, t1 + 4)
        X, Y, _ = self.bxyz(lo, hi, sg)
        PX, PY = self.players_xy(lo, hi, sg)
        d = np.where(m[None, :], np.hypot(PX - X[:, None], PY - Y[:, None]), np.nan)
        if not np.isfinite(d).any():
            return None, np.nan
        best = np.nanmin(d, axis=0)
        s = int(np.nanargmin(best))
        return s, float(best[s])

    def defending_gk(self, i: int, sg: int, att: str):
        dfn = "away" if att == "home" else "home"
        m = np.flatnonzero((self.side == dfn) & self.is_gk & self.on_pitch(i))
        if not m.size:
            return None
        if m.size == 1:
            return int(m[0])
        x = np.nanmedian(sg * self.tr.px[max(0, i - 30):i + 30, m], axis=0)
        return int(m[np.nanargmax(x)])

    def failed_blockers(self, i: int, sg: int, att: str, t_end: int, exclude: set, d_max: float):
        """Defenders (not keepers) within d_max of the ball in flight between kick and t_end."""
        dfn = "away" if att == "home" else "home"
        m = (self.side == dfn) & ~self.is_gk & self.on_pitch(i)
        lo, hi = i + 2, max(i + 3, min(self.tr.n_frames, t_end))
        X, Y, _ = self.bxyz(lo, hi, sg)
        PX, PY = self.players_xy(lo, hi, sg)
        d = np.where(m[None, :], np.hypot(PX - X[:, None], PY - Y[:, None]), np.nan)
        if not np.isfinite(d).any():
            return []
        best = np.nanmin(d, axis=0)
        cand = [(float(best[s]), s) for s in np.flatnonzero(np.isfinite(best)) if best[s] <= d_max and s not in exclude]
        return [s for _, s in sorted(cand)]

    # ---- group B ---------------------------------------------------------------------------------
    def group_b(self, i: int, s, sg: int) -> dict:
        tr, fps, L = self.tr, self.fps, self.L
        f = {}
        a = max(0, i - int(0.5 * fps))
        if s is not None:
            x, y = sg * tr.px[a:i + 1, s], sg * tr.py[a:i + 1, s]
            ok = np.isfinite(x)
            if ok.sum() >= 3:
                tt = np.flatnonzero(ok) / fps
                vx, vy = np.polyfit(tt, x[ok], 1)[0], np.polyfit(tt, y[ok], 1)[0]
                gx, gy = L - x[ok][-1], -y[ok][-1]
                gd = np.hypot(gx, gy)
                sp = np.hypot(vx, vy)
                f.update({"sh_speed": sp, "sh_v_to_goal": (vx * gx + vy * gy) / max(gd, 1e-6),
                          "sh_v_lat": abs(vx * gy - vy * gx) / max(gd, 1e-6)})
        for k in ("sh_speed", "sh_v_to_goal", "sh_v_lat"):
            f.setdefault(k, np.nan)
        sp = self.ball["speed2d"][a:i - 1]
        f["ball_speed_before"] = float(np.nanmedian(sp)) if np.isfinite(sp).any() else np.nan
        f["ball_speed_before_min"] = float(np.nanmin(sp)) if np.isfinite(sp).any() else np.nan
        e = min(tr.n_frames, i + int(0.5 * fps))
        sp2 = self.ball["speed2d"][i + 1:e]
        f["ball_speed_after_max"] = float(np.nanmax(sp2)) if np.isfinite(sp2).any() else np.nan
        z = self.ball["pos"][i + 1:e, 2]
        f["ball_z_max_05"] = float(np.nanmax(z)) if np.isfinite(z).any() else np.nan
        zk = self.ball["pos"][max(0, i - 2):i + 3, 2]
        f["ball_z_kick"] = float(np.nanmedian(zk)) if np.isfinite(zk).any() else np.nan
        return f


# -------------------------------------------------------------------------------------------------
def rule_outcome(f: dict) -> str:
    """Football rules on the outcome features (reference rules; the learned model may override)."""
    if f.get("line_seen") == 1.0 and f.get("line_before_touch") == 1.0 and f["line_y"] <= GOAL_HALF + 0.3 \
            and (np.isnan(f["line_z"]) or f["line_z"] <= BAR + 0.2):
        return "G"
    if f.get("rs_kickoff") == 1.0:
        return "G"
    if f.get("has_touch") == 1.0 and f["t1_dt"] < 1.5:
        gk, de, at = f["t1_gk_d"], f["t1_def_d"], f["t1_att_d"]
        gk = np.inf if np.isnan(gk) else gk; de = np.inf if np.isnan(de) else de; at = np.inf if np.isnan(at) else at
        if gk <= 2.5 and gk <= de:
            return "S" if f.get("t1_on_target", 0) == 1.0 or f.get("k_on_target", 0) == 1.0 else "F"
        if de <= 2.0 and de < at:
            if f.get("t1_behind_gk") == 1.0 and f.get("t1_x", 0) > 50.0:
                return "L"
            return "B" if f.get("k_on_target", 0) == 1.0 else "C"
    return "O"

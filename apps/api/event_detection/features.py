"""
features.py - describe every kick candidate with numbers taken from TRACKING ONLY
==================================================================================

For each kick (from ball_physics.kick_candidates) we compute, in the frame of the goal NEAREST to the
ball (that goal is always put at x = +L, L = half pitch length).  The attacking team is the team that
attacks that goal (from the metadata).  We do NOT trust "who is nearest the ball" to decide the
direction: step 1 showed the shooter is often not the nearest tracked player (headers, off-screen
players), and a wrong actor would flip the whole picture.
  actor      nearest player to the ball around the kick, is he an attacker?, nearest attacker / defender
  where      distance / angle to the goal, inside the box?
  path       ball speed and direction after the kick, where the straight path would cross the goal
             line (y_cross) and at what height (z_cross, simple gravity curve)
  future     what the ball does in the next 2 s: gets to the goal line?  goalkeeper / defender gets near
             it?  who touches it next and where?
  context    defenders inside the shot triangle, goalkeeper position, team-mates ahead (pass options),
             nearest opponent (pressure), time since the previous kick
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ball_physics import BallParams, clean_ball, kick_candidates
from .tracking_reader import Tracking

FEAT_VERSION = "1.4"
GOAL_HALF = 3.66
G = 9.81


def _in_triangle(px, py, ax, ay, L):
    """Points inside the triangle ball(ax,ay) - post (L,-3.66) - post (L,+3.66)."""
    def cross(x1, y1, x2, y2, x3, y3):
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    d1 = cross(ax, ay, L, -GOAL_HALF, px, py)
    d2 = cross(L, -GOAL_HALF, L, GOAL_HALF, px, py)
    d3 = cross(L, GOAL_HALF, ax, ay, px, py)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def build_features(tr: Tracking, bp: BallParams = BallParams()) -> tuple[pd.DataFrame, dict]:
    ball = clean_ball(tr, bp)
    c = kick_candidates(tr, ball, bp)
    fps = tr.fps
    L = float(tr.meta.get("pitch_length") or 105.0) / 2
    pos, vel, ok = ball["pos"], ball["vel"], ball["ok"]
    PX, PY = tr.px, tr.py
    T, P = PX.shape
    side = tr.players.side.values
    is_gk = (tr.players.position == "GK").values
    W = int(round(2.0 * fps))

    # ---- pass 1: actor of every candidate ----------------------------------------------------
    act, actd, sgn = np.full(len(c), -1), np.full(len(c), np.nan), np.zeros(len(c), int)
    for n, i in enumerate(c.idx.values):
        lo, hi = max(0, i - 4), min(T, i + 3)
        bxy = pos[lo:hi, :2]
        if not np.isfinite(bxy[:, 0]).any():
            j = int(np.clip(i, 0, T - 1)); bxy = np.array([[c.bx[n], c.by[n]]]); lo = j
        d = np.hypot(PX[lo:lo + len(bxy)] - bxy[:, :1], PY[lo:lo + len(bxy)] - bxy[:, 1:2])
        if np.isfinite(d).any():
            k, s = np.unravel_index(np.nanargmin(d), d.shape)
            act[n], actd[n] = s, d[k, s]
            sgn[n] = tr.attack_sign(side[s], int(c.period[n]))
    c["actor_slot"], c["actor_dist"], c["sign"] = act, actd, sgn
    c["actor_side"] = [side[s] if s >= 0 else None for s in act]
    c["actor_is_gk"] = [bool(is_gk[s]) if s >= 0 else False for s in act]

    rows = []
    cidx = c.idx.values
    for n, i in enumerate(cidx):
        s = act[n]
        sg = 1 if c.bx[n] >= 0 else -1                       # nearest goal -> +L
        per = int(c.period[n])
        my = "home" if tr.attack_sign("home", per) == sg else "away"   # attacking team for this goal
        X, Y = sg * c.bx[n], sg * c.by[n]
        Z = c.bz[n]
        vx, vy = sg * c.vx_after[n], sg * c.vy_after[n]
        sp = float(np.hypot(vx, vy))
        dgx, dgy = L - X, -Y
        dist = float(np.hypot(dgx, dgy))
        a1 = np.arctan2(GOAL_HALF - Y, L - X); a2 = np.arctan2(-GOAL_HALF - Y, L - X)
        open_ang = float(abs(a1 - a2))
        cos_goal = float((vx * dgx + vy * dgy) / max(sp * dist, 1e-6))
        # straight path to the goal line
        if vx > 0.5:
            tg = (L - X) / vx
            ycross = Y + vy * tg
            vz = np.nanmean(vel[i + 1:i + 6, 2]) if np.isfinite(vel[i + 1:i + 6, 2]).any() else 0.0
            zcross = Z + vz * tg - 0.5 * G * tg ** 2
        else:
            tg, ycross, zcross, vz = np.nan, np.nan, np.nan, np.nan
        # ---- future window: until the next candidate (at most 2 s) -----------------------------
        nxt = n + 1
        while nxt < len(cidx) and cidx[nxt] <= i + 2:
            nxt += 1
        end = min(T, i + W, cidx[nxt] + 1 if nxt < len(cidx) else T)
        fx, fy, fz = sg * pos[i + 1:end, 0], sg * pos[i + 1:end, 1], pos[i + 1:end, 2]
        fok = np.isfinite(fx)
        max_x = float(np.nanmax(fx)) if fok.any() else np.nan
        reach_line = bool(fok.any() and np.nanmax(fx) >= L - 0.5)
        y_at_max = float(fy[np.nanargmax(fx)]) if fok.any() else np.nan
        z_at_max = float(fz[np.nanargmax(fx)]) if fok.any() else np.nan
        lost_share = float(1 - fok.mean()) if len(fok) else 1.0
        # long future (2 s regardless of touches) - ball ends near / beyond goal line?
        e2 = min(T, i + W)
        gx = sg * pos[i + 1:e2, 0]
        max_x2 = float(np.nanmax(gx)) if np.isfinite(gx).any() else np.nan
        # opponents / keeper near the ball path in the future window
        opp = np.array([(side[k] != my) if my else False for k in range(P)])
        gk_def = opp & is_gk
        if end > i + 1 and fok.any():
            bxf, byf = pos[i + 1:end, 0:1], pos[i + 1:end, 1:2]
            dd = np.hypot(PX[i + 1:end] - bxf, PY[i + 1:end] - byf)
            dmin_opp = float(np.nanmin(dd[:, opp & ~is_gk])) if np.isfinite(dd[:, opp & ~is_gk]).any() else np.nan
            dmin_gk = float(np.nanmin(dd[:, gk_def])) if gk_def.any() and np.isfinite(dd[:, gk_def]).any() else np.nan
        else:
            dmin_opp = dmin_gk = np.nan
        # next touch
        if nxt < len(cidx):
            j = nxt
            n_dt = (cidx[j] - i) / fps
            n_side = c.actor_side[j]; n_same = float(n_side == my) if n_side else np.nan
            n_gk = float(bool(c.actor_is_gk[j]) and n_side != my)
            n_x = sg * c.bx[j]; n_y = sg * c.by[j]
            n_dv = c.dv[j]; n_speed = c.speed_after[j]; n_actd = c.actor_dist[j]
        else:
            n_dt = n_same = n_gk = n_x = n_y = n_dv = n_speed = n_actd = np.nan
        # richer ball-flight description after the kick (peak speed, height) - 0.7 s
        e7 = min(T, i + int(0.7 * fps))
        spd_seg = ball["speed2d"][i + 1:e7]
        peak_speed = float(np.nanmax(spd_seg)) if np.isfinite(spd_seg).any() else np.nan
        zseg = pos[i + 1:e7, 2]
        max_z07 = float(np.nanmax(zseg)) if np.isfinite(zseg).any() else np.nan
        n_touch_2s = int(((cidx > i + 2) & (cidx <= i + W)).sum())
        # ball disappears for good soon after the kick near the goal line (out of play / in the net)
        last = np.flatnonzero(np.isfinite(gx))
        last_x = float(gx[last[-1]]) if last.size else np.nan
        # previous touch geometry (e.g. a cross into the box)
        prv = n - 1
        if prv >= 0:
            p_x = sg * c.bx[prv]; p_absy = abs(sg * c.by[prv]); p_z = c.bz[prv]
            p_speed = c.speed_after[prv]; p_actor_att = float(c.actor_side[prv] == my) if c.actor_side[prv] else np.nan
        else:
            p_x = p_absy = p_z = p_speed = p_actor_att = np.nan
        p_dt = (i - cidx[prv]) / fps if prv >= 0 else np.nan
        p_same = float(c.actor_side[prv] == my) if prv >= 0 and c.actor_side[prv] else np.nan
        p_sameplayer = float(act[prv] == s) if prv >= 0 else np.nan
        # context at the kick
        px, py = sg * PX[i], sg * PY[i]
        fin = np.isfinite(px)
        tri = _in_triangle(px, py, X, Y, L) & fin
        n_def_tri = int((tri & opp).sum())
        gk = np.flatnonzero(gk_def & fin)
        if gk.size:
            g = gk[np.argmax(px[gk])]
            gk_dgoal = float(np.hypot(L - px[g], py[g])); gk_dball = float(np.hypot(px[g] - X, py[g] - Y))
            # keeper's distance from the straight path ball -> goal centre
            ux, uy = dgx / max(dist, 1e-6), dgy / max(dist, 1e-6)
            rx, ry = px[g] - X, py[g] - Y
            gk_off_line = float(abs(rx * uy - ry * ux))
        else:
            gk_dgoal = gk_dball = gk_off_line = np.nan
        team = np.array([(side[k] == my) for k in range(P)]) & fin
        if s >= 0:
            team[s] = False
        mates_ahead = int(((px > X + 3) & team).sum())
        mates_box = int(((px > L - 16.5) & (np.abs(py) < 20.16) & team).sum())
        od = np.hypot(px - X, py - Y)
        near_opp = float(np.nanmin(od[opp & fin])) if (opp & fin).any() else np.nan
        opp_box = int(((px > L - 16.5) & (np.abs(py) < 20.16) & opp & fin).sum())
        actor_att = float(s >= 0 and side[s] == my)
        lo2, hi2 = max(0, i - 4), min(T, i + 3)
        bb = pos[lo2:hi2, :2]
        if np.isfinite(bb[:, 0]).any():
            dd2 = np.hypot(PX[lo2:hi2] - bb[:, :1], PY[lo2:hi2] - bb[:, 1:2])
            att_mask = np.array([side[k] == my for k in range(P)])
            d_att = float(np.nanmin(dd2[:, att_mask])) if np.isfinite(dd2[:, att_mask]).any() else np.nan
            d_def = float(np.nanmin(dd2[:, ~att_mask & ~is_gk])) if np.isfinite(dd2[:, ~att_mask & ~is_gk]).any() else np.nan
        else:
            d_att = d_def = np.nan
        rows.append({
            "peak_speed_07": peak_speed, "max_z_07": max_z07, "n_touch_2s": n_touch_2s, "last_x_2s": last_x,
            "p_x": p_x, "p_absy": p_absy, "p_z": p_z, "p_speed": p_speed, "p_actor_att": p_actor_att,
            "actor_is_attacker": actor_att, "d_nearest_attacker": d_att, "d_nearest_defender": d_def,
            "X": X, "Y": Y, "absY": abs(Y), "Z": Z, "dist_goal": dist, "open_angle": open_ang,
            "in_box": float(X > L - 16.5 and abs(Y) < 20.16), "in_six": float(X > L - 5.5 and abs(Y) < 9.16),
            "speed_after": sp, "speed_before": c.speed_before[n], "dv": c.dv[n], "turn_deg": c.turn_deg[n],
            "gap_kick": float(c.kind[n] == "gap"), "vx_goal": vx, "cos_goal": cos_goal,
            "t_to_line": tg, "y_cross": ycross, "abs_y_cross": abs(ycross) if np.isfinite(ycross) else np.nan,
            "z_cross": zcross, "vz_after": vz,
            "actor_dist": actd[n], "actor_is_gk": float(c.actor_is_gk[n]),
            "f_max_x": max_x, "f_reach_line": float(reach_line), "f_y_at_max": abs(y_at_max) if np.isfinite(y_at_max) else np.nan,
            "f_z_at_max": z_at_max, "f_lost_share": lost_share, "f_max_x_2s": max_x2,
            "f_dmin_opp": dmin_opp, "f_dmin_gk": dmin_gk,
            "n_dt": n_dt, "n_same_team": n_same, "n_opp_gk": n_gk, "n_x": n_x, "n_absy": abs(n_y) if np.isfinite(n_y) else np.nan,
            "n_dv": n_dv, "n_speed": n_speed, "n_actor_dist": n_actd,
            "p_dt": p_dt, "p_same_team": p_same, "p_same_player": p_sameplayer,
            "n_def_triangle": n_def_tri, "gk_dist_goal": gk_dgoal, "gk_dist_ball": gk_dball, "gk_off_line": gk_off_line,
            "mates_ahead": mates_ahead, "mates_in_box": mates_box, "near_opp": near_opp, "opp_in_box": opp_box,
        })
    F = pd.DataFrame(rows)
    F = F.drop(columns=[k for k in F.columns if k in c.columns])  # same values already in c
    out = pd.concat([c.reset_index(drop=True), F], axis=1)
    out["match_id"] = tr.match_id
    out = add_reaction_features(tr, ball, out)
    return out, ball


def add_reaction_features(tr: Tracking, ball: dict, out: pd.DataFrame) -> pd.DataFrame:
    """v1.4: how the game REACTS to the kick (tracking only).
    * goalkeeper reaction: the defending keeper's top speed and sideways movement in 0.8 s after the kick
    * ball held / dead: ball stays slow next to the keeper; ball still for >= 1 s soon after (a restart);
      ball goes past the goal line or disappears near it
    * neighbours: how this kick compares with the other kicks within +-1.5 s (the shot is usually the
      fastest kick toward goal in a short scramble) -> fixes 'same shot, wrong moment' detections"""
    fps = tr.fps; T = tr.n_frames
    L = float(tr.meta.get("pitch_length") or 105.0) / 2
    side = tr.players.side.values; is_gk = (tr.players.position == "GK").values
    pos, ok, sp = ball["pos"], ball["ok"], ball["speed2d"]
    still = ok & (sp < 0.6)
    # start of every still run lasting >= 1 s
    d = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    st = st[(en - st) >= int(fps)]
    rows = []
    for n, i in enumerate(out.idx.values):
        sg = 1 if out.bx.values[n] >= 0 else -1
        per = int(out.period.values[n])
        att = "home" if tr.attack_sign("home", per) == sg else "away"
        gk = np.flatnonzero((side != att) & is_gk)
        e8 = min(T, i + int(0.8 * fps))
        gk_v = gk_lat = np.nan
        if gk.size:
            g = gk[np.argmax(np.nan_to_num(sg * np.nanmedian(tr.px[max(0, i - 15):i + 15, gk], axis=0), nan=-99))]
            gx, gy = tr.px[i:e8, g], tr.py[i:e8, g]
            okg = np.isfinite(gx)
            if okg.sum() >= 5:
                v = np.hypot(np.diff(gx[okg]), np.diff(gy[okg])) * fps
                gk_v = float(np.nanpercentile(v, 90))
                gk_lat = float(abs(gy[okg][-1] - gy[okg][0]))
            # ball held by the keeper: ball slow and within 1.5 m of him for >= 0.7 s in the next 3 s
            e3 = min(T, i + int(3 * fps))
            dd = np.hypot(tr.px[i:e3, g] - pos[i:e3, 0], tr.py[i:e3, g] - pos[i:e3, 1])
            held = float(np.nansum((dd < 1.5) & (sp[i:e3] < 4)) / fps)
        else:
            held = np.nan
        k = np.searchsorted(st, i + 5)
        still_dt = (st[k] - i) / fps if k < len(st) and tr.period[min(st[k], T - 1)] == tr.period[i] else 99.0
        e3 = min(T, i + int(3 * fps))
        x3 = sg * pos[i + 1:e3, 0]
        past_line = float(np.nanmax(x3) > L) if np.isfinite(x3).any() else 0.0
        okk = ok[i + 1:e3]
        lost_near_goal = 0.0
        if okk.any() and not okk[-1]:
            last = np.flatnonzero(okk)[-1]
            lost_near_goal = float(x3[last] > L - 6)
        rows.append({"gk_react_speed": gk_v, "gk_react_lat": gk_lat, "gk_hold_s": held,
                     "still_after_s": min(still_dt, 99.0), "past_goal_line_3s": past_line,
                     "lost_near_goal": lost_near_goal})
    R = pd.DataFrame(rows, index=out.index)
    # neighbour comparison within +-1.5 s
    idx = out.idx.values; vx = out.vx_goal.values; spd = out.speed_after.values; dist = out.dist_goal.values
    w = int(1.5 * fps)
    rank_vx, rel_spd, n_nb = [], [], []
    for n in range(len(out)):
        nb = np.flatnonzero(np.abs(idx - idx[n]) <= w)
        rank_vx.append(float((vx[nb] > vx[n]).sum()))
        rel_spd.append(float(spd[n] - np.nanmax(spd[nb])))
        n_nb.append(float(len(nb) - 1))
    R["nb_rank_vx"], R["nb_rel_speed"], R["nb_count"] = rank_vx, rel_spd, n_nb
    return pd.concat([out, R], axis=1)


FEATURES = ["X", "absY", "Z", "dist_goal", "open_angle", "in_box", "in_six", "speed_after", "speed_before", "dv",
            "turn_deg", "gap_kick", "vx_goal", "cos_goal", "t_to_line", "abs_y_cross", "z_cross", "vz_after",
            "actor_dist", "actor_is_gk", "actor_is_attacker", "d_nearest_attacker",
            "d_nearest_defender", "f_max_x", "f_reach_line", "f_y_at_max", "f_z_at_max", "f_lost_share",
            "f_max_x_2s", "f_dmin_opp", "f_dmin_gk", "n_dt", "n_same_team", "n_opp_gk", "n_x", "n_absy", "n_dv",
            "n_speed", "n_actor_dist", "p_dt", "p_same_team", "p_same_player", "n_def_triangle", "gk_dist_goal",
            "gk_dist_ball", "gk_off_line", "mates_ahead", "mates_in_box", "near_opp", "opp_in_box",
            "peak_speed_07", "max_z_07", "n_touch_2s", "last_x_2s", "p_x", "p_absy", "p_z", "p_speed", "p_actor_att"]
FEATURES_V14 = ["gk_react_speed", "gk_react_lat", "gk_hold_s", "still_after_s", "past_goal_line_3s", "lost_near_goal",
                "nb_rank_vx", "nb_rel_speed", "nb_count"]
FEATURES = FEATURES + FEATURES_V14

"""
attacks.py  -  find every attack of a match and rate how dangerous it was (1 - 100)
===================================================================================

What is an attack?  (Opta's "sequence" idea: a passage of play that belongs to one team)
  * it STARTS when a team gets the ball: a recovery, a duel it wins, a kick-off, throw-in, free kick, corner, goal kick ...
  * it ENDS when the other team controls the ball, the ball goes dead (out of play, foul, substitution ...),
    nothing happens for ATTACK_MAX_PAUSE_S seconds, or the period ends.
  * a short touch by a defender that is NOT control (a block, a clearance, a goalkeeper parry) followed within
    REBOUND_MAX_S seconds by the same attacking team keeps the attack alive (second balls / rebounds).
  * who has the ball after a duel = the duel's WINNER (not the challenger).

How dangerous was it?  Two real-world measures, both in the same unit: "chance of a goal"
  1. Attack xG    = chance that at least one of its shots went in: 1 - (1 - xG1)(1 - xG2)...  (as FBref / Opta do)
                    Shot xG comes from model v2 (metrics_engine.XG_V2): fitted on ~10,700 public StatsBomb shots, it knows
                    distance, angle, headers, direct free kicks, defenders between ball and goal, the goalkeeper's position
                    and opponents close to the shooter (read here from the PFF tracking). Goals ruled out count 0, penalties 0.79.
  2. Peak xT      = the best spot the team CONTROLLED (where its players played the ball, or where its completed passes
                    and carries arrived), valued as 0.852 x EPV(spot): Laurie Shaw's EPV grid (an xT-type surface) rescaled
                    to the goal rate at shot spots on 10,708 public shots. The shot spot itself is left to the shot's xG.
  Danger probability = max(attack xG, peak xT)   (the larger, not the sum: both estimate the same chance)
  Danger score 1-100 = the same probability on a fixed logarithmic scale between 0.4 % (1) and 80 % (100):
                           2 % -> 30     5 % -> 48     10 % -> 61     20 % -> 74     40 % -> 87     79 % (penalty) -> 100
                       The scale is FIXED (not "best attack of this match = 100"), so scores compare across matches.

  Also reported (not part of the score): xT gained (sum of the positive xT steps of completed passes and carries, the
  usual way xT credits a move), passes, players involved, box entry, and the tracking picture at the key moment:
  dangerous accessible space (DAS), OBSO, and how many defenders were still goal-side of the ball.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import reference_metrics as ref
from metrics_engine import PENALTY_XG, MetricsEngine, compute_pitch_control, xg_at

ATTACKS_VERSION = "2.0"
BIG_CHANCE_XG = 0.38  # typical xG of an Opta "big chance" (secondary source); OUR threshold, Opta uses no xG cut-off
ATTACK_MAX_PAUSE_S = 12.0  # no on-ball event for this long -> the attack is over
DEAD_BALL_MAX_S = 3.0  # the ball is untracked (dead) for longer than this between two events -> the attack is over
REBOUND_MAX_S = 3.0  # a defender's block / clearance / parry followed by the same team within this time = same attack
NON_CONTROL = {"CL", "RE", "TC", "CH"}  # defender touches that are not real control of the ball
PLAY_TYPES = {"OTB", "FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}  # events that are part of play
LEAD_IN_S, LEAD_OUT_S = 3.0, 2.0  # replay window around the attack
DANGER_P_MIN, DANGER_P_MAX = 0.004, 0.8  # danger probability mapped to 1 and 100 (a penalty, 0.79, reaches the top)
# Laurie Shaw's EPV grid reads about 15 % too high at the spots where shots are taken: on 10,708 public StatsBomb shots its
# mean value at the shot location is 0.109 while the goal rate is 0.093. Scaling it by 0.852 puts "best spot reached" and
# shot xG on the same ruler (fitted on public data only; World Cup 2022 not used).
EPV_TO_GOAL_SCALE = 0.852
FINAL_THIRD_X, BOX_X, BOX_HALF_W = 17.5, 36.0, 20.16

PATTERN = {"K": "Kick-off", "C": "Corner", "F": "Free kick", "P": "Penalty", "T": "Throw-in", "G": "Goal kick", "D": "Drop ball"}
SET_PIECES = {"Corner", "Free kick", "Penalty"}
SHOT_OUTCOME = {"G": "goal", "S": "saved", "O": "off target", "B": "blocked"}


def danger_score(p: float) -> int:
    """Danger probability -> 1..100 on the fixed logarithmic scale described above."""
    if not np.isfinite(p) or p <= DANGER_P_MIN:
        return 1
    s = 100.0 * math.log(p / DANGER_P_MIN) / math.log(DANGER_P_MAX / DANGER_P_MIN)
    return int(max(1, min(100, round(s))))


def danger_label(score: int) -> str:
    return "Very dangerous" if score >= 75 else "Dangerous" if score >= 55 else "Promising" if score >= 35 else "Low threat"


class AttackFinder:
    def __init__(self, engine: MetricsEngine):
        self.e = engine
        self.m = engine.m
        self._attacks: Optional[list[dict]] = None
        self._by_id: dict[int, dict] = {}

    # ------------------------------------------------------------------ segmentation
    def _segments(self) -> list[dict]:
        m, e = self.m, self.e
        ev = m.events
        if not len(ev):
            return []
        fps = m.fps
        side_ff = e._side_ff  # possession side after every event (duel winner aware)
        fi = ev["frame_idx"].to_numpy()
        gtype = ev["game_event_type"].astype(str).to_numpy() if "game_event_type" in ev.columns else np.full(len(ev), "OTB")
        ptype = ev["poss_type"].astype(str).to_numpy()
        period = ev["period"].to_numpy()
        ball_ok = np.isfinite(m.ball_x)
        raw: list[dict] = []
        cur: Optional[dict] = None

        def close():
            nonlocal cur
            if cur is not None:
                raw.append(cur)
            cur = None

        last_f = -1
        for i in range(len(ev)):
            f = int(fi[i])
            if f < 0:
                continue
            s = side_ff[i] if i < len(side_ff) else None
            if gtype[i] not in PLAY_TYPES or s not in ("home", "away"):
                close()
                last_f = f
                continue
            if cur is not None:
                gap_s = (f - last_f) / fps
                dead = False
                if f - last_f > DEAD_BALL_MAX_S * fps:
                    seg = ~ball_ok[last_f:f]
                    if seg.any():  # longest run of untracked ball between the two events
                        edges = np.diff(np.concatenate(([0], seg.astype(np.int8), [0])))
                        dead = (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max() > DEAD_BALL_MAX_S * fps
                if cur["side"] != s or period[i] != cur["period"] or gap_s > ATTACK_MAX_PAUSE_S or dead:
                    close()
            if cur is None:
                cur = {"side": s, "period": int(period[i]), "rows": []}
            cur["rows"].append(i)
            last_f = f
        close()

        # rebounds / second balls: A (team X) - B (team Y, only non-control touches, short) - C (team X)  ->  A + C
        merged: list[dict] = []
        k = 0
        while k < len(raw):
            a = raw[k]
            while k + 2 < len(raw):
                b, c = raw[k + 1], raw[k + 2]
                if not (c["side"] == a["side"] and b["side"] != a["side"] and c["period"] == a["period"]):
                    break
                b_rows = b["rows"]
                if len(b_rows) > 2 or not all(ptype[r] in NON_CONTROL for r in b_rows):
                    break
                if any(gtype[r] not in PLAY_TYPES for r in range(a["rows"][-1] + 1, c["rows"][0])):
                    break  # the ball went dead in between (out, foul ...): the restart is a NEW attack
                if (fi[c["rows"][0]] - fi[a["rows"][-1]]) / fps > REBOUND_MAX_S + 2.0 or (fi[b_rows[-1]] - fi[b_rows[0]]) / fps > REBOUND_MAX_S:
                    break
                a = {"side": a["side"], "period": a["period"], "rows": a["rows"] + c["rows"], "defender_rows": a.get("defender_rows", []) + b_rows}
                k += 2
            merged.append(a)
            k += 1
        return merged

    # ------------------------------------------------------------------ one attack
    def _describe(self, aid: int, seg: dict, prev_side: Optional[str], prev_stop: bool) -> Optional[dict]:
        m, e = self.m, self.e
        ev = m.events
        fps = m.fps
        rows = ev.iloc[seg["rows"]]
        side = seg["side"]
        f0 = int(rows["frame_idx"].min())
        last = rows.iloc[-1]
        f_last = int(last["frame_idx"])
        nxt = int(last["next_frame_idx"]) if last["next_frame_idx"] >= 0 else f_last
        cap = 3.0 if last["poss_type"] == "SH" else 5.0
        f1 = int(min(max(nxt, f_last), f_last + cap * fps, m.n_frames - 1))
        per_mask = np.flatnonzero(m.period == m.period[f0])
        p_lo, p_hi = int(per_mask[0]), int(per_mask[-1])
        adir = m.attack_direction(side, f0)

        # ball path and xT along it
        idx = np.arange(f0, f1 + 1, 3)
        bx, by = m.ball_x[idx], m.ball_y[idx]
        okb = np.isfinite(bx) & np.isfinite(by)
        xt = np.full(len(idx), np.nan)
        if okb.any():
            xt[okb] = e.epv.value(bx[okb], by[okb], adir)
        ax = bx * adir
        # xT is earned where the team CONTROLS the ball (as in the xT literature): where its players play the ball and where
        # completed passes / carries arrive. The ball in flight (a cross over the six-yard box) or after a shot (in the net)
        # is not a controlled position and would inflate the value.
        # Each controlled spot is worth  0.852 x EPV(spot)  (EPV rescaled to the goal rate at shot spots on public data).
        # Weighting it by pitch control ("expected EPV" in Laurie Shaw's code) was tested and NOT adopted: on this match it
        # predicted shots slightly worse (AUC 0.70 vs 0.73), so the team's control of the spot is only shown as context.
        # The spot from which a SHOT is taken is left out: the shot's own xG (which sees defenders and keeper) values it.
        ctrl = []  # (value, frame index, raw EPV, pitch control)
        for r in rows.itertuples():
            if r.side != side:
                continue
            if r.poss_type != "SH" and np.isfinite(r.ball_x) and np.isfinite(r.ball_y):
                ctrl.append(self._eepv(int(r.frame_idx), side, adir, r.ball_x, r.ball_y))
            done = (r.poss_type == "PA" and r.pass_outcome == "C") or (r.poss_type == "CR" and r.cross_outcome == "C") or r.poss_type == "BC"
            if done and np.isfinite(r.end_ball_x) and np.isfinite(r.end_ball_y) and r.next_frame_idx >= 0:
                ctrl.append(self._eepv(int(r.next_frame_idx), side, adir, r.end_ball_x, r.end_ball_y))
        best_spot = max(ctrl, key=lambda c: c[0]) if ctrl else (0.0, f0, 0.0, 1.0)
        peak_xt, peak_f = best_spot[0], best_spot[1]
        first_shot_f = int(rows[rows["poss_type"] == "SH"]["frame_idx"].min()) if (rows["poss_type"] == "SH").any() else None
        pre = [c[0] for c in ctrl if first_shot_f is None or c[1] < first_shot_f]
        pre_shot_xt = max(pre) if pre else 0.0
        reached_ft = bool(np.nanmax(np.where(okb, ax, -99)) > FINAL_THIRD_X) if okb.any() else False
        in_box = bool(np.any(okb & (ax > BOX_X) & (np.abs(by) < BOX_HALF_W)))
        sx = float(rows.iloc[0]["ball_x"]) * adir if np.isfinite(rows.iloc[0]["ball_x"]) else (float(ax[okb][0]) if okb.any() else 0.0)

        # shots
        shots = []
        for r in rows[rows["poss_type"] == "SH"].itertuples():
            if not (np.isfinite(r.ball_x) and np.isfinite(r.ball_y)):
                continue
            sdir = m.attack_direction(r.side, int(r.frame_idx))
            pen = getattr(r, "setpiece_type", None) == "P"
            q, feat = e.shot_xg(ev.index.get_loc(r.Index), detail=True)
            real_goal = bool(r.is_goal)
            outcome = str(r.shot_outcome) if pd.notna(r.shot_outcome) else ""
            disallowed = outcome == "G" and not real_goal
            dist = float(np.hypot(52.5 - r.ball_x * sdir, r.ball_y))
            shots.append({"event_idx": int(r.Index), "frame_num": int(m.frame_num[int(r.frame_idx)]), "t": round((int(r.frame_idx) - f0) / fps, 2),
                          "player": str(r.player_name) if pd.notna(r.player_name) else "", "slot": int(r.actor_slot),
                          "xg": round(q, 3), "penalty": pen, "goal": real_goal, "disallowed": disallowed,
                          "outcome": "goal ruled out" if disallowed else SHOT_OUTCOME.get(outcome, "shot"), "distance_m": round(dist, 1),
                          "header": bool(feat.get("header", 0)), "defenders_in_way": int(feat["n_block"]) if "n_block" in feat else None,
                          "keeper_from_goal_m": round(feat["gk_dist"], 1) if "gk_dist" in feat else None,
                          "big_chance": bool(pen or q >= BIG_CHANCE_XG),
                          "x": float(r.ball_x), "y": float(r.ball_y)})
        counted = [s["xg"] for s in shots if not s["disallowed"]]
        att_xg = 1.0 - float(np.prod([1.0 - q for q in counted])) if counted else 0.0
        goal = any(s["goal"] for s in shots)
        disallowed = any(s["disallowed"] for s in shots) and not goal

        # qualification: it must have threatened at least a little
        if not (shots or reached_ft):
            return None

        # xT gained: positive xT steps of completed passes and carries of this team
        av = e.action_values()
        gained = 0.0
        if len(av):
            mine = av[av["event_idx"].isin(rows.index) & (av["side"] == side) & av["success"] & av["action"].isin(["pass", "carry"])]
            gained = float(np.clip(mine["delta_epv"].to_numpy(), 0, None).sum())

        passes = rows[rows["poss_type"].isin(["PA", "CR"])]
        n_pass = int(len(passes))
        n_comp = int(((passes["pass_outcome"] == "C") | (passes["cross_outcome"] == "C")).sum()) if n_pass else 0
        slots = sorted({int(s) for s in rows["actor_slot"] if s >= 0 and m.players["side"].iloc[int(s)] == side})

        # play pattern (StatsBomb-style)
        sp = rows.iloc[0]["setpiece_type"]
        pattern = PATTERN.get(sp, "Regular play") if pd.notna(sp) else "Regular play"
        duration = (f1 - f0) / fps
        # how directly did the ball travel towards goal? progress upfield / distance the ball travelled (ball path every 0.1 s)
        bxs, bys = bx[okb] * adir, by[okb]
        path = float(np.hypot(np.diff(bxs), np.diff(bys)).sum()) if len(bxs) > 1 else 0.0
        max_ax = float(np.nanmax(np.where(okb, ax, np.nan))) if okb.any() else sx
        progress = max(0.0, max_ax - sx)
        directness = float(min(1.0, progress / path)) if path > 1.0 else 0.0
        direct_speed = progress / max(duration, 1.0)  # Opta's "direct speed": metres upfield per second
        won_from_opponent = prev_side not in (None, side) and not prev_stop
        fast_break = False
        if pattern == "Regular play" and won_from_opponent and sx < 0 and reached_ft:
            ft_i = np.flatnonzero(okb & (ax > FINAL_THIRD_X))
            fast_break = bool(len(ft_i) and (idx[ft_i[0]] - f0) / fps <= 12.0)  # own half -> final third within 12 s
        # counter-attack, StatsBomb-style: an open-play turnover outside the final third, mostly direct (>= 75 % of the ball's
        # travel goes towards goal) and at least 18 yards (16.5 m) of progress
        if pattern == "Regular play" and won_from_opponent and sx < FINAL_THIRD_X and directness >= 0.75 and progress >= 16.5:
            pattern = "Counter-attack"

        danger_p = max(att_xg, peak_xt)
        score = danger_score(danger_p)
        # Opta-style attack types (several can apply)
        tags = []
        if any(s_["big_chance"] and not s_["disallowed"] for s_ in shots):
            tags.append("Big chance")
        if fast_break:
            tags.append("Fast break")
        if pattern in ("Regular play", "Counter-attack") and won_from_opponent and sx >= 12.5:
            tags.append("High turnover")  # ball won 40 m or less from the opponent's goal
        ends_well = bool(shots) or in_box
        if sx < 0 and directness >= 0.5 and ends_well:
            tags.append("Direct attack")  # from own half, at least half of the movement towards goal, ends in a shot or the box
        if len(rows[rows["poss_type"].isin(["PA", "CR"])]) >= 10 and ends_well:
            tags.append("Build-up")  # 10+ passes ending in a shot or the box

        # key moment: the best counted shot, otherwise the peak of xT
        best = max((s for s in shots if not s["disallowed"]), key=lambda s: s["xg"], default=None)
        key_idx = m.frame_to_idx(best["frame_num"]) if best else int(min(max(peak_f, f0), f1))
        context = self._context(key_idx, side, adir)

        team = m.teams[side]["name"] or side
        a = {
            "id": aid, "side": side, "team": team, "period": int(m.period[f0]),
            "minute": float(m.match_minute[f0]), "start_frame": int(m.frame_num[f0]), "end_frame": int(m.frame_num[f1]),
            "replay_start_frame": int(m.frame_num[max(p_lo, f0 - int(LEAD_IN_S * fps))]),
            "replay_end_frame": int(m.frame_num[min(p_hi, f1 + int(LEAD_OUT_S * fps))]),
            "key_frame": int(m.frame_num[key_idx]),
            "duration_s": round(duration, 1), "pattern": pattern, "set_piece": pattern in SET_PIECES,
            "n_events": int(len(rows)), "passes": n_pass, "passes_completed": n_comp, "players_involved": len(slots),
            "start_x": round(sx, 1), "max_x": round(float(np.nanmax(np.where(okb, ax, np.nan))) if okb.any() else sx, 1),
            "reached_final_third": reached_ft, "entered_box": in_box,
            "shots": shots, "n_shots": len(shots), "xg": round(att_xg, 3), "goal": goal, "disallowed_goal": disallowed,
            "peak_xt": round(peak_xt, 3), "peak_epv_raw": round(best_spot[2], 3), "peak_control": round(best_spot[3], 2),
            "pre_shot_xt": round(pre_shot_xt, 3), "xt_gained": round(gained, 3),
            "danger_p": round(danger_p, 4), "danger": score, "danger_label": danger_label(score),
            "danger_source": "shot" if att_xg >= peak_xt and counted else "position",
            "tags": tags, "directness": round(directness, 2), "direct_speed": round(direct_speed, 2), "progress_m": round(progress, 1),
            **context,
        }
        a["story"] = self._story(a, rows)
        return a

    def _eepv(self, idx: int, side: str, adir: int, x: float, y: float) -> tuple:
        """(scale x EPV x pitch control, frame, raw EPV, pitch control) for the attacking team at (x, y) on frame idx."""
        m, e = self.m, self.e
        raw = float(e.epv.value(x, y, adir))
        pc = 1.0
        try:
            slots = m.present_slots(idx)
            is_home = m.players["side"].to_numpy()[slots] == "home"
            is_att = is_home if side == "home" else ~is_home
            if is_att.any() and (~is_att).any():
                pos = np.column_stack([m.X[idx, slots], m.Y[idx, slots]]).astype(float)
                vel = np.column_stack([m.VX[idx, slots], m.VY[idx, slots]]).astype(float)
                acc = np.column_stack([m.AX[idx, slots], m.AY[idx, slots]]).astype(float)
                gk = e._goalkeepers(idx, pos, is_home)
                ball = np.array([x, y], dtype=float)
                pa, _, _ = compute_pitch_control(pos, vel, acc, is_att, ball, m.grid, e.pc_params, points=ball[None, :], is_gk=gk)
                pc = float(pa[0])
        except Exception:  # pitch control is a refinement: never let it break the list
            pc = 1.0
        return (EPV_TO_GOAL_SCALE * raw, idx, raw, pc)

    def _context(self, idx: int, side: str, adir: int) -> dict:
        """Tracking picture at the key moment: DAS, OBSO and defenders still goal-side of the ball."""
        m, e = self.m, self.e
        out = {"das": None, "obso": None, "defenders_goal_side": None}
        bx, by = m.ball_x[idx], m.ball_y[idx]
        if not (np.isfinite(bx) and np.isfinite(by)):
            return out
        sides = m.players["side"].to_numpy()
        opp = [s for s in m.present_slots(idx) if sides[s] != side and np.isfinite(m.X[idx, s])]
        out["defenders_goal_side"] = int(sum(1 for s in opp if m.X[idx, s] * adir > bx * adir))
        try:
            fm = e.frame_metrics(idx, include_obso=True)
            if fm.get("attacking_side") == side:
                out["das"] = None if fm.get("das_value") is None else round(float(fm["das_value"]), 2)
                out["obso"] = None if fm.get("obso_total") is None else round(float(fm["obso_total"]), 3)
        except Exception:  # the tracking picture is extra information: never let it break the list
            pass
        return out

    def _story(self, a: dict, rows: pd.DataFrame) -> str:
        """One plain-English sentence for the card."""
        zone = "own half" if a["start_x"] < 0 else ("final third" if a["start_x"] > FINAL_THIRD_X else "opponent's half")
        start = {
            "Counter-attack": f"win the ball in their {zone} and break forward",
            "Regular play": f"build from their {zone}" if a["start_x"] < FINAL_THIRD_X else "keep the ball in the final third",
        }.get(a["pattern"], f"attack from a {a['pattern'].lower()}")
        parts = [f"{a['team']} {start}"]
        if a["passes"]:
            parts.append(f"{a['passes_completed']}/{a['passes']} passes in {a['duration_s']:.0f} s")
        if a["shots"]:
            s = max(a["shots"], key=lambda s: (s["goal"], s["xg"]))
            who = s["player"] or "a player"
            what = "GOAL" if s["goal"] else s["outcome"]
            kind = "penalty" if s["penalty"] else f"shot from {s['distance_m']:.0f} m"
            parts.append(f"{who}'s {kind} (xG {s['xg']:.2f}): {what}")
        elif a["entered_box"]:
            parts.append("reach the penalty box without a shot")
        else:
            parts.append("reach the final third")
        return ", ".join(parts[:1]) + (": " + ", ".join(parts[1:]) if len(parts) > 1 else "") + "."

    # ------------------------------------------------------------------ public
    def attacks(self) -> list[dict]:
        if self._attacks is not None:
            return self._attacks
        segs = self._segments()
        ev = self.m.events
        gtype = ev["game_event_type"].astype(str).to_numpy() if "game_event_type" in ev.columns else None
        out: list[dict] = []
        prev_side: Optional[str] = None
        prev_end_row = -1
        for k, seg in enumerate(segs):
            first = seg["rows"][0]
            stop_between = gtype is not None and any(gtype[r] not in PLAY_TYPES for r in range(prev_end_row + 1, first))
            a = self._describe(len(out) + 1, seg, prev_side, stop_between)
            prev_side, prev_end_row = seg["side"], seg["rows"][-1]
            if a is not None:
                out.append(a)
        # ranks: (1) goals first, then danger;  (2) pure danger
        def order(lst, goals_first):
            return sorted(lst, key=lambda a: ((1 if (goals_first and a["goal"]) else 0), a["danger_p"], a["xt_gained"]), reverse=True)
        for key, gf in (("rank_goals_first", True), ("rank_danger", False)):
            for r, a in enumerate(order(out, gf), start=1):
                a[key] = r
            for side in ("home", "away"):
                for r, a in enumerate(order([x for x in out if x["side"] == side], gf), start=1):
                    a[key + "_team"] = r
        self._attacks = out
        self._by_id = {a["id"]: a for a in out}
        return out

    def summary(self) -> dict:
        at = self.attacks()
        res = {}
        for side in ("home", "away"):
            mine = [a for a in at if a["side"] == side]
            res[side] = {
                "team": self.m.teams[side]["name"], "attacks": len(mine), "shots": sum(a["n_shots"] for a in mine),
                "goals": sum(1 for a in mine if a["goal"]), "dangerous": sum(1 for a in mine if a["danger"] >= 55),
                "xg": round(sum(a["xg"] for a in mine), 2), "mean_danger": round(float(np.mean([a["danger"] for a in mine])), 1) if mine else 0.0,
            }
        return res

    def track(self, aid: int, step: int = 2) -> dict:
        """Everything the replay needs for one attack in ONE call: positions every `step` frames, events, xT curve."""
        self.attacks()
        a = self._by_id.get(int(aid))
        if a is None:
            raise KeyError(aid)
        m, e = self.m, self.e
        i0, i1 = m.frame_to_idx(a["replay_start_frame"]), m.frame_to_idx(a["replay_end_frame"])
        f0 = m.frame_to_idx(a["start_frame"])
        adir = m.attack_direction(a["side"], f0)
        frames = []
        for i in range(i0, i1 + 1, step):
            slots = m.present_slots(i)
            b = [round(float(m.ball_x[i]), 2), round(float(m.ball_y[i]), 2)] if np.isfinite(m.ball_x[i]) else None
            frames.append({"f": int(m.frame_num[i]), "t": round((i - f0) / m.fps, 3), "b": b,
                           "p": [[int(s), round(float(m.X[i, s]), 2), round(float(m.Y[i, s]), 2)] for s in slots]})
        # xT of the ball for the chart. It stops with the attack (or just after its last shot): after a shot the ball flies
        # into the goal area, which is not a position the team controls and would draw a false peak.
        t_stop = a["duration_s"] + 0.3
        if a["shots"]:
            t_stop = min(t_stop, max(s["t"] for s in a["shots"]) + 0.3)
        xt = []
        for i in range(i0, i1 + 1, 3):
            tt = (i - f0) / m.fps
            if np.isfinite(m.ball_x[i]) and tt <= t_stop:
                xt.append([round(tt, 2), round(float(e.epv.value(m.ball_x[i], m.ball_y[i], adir)), 4)])
        ev = m.events
        rows = ev[(ev["frame_idx"] >= i0) & (ev["frame_idx"] <= i1) & ev["poss_type"].notna()]
        events = []
        for r in rows.itertuples():
            fi = int(r.frame_idx)
            ok = ("pass_outcome" in ev.columns and r.pass_outcome == "C") or ("cross_outcome" in ev.columns and r.cross_outcome == "C")
            events.append({"t": round((fi - f0) / m.fps, 2), "frame_num": int(m.frame_num[fi]), "type": str(r.poss_type),
                           "side": r.side if r.side in ("home", "away") else None, "slot": int(r.actor_slot),
                           "player": str(r.player_name) if pd.notna(r.player_name) else "",
                           "x": None if not np.isfinite(r.ball_x) else round(float(r.ball_x), 2),
                           "y": None if not np.isfinite(r.ball_y) else round(float(r.ball_y), 2),
                           "end_x": None if not np.isfinite(r.end_ball_x) else round(float(r.end_ball_x), 2),
                           "end_y": None if not np.isfinite(r.end_ball_y) else round(float(r.end_ball_y), 2),
                           "completed": bool(ok), "in_attack": bool(a["start_frame"] <= int(m.frame_num[fi]) <= a["end_frame"])})
        players = [{"slot": int(p.slot), "side": p.side, "jersey": str(p.jersey), "name": str(p.name)} for p in m.players.itertuples()]
        return {"attack": a, "fps": m.fps / step, "step": step, "frames": frames, "xt": xt, "events": events, "players": players,
                "attack_dir": adir, "pitch": {"length": m.grid.length, "width": m.grid.width}}

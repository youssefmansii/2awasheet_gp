"""
player_live.py  -  live, cumulative statistics of every player "as of" any moment of the match
==============================================================================================

Everything here answers one question: *what had this player done by minute X, and how good was it?*

Two sources, both in the same units so they can be shown side by side:
  * the EVENT file  -> passes, shots, duels, defending, goalkeeper work (counted, never estimated)
  * the TRACKING    -> minutes, distance, high-speed running, sprints, top speed, where he played (heat map, thirds)
  * plus two model values from metrics_engine: xG of every shot (spot-based) and VAEP / OBV of every on-ball action.

Event codes used (PFF World Cup 2022 event data; the meaning of each code was checked against the data itself):
  possession event  PA pass, CR cross, SH shot, CH challenge (duel), CL clearance, BC ball carry, IT initial touch (reception)
  pass / cross outcome   C complete, D defended (intercepted/deflected), B blocked, O out of play, S stopped (keeper)
  shot outcome           G ball in the net (a goal only if the next restart is a kick-off, see data_loader: is_goal),
                         S saved, O off target, B blocked, C / L cleared / other (counted as "other", not on target)
  challenge winner       'challengeWinnerPlayerId' (the id of the player who won the duel)

Self-test:   python player_live.py --match 3816
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from metrics_engine import PENALTY_XG, MetricsEngine, xg_surface

# ---- physical thresholds (the usual GPS-vendor bands: 19.8 and 25.2 km/h) ----
HSR_SPEED = 5.5  # m/s  high-speed running starts here
SPRINT_SPEED = 7.0  # m/s  sprinting starts here
RUN_MIN_S = 1.0  # a run counts when the speed stays above the line for at least this long
SPEED_CAP = 10.0  # m/s (36 km/h): faster is a tracking jump, not running (elite top speed ~ 36 km/h)
SPEED_MEDIAN = 9  # frames (0.3 s) of the median filter that removes one-frame tracking spikes
STRIDE_S = 1.0  # cumulative arrays are kept once per second; the last <1 s is added exactly

KEY_PASS_WINDOW_S = 8.0  # a completed pass is a "key pass" if the receiver shoots within this time
PROGRESSIVE_M = 10.0  # a completed pass that brings the ball this many metres closer to the goal
FINAL_THIRD_X = 17.5  # attacking third starts here (pitch centre = 0, length 105)
BOX_X, BOX_HALF_W = 36.0, 20.16  # penalty box
HEAT_NX, HEAT_NY = 21, 14  # heat map cells (5 m x 4.9 m)
MIN_MIN_FOR_RANK = 10.0  # a player needs this many minutes to be ranked against his team-mates

_SHOT_LABEL = {"goal": "Goal", "saved": "Saved", "off": "Off target", "blocked": "Blocked", "disallowed": "Goal ruled out", "other": "Cleared / other"}


def _clock(period: int, minute: float) -> str:
    """45+3' style clock from the period and the period-aware match minute."""
    end = {1: 45, 2: 90, 3: 105}.get(int(period), 120)
    return f"{end}+{int(np.ceil(minute - end))}'" if minute > end else f"{max(1, int(np.ceil(minute)))}'"


def _pid(v) -> Optional[str]:
    """PFF ids come as 14004.0 / '14004' / None -> '14004' or None."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s and s.lower() not in ("nan", "none") else None


def _col(ev: pd.DataFrame, name: str) -> pd.Series:
    return ev[name] if name in ev.columns else pd.Series([None] * len(ev), index=ev.index, dtype=object)


# ===========================================================================
# Physical side (tracking)
# ===========================================================================
class PhysicalIndex:
    """Distance / high-speed running / sprints / top speed for every player, available as of any frame in O(1)."""

    def __init__(self, md):
        t0 = time.time()
        self.md = md
        T, P = md.X.shape
        self.T, self.P = T, P
        self.fps = md.fps
        self.stride = max(1, int(round(STRIDE_S * md.fps)))
        raw = md.SPEED.astype(np.float32)
        valid = np.isfinite(raw) & np.isfinite(md.X)
        filled = np.where(valid, raw, 0.0).astype(np.float32)
        S = np.empty_like(filled)
        for p in range(P):
            S[:, p] = median_filter(filled[:, p], size=SPEED_MEDIAN, mode="nearest")
        S[~valid] = np.nan
        S[S > SPEED_CAP] = np.nan  # glitch that survived the filter
        self.S = S
        self.present = valid & np.isfinite(S)
        S0 = np.nan_to_num(S, nan=0.0)
        self._blocks = {}
        for name, arr in (("dist", S0 / self.fps),
                          ("hsr", np.where((S0 >= HSR_SPEED) & (S0 < SPRINT_SPEED), S0, 0.0) / self.fps),
                          ("spr", np.where(S0 >= SPRINT_SPEED, S0, 0.0) / self.fps),
                          ("pres", self.present.astype(np.float32) / self.fps)):
            self._blocks[name] = self._cum(arr)
        # top speed = the best speed HELD for one second (trailing 1 s mean): single noisy frames cannot set it
        k = max(2, int(round(md.fps)))
        cs = np.cumsum(S0, axis=0, dtype=np.float64)
        M = np.zeros_like(S0, dtype=np.float32)
        M[k - 1:] = ((cs[k - 1:] - np.vstack([np.zeros((1, P)), cs[:-k]])) / k).astype(np.float32)
        self.M = M
        vmax = np.fmax.reduce(np.pad(M, ((0, (-T) % self.stride), (0, 0))).reshape(-1, self.stride, P), axis=1)
        self._vmax = np.concatenate([np.zeros((1, P), np.float32), np.maximum.accumulate(vmax, axis=0)])
        self.runs = [self._find_runs(S0[:, p], p) for p in range(P)]
        self.build_seconds = round(time.time() - t0, 1)

    def _cum(self, arr: np.ndarray) -> np.ndarray:
        T = arr.shape[0]
        pad = (-T) % self.stride
        a = np.pad(arr, ((0, pad), (0, 0))).reshape(-1, self.stride, arr.shape[1]).sum(axis=1, dtype=np.float64)
        return np.concatenate([np.zeros((1, arr.shape[1])), np.cumsum(a, axis=0)])

    def _find_runs(self, s: np.ndarray, p: int) -> dict:
        """High-speed runs (>= 5.5 m/s) and sprints (>= 7 m/s) lasting >= RUN_MIN_S: (start, end, peak, metres)."""
        out = {}
        need = int(round(RUN_MIN_S * self.fps))
        for name, thr in (("hsr", HSR_SPEED), ("spr", SPRINT_SPEED)):
            fast = s >= thr
            edges = np.diff(np.concatenate(([0], fast.astype(np.int8), [0])))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            rows = [(int(a), int(b), float(s[a:b].max()), float(s[a:b].sum() / self.fps)) for a, b in zip(starts, ends)
                    if b - a >= need and self.md.period[a] == self.md.period[b - 1]]
            out[name] = np.array(rows, dtype=float).reshape(-1, 4)
        return out

    def at(self, idx: int) -> dict:
        """Totals for every player from the start of the data up to and including frame `idx`."""
        k = idx // self.stride
        a, b = k * self.stride, idx + 1
        tail = np.nan_to_num(self.S[a:b], nan=0.0)
        dist = self._blocks["dist"][k] + tail.sum(axis=0) / self.fps
        hsr = self._blocks["hsr"][k] + np.where((tail >= HSR_SPEED) & (tail < SPRINT_SPEED), tail, 0.0).sum(axis=0) / self.fps
        spr = self._blocks["spr"][k] + np.where(tail >= SPRINT_SPEED, tail, 0.0).sum(axis=0) / self.fps
        pres = self._blocks["pres"][k] + self.present[a:b].sum(axis=0) / self.fps
        mt = self.M[a:b]
        vmax = np.maximum(self._vmax[k], mt.max(axis=0) if len(mt) else 0.0)
        n_spr = np.array([int((r["spr"][:, 1] <= idx + 1).sum()) for r in self.runs])
        n_hsr = np.array([int((r["hsr"][:, 1] <= idx + 1).sum()) for r in self.runs])
        return {"distance_m": dist, "hsr_m": hsr, "sprint_m": spr, "seconds": pres, "top_speed": vmax, "sprints": n_spr, "hsr_runs": n_hsr}

    def cumulative_at(self, slot: int, idxs: np.ndarray) -> dict:
        """Cumulative distance / hsr / sprint metres of one player at many frames (for the match-long chart)."""
        k = np.minimum(idxs // self.stride, self._blocks["dist"].shape[0] - 1)
        out = {}
        for name in ("dist", "hsr", "spr"):
            out[name] = self._blocks[name][k, slot].copy()
        return out


# ===========================================================================
# The live statistics
# ===========================================================================
class LivePlayers:
    def __init__(self, engine: MetricsEngine):
        self.e = engine
        self.m = engine.m
        self.lock = threading.RLock()
        self._phys: Optional[PhysicalIndex] = None
        self._acts: Optional[pd.DataFrame] = None
        self._xg = {1: xg_surface(self.m.grid, 1), -1: xg_surface(self.m.grid, -1)}
        self._flip = self._build_flip()
        self._heat_cache: dict = {}
        self._detail_cache: dict = {}

    # ------------------------------------------------------------------ geometry helpers
    def _build_flip(self) -> dict:
        """+1 / -1 per frame and side: multiply x and y by it and the team always attacks towards +x."""
        m = self.m
        out = {}
        for side in ("home", "away"):
            f = np.ones(m.n_frames, dtype=np.int8)
            for p, d in m.attack_dir.items():
                f[m.period == p] = int(d[side])
            out[side] = f
        return out

    def flip(self, side: str, fi: int) -> int:
        return int(self._flip[side][fi])

    @property
    def phys(self) -> PhysicalIndex:
        with self.lock:
            if self._phys is None:
                self._phys = PhysicalIndex(self.m)
            return self._phys

    # ------------------------------------------------------------------ the action table (one row per credited event)
    @property
    def acts(self) -> pd.DataFrame:
        with self.lock:
            if self._acts is None:
                self._acts = self._build_actions()
            return self._acts

    def _build_actions(self) -> pd.DataFrame:
        m, ev = self.m, self.m.events
        cols = ["eidx", "slot", "side", "kind", "fi", "frame_num", "period", "minute", "ok", "x", "y", "ex", "ey", "detail",
                "recv", "key", "assist", "xa", "xg", "shot_cat", "lines", "role", "pressured", "t_ev"]
        if not len(ev):
            return pd.DataFrame(columns=cols)
        pid2slot = {p: int(s) for p, s in zip(m.players["player_id"], m.players["slot"]) if p}
        slot_side = m.players["side"].to_numpy()
        rows: list[dict] = []

        def to_slot(v) -> int:
            return pid2slot.get(_pid(v), -1)

        def add(kind, slot, i, e, ok=None, **kw):
            if slot is None or slot < 0:
                return
            fi = int(e["frame_idx"])
            x, y = e["ball_x"], e["ball_y"]
            if not (np.isfinite(x) and np.isfinite(y)) and np.isfinite(m.X[fi, slot]):  # ball not tracked: use the player
                x, y = float(m.X[fi, slot]), float(m.Y[fi, slot])
            r = {"eidx": i, "slot": int(slot), "side": slot_side[slot], "kind": kind, "fi": fi, "frame_num": int(e["frame_num"]),
                 "period": int(m.period[fi]), "minute": float(m.match_minute[fi]), "ok": ok, "x": x, "y": y,
                 "ex": e["end_ball_x"], "ey": e["end_ball_y"], "detail": None, "recv": -1, "key": False, "assist": False,
                 "xa": 0.0, "xg": 0.0, "shot_cat": None, "lines": 0, "role": None, "pressured": False,
                 "t_ev": float(e["event_time_s"])}
            r.update(kw)
            rows.append(r)

        need = ["frame_idx", "frame_num", "ball_x", "ball_y", "end_ball_x", "end_ball_y", "event_time_s", "poss_type", "player_id", "passer_id",
                "receiver_id", "target_id", "shooter_id", "crosser_id", "carrier_id", "clearer_id", "dribbler_id", "pass_outcome", "cross_outcome",
                "shot_outcome", "setpiece_type", "pressure_type", "is_goal", "challenge_type", "side", "game_event_type"]
        extra = ["possessionEvents.defenderPlayerId", "possessionEvents.blockerPlayerId", "possessionEvents.keeperPlayerId",
                 "possessionEvents.challengeWinnerPlayerId", "possessionEvents.challengerPlayerId",
                 "possessionEvents.homeDuelPlayerId", "possessionEvents.awayDuelPlayerId", "possessionEvents.linesBrokenType",
                 "possessionEvents.disciplinePlayerId"]
        sub = pd.DataFrame({c: _col(ev, c) for c in need + extra})
        for i, e in enumerate(sub.to_dict("records")):
            if not (e["frame_idx"] is not None and int(e["frame_idx"]) >= 0):
                continue
            pt = e["poss_type"]
            if pt == "PA" or pt == "CR":
                is_cross = pt == "CR"
                who = e["crosser_id"] if is_cross else e["passer_id"]
                outcome = e["cross_outcome"] if is_cross else e["pass_outcome"]
                s = to_slot(who or e["player_id"])
                ok = outcome == "C"
                recv = to_slot(e["receiver_id"]) if ok else to_slot(e["target_id"])
                lines = 0 if is_cross or pd.isna(e["possessionEvents.linesBrokenType"]) else len(str(e["possessionEvents.linesBrokenType"]))
                add("cross" if is_cross else "pass", s, i, e, ok=ok, recv=recv, lines=lines, detail=outcome)
                d, b = _pid(e["possessionEvents.defenderPlayerId"]), _pid(e["possessionEvents.blockerPlayerId"])
                if outcome == "D" and d:
                    add("intercept", to_slot(d), i, e, ok=True, detail="pass")
                if outcome == "B" and b:
                    add("block", to_slot(b), i, e, ok=True, detail="pass")
            elif pt == "SH":
                s = to_slot(e["shooter_id"] or e["player_id"])
                oc = e["shot_outcome"]
                real = bool(e["is_goal"])
                cat = ("goal" if real else "disallowed") if oc == "G" else {"S": "saved", "O": "off", "B": "blocked"}.get(oc, "other")
                add("shot", s, i, e, ok=real, detail=oc, shot_cat=cat)
                k, b = _pid(e["possessionEvents.keeperPlayerId"]), _pid(e["possessionEvents.blockerPlayerId"])
                if k and cat in ("goal", "saved"):
                    add("gk_shot", to_slot(k), i, e, ok=(cat == "saved"), detail=cat)
                if oc == "B" and b:
                    add("block", to_slot(b), i, e, ok=True, detail="shot")
            elif pt == "CH":
                winner = _pid(e["possessionEvents.challengeWinnerPlayerId"])
                chal, drib = _pid(e["possessionEvents.challengerPlayerId"]), _pid(e["dribbler_id"])
                hd, ad = _pid(e["possessionEvents.homeDuelPlayerId"]), _pid(e["possessionEvents.awayDuelPlayerId"])
                parts = [p for p in ((hd, ad) if hd and ad else (_pid(e["player_id"]), chal)) if p]
                if winner and winner not in parts:
                    parts.append(winner)
                for p in dict.fromkeys(parts):
                    role = "challenger" if p == chal else "dribbler" if p == drib else "other"
                    add("duel", to_slot(p), i, e, ok=(p == winner) if winner else None, detail=e["challenge_type"], role=role)
            elif pt == "CL":
                add("clearance", to_slot(e["clearer_id"] or e["player_id"]), i, e, ok=True)
            elif pt == "BC":
                add("carry", to_slot(e["carrier_id"] or e["player_id"]), i, e, ok=True)
            elif pt == "IT":
                add("reception", to_slot(e["player_id"]), i, e, ok=True, pressured=(e["pressure_type"] == "P"))
            if not pd.isna(e["possessionEvents.disciplinePlayerId"]):
                add("card", to_slot(e["possessionEvents.disciplinePlayerId"]), i, e, ok=False)

        a = pd.DataFrame(rows, columns=cols)
        if not len(a):
            return a
        # ---- xG of every shot (spot model, penalties at the usual conversion rate) ----
        sh = a["kind"] == "shot"
        for j in a.index[sh]:
            r = a.loc[j]
            adir = int(m.attack_dir.get(int(r["period"]), next(iter(m.attack_dir.values())))[r["side"]])
            if str(ev["setpiece_type"].iloc[int(r["eidx"])]) == "P":
                xg = PENALTY_XG
            elif np.isfinite(r["x"]) and np.isfinite(r["y"]):
                iy, ix = m.grid.cell_index(float(r["x"]), float(r["y"]))
                xg = float(self._xg[1 if adir == 1 else -1][iy, ix])
            else:
                xg = 0.0
            a.at[j, "xg"] = xg
        # ---- key passes / assists: a completed pass whose receiver shoots within a few seconds ----
        shots = a[sh]
        passes = a[a["kind"].isin(["pass", "cross"]) & (a["ok"] == True) & (a["recv"] >= 0)]  # noqa: E712
        for j, s in shots.iterrows():
            cand = passes[(passes["recv"] == s["slot"]) & (passes["side"] == s["side"]) & (passes["t_ev"] <= s["t_ev"])
                          & (passes["t_ev"] >= s["t_ev"] - KEY_PASS_WINDOW_S) & (passes["eidx"] < s["eidx"])]
            if len(cand):
                pj = cand.index[-1]  # the latest one
                a.at[pj, "key"] = True
                a.at[pj, "xa"] = float(s["xg"])
                a.at[pj, "assist"] = bool(s["shot_cat"] == "goal")
        # ---- model values (VAEP / OBV of on-ball actions) ----
        av = self.e.action_values()
        for c in ("vaep", "vaep_oa", "obv"):
            a[c] = np.nan
        if len(av):
            key = av.set_index(["event_idx", "slot"])[["vaep", "vaep_oa", "obv"]]
            idx = pd.MultiIndex.from_arrays([a["eidx"], a["slot"]])
            got = key.reindex(idx)
            on_ball = a["kind"].isin(["pass", "cross", "shot", "carry", "duel", "clearance"]).to_numpy()
            for c in ("vaep", "vaep_oa", "obv"):
                a[c] = np.where(on_ball, got[c].to_numpy(), np.nan)
        return a.sort_values(["fi", "eidx"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ counting
    @staticmethod
    def _ratio(a: float, b: float) -> Optional[float]:
        return None if not b else float(a) / float(b)

    def _count(self, g: pd.DataFrame, side: str) -> dict:
        """All event-based numbers of ONE player from his rows `g` (already limited to the moment of interest)."""
        k = g["kind"]
        ps, cr, sh = g[k == "pass"], g[k == "cross"], g[k == "shot"]
        du = g[k == "duel"]
        ok = lambda d: int((d["ok"] == True).sum())  # noqa: E712
        n_ps, n_ps_ok = len(ps), ok(ps)
        comp = ps[ps["ok"] == True]  # noqa: E712
        flip = np.where(ps["period"] == 1, 1, 1)  # placeholder (kept simple): direction handled by caller
        del flip
        dirs = np.array([self._dir(side, int(p)) for p in comp["period"]]) if len(comp) else np.array([])
        if len(comp):
            dx = (comp["ex"].to_numpy(dtype=float) - comp["x"].to_numpy(dtype=float)) * dirs
            sx = comp["x"].to_numpy(dtype=float) * dirs
            exn = comp["ex"].to_numpy(dtype=float) * dirs
            eyn = np.abs(comp["ey"].to_numpy(dtype=float))
            fin = np.isfinite(dx)
            prog = int(((dx >= PROGRESSIVE_M) & fin).sum())
            third = int(((exn >= FINAL_THIRD_X) & (sx < FINAL_THIRD_X) & fin).sum())
            in_box = lambda ex, ey: (ex >= BOX_X) & (ey <= BOX_HALF_W)
            box = int((in_box(exn, eyn) & ~in_box(sx, np.abs(comp["y"].to_numpy(dtype=float))) & fin).sum())
        else:
            prog = third = box = 0
        cats = sh["shot_cat"].value_counts().to_dict()
        t_ = du[du["detail"] == "T"]
        tackles = t_[t_["role"] == "challenger"]
        aer = du[du["detail"] == "A"]
        drb = du[(du["detail"] == "D") & (du["role"] == "dribbler")]
        ground = du[~du["detail"].isin(["T", "A"]) & ~((du["detail"] == "D") & (du["role"] == "dribbler"))]
        known = du[du["ok"].notna()]
        gk = g[k == "gk_shot"]
        n_saves, n_conc = int((gk["detail"] == "saved").sum()), int((gk["detail"] == "goal").sum())
        out = {
            "passes": n_ps, "passes_ok": n_ps_ok, "pass_acc": self._ratio(n_ps_ok, n_ps),
            "progressive_passes": prog, "final_third_passes": third, "box_passes": box,
            "key_passes": int(ps["key"].sum() + cr["key"].sum()), "assists": int(ps["assist"].sum() + cr["assist"].sum()),
            "xa": float(ps["xa"].sum() + cr["xa"].sum()), "lines_broken": int(ps["lines"].sum()),
            "crosses": len(cr), "crosses_ok": ok(cr),
            "receptions": int((k == "reception").sum()), "receptions_pressured": int(g[k == "reception"]["pressured"].sum()),
            "shots": len(sh), "goals": int(cats.get("goal", 0)),
            "shots_on_target": int(cats.get("goal", 0) + cats.get("saved", 0)),
            "shots_saved": int(cats.get("saved", 0)), "shots_off": int(cats.get("off", 0)), "shots_blocked": int(cats.get("blocked", 0)),
            "shots_other": int(cats.get("other", 0)), "goals_ruled_out": int(cats.get("disallowed", 0)),
            "xg": float(sh["xg"].sum()),
            "tackles": len(tackles), "tackles_won": ok(tackles),
            "aerials": len(aer), "aerials_won": ok(aer),
            "dribbles": len(drb), "dribbles_won": ok(drb),
            "ground_duels": len(ground), "ground_duels_won": ok(ground),
            "duels": len(known), "duels_won": ok(known),
            "interceptions": int((k == "intercept").sum()), "blocks": int((k == "block").sum()),
            "clearances": int((k == "clearance").sum()), "carries": int((k == "carry").sum()),
            "cautions": int((k == "card").sum()),
            "saves": n_saves, "goals_conceded": n_conc, "shots_on_target_faced": n_saves + n_conc,
            "save_pct": self._ratio(n_saves, n_saves + n_conc),
        }
        out["duel_win_pct"] = self._ratio(out["duels_won"], out["duels"])
        out["cross_acc"] = self._ratio(out["crosses_ok"], out["crosses"])
        out["defensive_actions"] = out["tackles_won"] + out["interceptions"] + out["blocks"] + out["clearances"]
        out["on_ball_actions"] = out["passes"] + out["crosses"] + out["shots"] + out["carries"] + out["dribbles"] + out["clearances"] + out["tackles"]
        val = g.dropna(subset=["vaep_oa"]) if "vaep_oa" in g.columns else g.iloc[0:0]
        out["value_actions"] = int(len(val))
        out["vaep_oa"] = float(val["vaep_oa"].sum()) if len(val) else 0.0
        out["vaep_raw"] = float(val["vaep"].sum()) if len(val) else 0.0
        out["obv"] = float(val["obv"].sum()) if len(val) else 0.0
        return out

    def _dir(self, side: str, period: int) -> int:
        d = self.m.attack_dir.get(period) or next(iter(self.m.attack_dir.values()))
        return int(d[side])

    # ------------------------------------------------------------------ positions "now" (with a 1 s look-back: single frames can be empty, e.g. the very last one)
    def recent(self, idx: int) -> dict:
        m = self.m
        w = max(2, int(round(m.fps)))
        a = max(0, idx - w)
        seg = m.X[a: idx + 1]
        fin = np.isfinite(seg)
        has = fin.any(axis=0)
        last = np.where(has, seg.shape[0] - 1 - np.argmax(fin[::-1], axis=0), 0)
        cols = np.arange(seg.shape[1])
        fr = a + last
        return {"on": has, "x": m.X[fr, cols], "y": m.Y[fr, cols], "speed": self.phys.S[fr, cols] * 3.6, "frame": fr}

    def ball_at(self, idx: int) -> Optional[dict]:
        m = self.m
        w = max(2, int(round(m.fps)))
        for f in range(idx, max(-1, idx - w - 1), -1):
            if np.isfinite(m.ball_x[f]) and np.isfinite(m.ball_y[f]):
                return {"x": float(m.ball_x[f]), "y": float(m.ball_y[f])}
        return None

    # ------------------------------------------------------------------ everybody at one moment
    def table(self, idx: int) -> list[dict]:
        """One row per player with the numbers that fit in a list (the picker and the team comparison)."""
        m = self.m
        ph = self.phys.at(idx)
        rc = self.recent(idx)
        acts = self.acts
        upto = acts[acts["fi"] <= idx] if len(acts) else acts
        by_slot = {int(s): g for s, g in upto.groupby("slot")} if len(upto) else {}
        out = []
        for _, p in m.players.iterrows():
            s = int(p["slot"])
            g = by_slot.get(s)
            c = self._count(g if g is not None else upto.iloc[0:0], p["side"]) if len(acts) else self._count(acts, p["side"])
            minutes = float(ph["seconds"][s]) / 60.0
            c.update({
                "slot": s, "player_id": p["player_id"], "name": p["name"], "jersey": p["jersey"], "side": p["side"],
                "team": p["team_name"], "position": p["position"], "minutes": round(minutes, 1),
                "on_pitch": bool(rc["on"][s]),
                "x": float(rc["x"][s]) if rc["on"][s] else None, "y": float(rc["y"][s]) if rc["on"][s] else None,
                "speed_kmh": float(rc["speed"][s]) if rc["on"][s] and np.isfinite(rc["speed"][s]) else None,
                "distance_m": float(ph["distance_m"][s]), "hsr_m": float(ph["hsr_m"][s]), "sprint_m": float(ph["sprint_m"][s]),
                "sprints": int(ph["sprints"][s]), "hsr_runs": int(ph["hsr_runs"][s]), "top_speed_kmh": float(ph["top_speed"][s]) * 3.6,
            })
            out.append(c)
        return out

    # ------------------------------------------------------------------ one player in detail
    def heatmap(self, slot: int, idx: int) -> dict:
        m = self.m
        side = str(m.players.iloc[slot]["side"])
        step = max(1, int(round(m.fps / 5)))  # 5 samples per second is plenty for a heat map
        xs, ys = m.X[: idx + 1 : step, slot], m.Y[: idx + 1 : step, slot]
        f = self._flip[side][: idx + 1 : step].astype(float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not ok.any():
            return {"nx": HEAT_NX, "ny": HEAT_NY, "cells": np.zeros((HEAT_NY, HEAT_NX)).tolist(), "avg": None, "thirds": None, "width": None}
        x, y = xs[ok] * f[ok], ys[ok] * f[ok]
        L, W = m.grid.length, m.grid.width
        h, _, _ = np.histogram2d(y, x, bins=[HEAT_NY, HEAT_NX], range=[[-W / 2, W / 2], [-L / 2, L / 2]])
        h = h / h.sum()
        thirds = [float((x < -FINAL_THIRD_X).mean()), float(((x >= -FINAL_THIRD_X) & (x < FINAL_THIRD_X)).mean()), float((x >= FINAL_THIRD_X).mean())]
        return {"nx": HEAT_NX, "ny": HEAT_NY, "cells": np.round(h, 5).tolist(), "avg": [float(x.mean()), float(y.mean())],
                "thirds": thirds, "width": float(np.percentile(y, 90) - np.percentile(y, 10))}

    def _maps(self, g: pd.DataFrame, side: str) -> dict:
        """Everything that is drawn on the pitch, attacking left to right."""
        m = self.m
        out = {"passes": [], "shots": [], "defending": [], "duels": []}
        for r in g.itertuples():
            f = self.flip(side, int(r.fi))
            sx, sy = (r.x * f, r.y * f) if np.isfinite(r.x) else (None, None)
            ex, ey = (r.ex * f, r.ey * f) if np.isfinite(r.ex) else (None, None)
            if sx is None:
                continue
            base = {"clock": _clock(r.period, float(r.minute)), "minute": round(float(r.minute), 1), "frame_num": int(r.frame_num), "x": round(float(sx), 1), "y": round(float(sy), 1)}
            if r.kind in ("pass", "cross"):
                out["passes"].append({**base, "ex": None if ex is None else round(float(ex), 1), "ey": None if ey is None else round(float(ey), 1),
                                      "ok": bool(r.ok), "key": bool(r.key), "assist": bool(r.assist), "cross": r.kind == "cross",
                                      "to": self._short(r.recv)})
            elif r.kind == "shot":
                out["shots"].append({**base, "xg": round(float(r.xg), 3), "cat": r.shot_cat, "label": _SHOT_LABEL.get(r.shot_cat, r.shot_cat),
                                     "ex": None if ex is None else round(float(ex), 1), "ey": None if ey is None else round(float(ey), 1)})
            elif r.kind in ("intercept", "block", "clearance"):
                out["defending"].append({**base, "type": {"intercept": "Interception", "block": "Block", "clearance": "Clearance"}[r.kind]})
            elif r.kind == "duel" and r.ok is not None and not pd.isna(r.ok):
                out["duels"].append({**base, "type": self._duel_name(r.detail, r.role), "won": bool(r.ok)})
        return out

    def _short(self, slot) -> Optional[str]:
        if slot is None or int(slot) < 0:
            return None
        p = self.m.players.iloc[int(slot)]
        return f"#{p['jersey']} {p['name']}"

    @staticmethod
    def _duel_name(code, role) -> str:
        if code == "T":
            return "Tackle" if role == "challenger" else "Dribbled past / tackled"
        if code == "A":
            return "Aerial duel"
        if code == "D":
            return "Dribble" if role == "dribbler" else "Dribble defended"
        if code == "FIFTY":
            return "50-50 duel"
        return "Duel"

    def detail(self, slot: int, idx: int) -> dict:
        """Everything about one player as of frame `idx` (what the live view shows)."""
        key = (slot, idx)
        hit = self._detail_cache.get(key)
        if hit is not None:
            return hit
        m = self.m
        rows = self.table(idx)
        me = next(r for r in rows if r["slot"] == slot)
        side = me["side"]
        mates = [r for r in rows if r["side"] == side]
        acts = self.acts
        g = acts[(acts["slot"] == slot) & (acts["fi"] <= idx)] if len(acts) else acts
        heat = self.heatmap(slot, idx)
        maps = self._maps(g, side)
        f = self.flip(side, idx)
        now = None
        if me["on_pitch"] and me["x"] is not None and me["y"] is not None:
            now = {"x": float(me["x"] * f), "y": float(me["y"] * f), "speed_kmh": me["speed_kmh"]}
        team, team_pool = self._team_context(me, mates)
        res = {
            "match_id": m.match_id, "frame_num": int(m.frame_num[idx]), "period": int(m.period[idx]),
            "minute": round(float(m.match_minute[idx]), 1), "stats": me, "now": now,
            "heat": heat, "maps": maps, "team": team, "team_pool": team_pool,
            "has_events": bool(len(acts)), "events_note": None if len(acts) else "No events file for this match: only the tracking numbers are available.",
        }
        res["summary"] = self._summary(me, res["team"], heat, res["team_pool"]["label"])
        if len(self._detail_cache) > 80:
            self._detail_cache.clear()
        self._detail_cache[key] = res
        return res

    # ------------------------------------------------------------------ comparison with team-mates and words
    RANKED = [  # key, label, minimum of "chances" needed (None = no minimum), what the minimum counts, higher is better
        ("pass_acc", "Pass accuracy", 10, "passes", True),
        ("passes", "Passes", None, None, True),
        ("progressive_passes", "Progressive passes", None, None, True),
        ("key_passes", "Key passes", None, None, True),
        ("xg", "xG (shot quality)", None, None, True),
        ("defensive_actions", "Defensive actions", None, None, True),
        ("duel_win_pct", "Duels won %", 5, "duels", True),
        ("distance_m", "Distance covered", None, None, True),
        ("hsr_m", "High-speed running", None, None, True),
        ("vaep_oa", "Value added (VAEP above avg)", None, None, True),
    ]

    @staticmethod
    def pos_group(pos: Optional[str]) -> str:
        p = (pos or "").upper()
        if p == "GK":
            return "goalkeepers"
        if p in ("CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB", "WB", "D", "DF"):
            return "defenders"
        if p in ("CF", "ST", "SS", "LW", "RW", "F", "FW", "CF-L", "CF-R"):
            return "forwards"
        return "midfielders"

    def _team_context(self, me: dict, mates: list[dict]) -> tuple[dict, dict]:
        """Rank among team-mates of the same kind (defenders / midfielders / forwards; else all outfield players) who played enough."""
        gk = me["position"] == "GK"
        base = [r for r in mates if r["minutes"] >= MIN_MIN_FOR_RANK and (r["position"] == "GK") == gk]
        group = self.pos_group(me["position"])
        same = [r for r in base if self.pos_group(r["position"]) == group]
        if gk:
            pool, label = base, "goalkeepers"
        elif len(same) >= 4:
            pool, label = same, f"{group} of his team"
        else:
            pool, label = base, "outfield team-mates"
        out = {}
        for key, name, need, need_key, hi in self.RANKED:
            vals = [(r["slot"], r[key]) for r in pool if r[key] is not None and (need is None or r[need_key] >= need)]
            mine = me.get(key)
            entry = {"label": name, "value": mine, "avg": float(np.mean([v for _, v in vals])) if vals else None, "n": len(vals), "rank": None}
            if mine is not None and len(vals) >= 3 and (need is None or me[need_key] >= need) and me["minutes"] >= MIN_MIN_FOR_RANK:
                better = sum(1 for s_, v in vals if s_ != me["slot"] and v > mine)
                entry["rank"] = better + 1
                entry["n"] = len(vals) if any(s_ == me["slot"] for s_, _ in vals) else len(vals) + 1
            out[key] = entry
        return out, {"size": len(pool), "label": label}

    def _summary(self, me: dict, team: dict, heat: dict, pool_label: str = "team-mates") -> list[dict]:
        """Short plain-English readout. tone: good / ok / poor / info."""
        s: list[dict] = []
        name = me["name"]
        if me["minutes"] < 0.5:
            return [{"tone": "info", "text": f"{name} has not played yet at this moment."}]

        def add(tone, text):
            s.append({"tone": tone, "text": text})

        def rank_tone(key):
            e = team.get(key) or {}
            if e.get("rank") is None or not e.get("n"):
                return "info", ""
            frac = (e["rank"] - 1) / max(e["n"] - 1, 1)
            return ("good" if frac <= 0.25 else "poor" if frac >= 0.75 else "ok"), f" (ranked {e['rank']} of {e['n']} {pool_label})"

        mins = round(me["minutes"])
        add("info", f"{name} has played {mins} minute{'' if mins == 1 else 's'} so far" + ("" if me["on_pitch"] else " and is not on the pitch at this moment") + ".")
        if me["position"] == "GK":
            if me["shots_on_target_faced"]:
                add("good" if (me["save_pct"] or 0) >= 0.7 else "ok" if (me["save_pct"] or 0) >= 0.5 else "poor",
                    f"Faced {me['shots_on_target_faced']} shot(s) on target: {me['saves']} saved, {me['goals_conceded']} conceded ({(me['save_pct'] or 0) * 100:.0f}% saved).")
            else:
                add("info", "No shot on target faced yet.")
        if me["passes"]:
            tone, rk = rank_tone("pass_acc")
            avg = (team.get("pass_acc") or {}).get("avg")
            cmp_ = f", {pool_label} average {avg * 100:.0f}%" if avg is not None else ""
            add(tone, f"Passing: {me['passes_ok']} of {me['passes']} passes completed ({(me['pass_acc'] or 0) * 100:.0f}%{cmp_}){rk}.")
            bits = []
            if me["key_passes"]:
                bits.append(f"{me['key_passes']} key pass(es) leading to a shot" + (f", {me['assists']} assist(s)" if me["assists"] else ""))
            if me["progressive_passes"]:
                bits.append(f"{me['progressive_passes']} progressive pass(es)")
            if me["lines_broken"]:
                bits.append(f"{me['lines_broken']} line(s) of the defence broken")
            if bits:
                add("info", "Creating: " + ", ".join(bits) + ".")
        if me["shots"]:
            parts = [f"{me['shots_on_target']} on target" + (f" ({me['goals']} goal{'s' if me['goals'] != 1 else ''})" if me["goals"] else ""),
                     f"{me['shots_off']} off target", f"{me['shots_blocked']} blocked"]
            if me["goals_ruled_out"]:
                parts.append(f"{me['goals_ruled_out']} goal(s) ruled out")
            add("good" if me["goals"] else "info", f"Shooting: {me['shots']} shot(s): " + ", ".join(parts) + f". Shot quality (xG) {me['xg']:.2f}.")
        if me["duels"]:
            tone, rk = rank_tone("duel_win_pct")
            add(tone, f"Duels: won {me['duels_won']} of {me['duels']} ({(me['duel_win_pct'] or 0) * 100:.0f}%){rk}.")
        if me["defensive_actions"]:
            add("info", f"Defending: {me['tackles_won']} tackle(s) won, {me['interceptions']} interception(s), {me['blocks']} block(s), {me['clearances']} clearance(s).")
        if me["minutes"] >= 1:
            tone, rk = rank_tone("distance_m")
            add(tone, f"Running: {me['distance_m'] / 1000:.1f} km ({me['distance_m'] / me['minutes']:.0f} m per minute), {me['sprints']} sprint(s), top speed {me['top_speed_kmh']:.1f} km/h{rk}.")
        if heat.get("thirds"):
            d, mi, a = heat["thirds"]
            add("info", f"Position: {d * 100:.0f}% of his time in his own third, {mi * 100:.0f}% in the middle, {a * 100:.0f}% in the attacking third.")
        if me["value_actions"]:
            tone, rk = rank_tone("vaep_oa")
            v = me["vaep_oa"]
            add(tone if tone != "info" else ("good" if v > 0.02 else "poor" if v < -0.02 else "ok"),
                f"Value added: {v:+.2f} (VAEP above average over {me['value_actions']} valued actions; above 0 means he helped his team's chances more than an average player would){rk}.")
        if me["cautions"]:
            add("poor", f"Booked ({me['cautions']} caution).")
        return s

    # ------------------------------------------------------------------ the whole match for one player (static chart data)
    def timeline(self, slot: int) -> dict:
        """The whole match of one player, one row per minute of tracked play (cumulative numbers up to the END of that minute).
        'k' counts minutes of play in the order they happened (the match clock restarts at the second half, so it cannot be an axis)."""
        m = self.m
        ph = self.phys
        present = np.flatnonzero(np.isfinite(m.X[:, slot]))
        if not len(present):
            return {"minutes": [], "moments": []}
        first, last = int(present[0]), int(present[-1])
        fps = m.fps
        ks = np.arange(0, int(np.ceil(m.n_frames / fps / 60.0)) + 1)
        fidx = np.minimum(np.round(ks * 60.0 * fps).astype(int), m.n_frames - 1)
        cum = ph.cumulative_at(slot, fidx)
        acts = self.acts
        g = acts[acts["slot"] == slot] if len(acts) else acts
        rows = []
        for i, k in enumerate(ks):
            fi = int(fidx[i])
            gi = g[g["fi"] <= fi]
            ps = gi[gi["kind"] == "pass"]
            vv = gi["vaep_oa"].dropna() if "vaep_oa" in gi.columns else []
            rows.append({
                "k": int(k), "period": int(m.period[fi]), "match_minute": round(float(m.match_minute[fi]), 1), "frame_num": int(m.frame_num[fi]),
                "on_pitch": bool(first <= fi <= last),
                "distance_km": float(cum["dist"][i]) / 1000.0, "hsr_m": float(cum["hsr"][i]), "sprint_m": float(cum["spr"][i]),
                "passes": int(len(ps)), "passes_ok": int((ps["ok"] == True).sum()),  # noqa: E712
                "shots": int((gi["kind"] == "shot").sum()), "value": float(np.sum(vv)) if len(vv) else 0.0,
                "defensive": int(gi["kind"].isin(["intercept", "block", "clearance"]).sum() + ((gi["kind"] == "duel") & (gi["detail"] == "T") & (gi["role"] == "challenger") & (gi["ok"] == True)).sum()),  # noqa: E712
            })
        moments = []
        for r in g.itertuples():
            tone = None
            if r.kind == "shot":
                tone = ("goal" if r.shot_cat == "goal" else "shot")
                label = f"{_SHOT_LABEL.get(r.shot_cat, 'Shot')} (xG {r.xg:.2f})"
            elif r.kind in ("pass", "cross") and r.key:
                tone, label = ("assist" if r.assist else "key"), ("Assist" if r.assist else "Key pass") + (f" to {self._short(r.recv)}" if r.recv >= 0 else "")
            elif r.kind == "card":
                tone, label = "card", "Booked"
            elif r.kind == "gk_shot":
                tone, label = "save", "Save" if r.ok else "Goal conceded"
            elif r.kind == "duel" and r.ok == True and r.detail == "T" and r.role == "challenger":  # noqa: E712
                tone, label = "def", "Tackle won"
            if tone:
                moments.append({"k": round(float(r.fi) / fps / 60.0, 2), "period": int(r.period), "match_minute": round(float(r.minute), 1), "clock": _clock(r.period, float(r.minute)),
                                "frame_num": int(r.frame_num), "tone": tone, "label": label})
        period_starts = [round(float(np.flatnonzero(m.period == p)[0]) / fps / 60.0, 2) for p in sorted(set(m.period.tolist()))]
        return {"slot": slot, "minutes": rows, "moments": moments, "period_starts": period_starts,
                "first_k": round(first / fps / 60.0, 2), "last_k": round(last / fps / 60.0, 2)}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_match

    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    md = load_match(args.match, data_dir=args.data_dir, verbose=False)
    eng = MetricsEngine(md)
    lp = LivePlayers(eng)
    t = time.time()
    rows = lp.table(md.n_frames - 1)
    print(f"full-time table in {time.time() - t:.1f}s (physical index {lp.phys.build_seconds}s)")
    df = pd.DataFrame(rows).sort_values("minutes", ascending=False)
    print(df[["name", "jersey", "side", "minutes", "passes", "pass_acc", "shots", "shots_on_target", "goals", "xg", "defensive_actions", "distance_m", "sprints", "top_speed_kmh", "vaep_oa"]].round(2).head(30).to_string(index=False))

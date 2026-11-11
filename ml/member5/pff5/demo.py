"""
Live demo: run the trained Member 5 models on a match and look at single moments.

    from pff5 import demo
    d = demo.Demo(compact="compact", run="runs/raw", game=10517)
    d.summary()                 # how the models do on this match vs PFF's annotators
    d.pressure_moments()        # table of moments to pick from
    d.show_pressure(12)         # pitch picture of one moment: model vs PFF
    d.fouls_flagged()           # events the model flags as foul / offside
    d.show_event(5, "offside")  # pitch picture of a flagged event
"""
from __future__ import annotations

import gzip
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import features as F
from .data import game_path, load_match
from .figures import BLUE, INK, INK2, MUTED, ORANGE, draw_pitch
from .models import binary_report
from .predict import Member5

KIND = {"PA": "pass", "BC": "ball carry", "SH": "shot", "CR": "cross", "CL": "clearance", "TC": "touch",
        "RE": "rebound", "IT": "first touch", "CH": "challenge"}
PNAME = {"N": "no pressure", "A": "attempted pressure", "L": "passing-lane pressure", "P": "pressured"}


class Demo:
    def __init__(self, compact: str, run: str, game: int = 10517):
        self.m5 = Member5(run)
        self.m = load_match(game_path(compact, game), with_events=True, ball_source=self.m5.ball)
        self.gid = game
        self.names = {}
        p = os.path.join(compact, f"{game}_events.json.gz")
        for e in json.load(gzip.open(p, "rt")):
            for part in ("gameEvents", "possessionEvents", "initialTouch", "fouls"):
                d = e.get(part) or {}
                for k, v in d.items():
                    if k.endswith("PlayerId") and v is not None and d.get(k[:-2] + "Name"):
                        self.names[int(v)] = d[k[:-2] + "Name"]
        self._press = None
        self._fouls = None

    def name(self, slot):
        if slot is None or slot < 0:
            return "-"
        return self.names.get(int(self.m.player_id[slot]), f"#{self.m.jersey[slot]}")

    def _clock(self, i):
        m = self.m
        t = (i - m._period_bounds()[0][m.period[i]]) / m.fps
        base = {1: 0, 2: 45, 3: 90, 4: 105}.get(int(m.period[i]), 0)
        mm = base + int(t // 60)
        return f"{mm:02d}:{int(t % 60):02d}"

    # ------------------------------------------------------------------ pressure
    def _run_pressure(self):
        if self._press is not None:
            return self._press
        m = self.m
        rows = F.pressure_rows(m)
        pr = self.m5.pressure(m, rows["idx"].values, m.player_id[rows["actor"].values], rows["pe_type"].values,
                              rows["kind"].values == 0, rows["hold"].values)
        d = pd.DataFrame(dict(
            time=[self._clock(i) for i in rows["idx"].values], idx=rows["idx"].values,
            moment=[KIND.get(t, t) if k == 1 else "first touch" for t, k in zip(rows["pe_type"], rows["kind"])],
            player=[self.name(s) for s in rows["actor"].values],
            PFF_type=rows["y_type"].values, model_type=pr["pressureType"].values,
            model_confidence=pr[["p_N", "p_A", "p_L", "p_P"]].max(1).round(2).values,
            PFF_presser=[self.name(s) for s in rows["presser"].values],
            model_presser=[self.names.get(int(p), "-") if p and p > 0 else "-" for p in pr["likelyPresserPlayerId"].values],
            actor=rows["actor"].values, presser=rows["presser"].values, pN=pr["p_N"].values,
            pid_pred=pr["likelyPresserPlayerId"].values))
        self._press = d
        return d

    def summary(self):
        """Scores on this match only (the models never trained on it)."""
        d = self._run_pressure()
        ok = d["PFF_type"].isin(list("NALP"))
        acc = (d.loc[ok, "PFF_type"] == d.loc[ok, "model_type"]).mean()
        base = (d.loc[ok, "PFF_type"] == "N").mean()
        y = (d.loc[ok, "PFF_type"] != "N").astype(int).values
        auc = binary_report(y, 1 - d.loc[ok, "pN"].values).get("auc")
        has = ok & (d["presser"] >= 0) & (d["PFF_type"] != "N")
        pres = (self.m.player_id[d.loc[has, "presser"].values] == d.loc[has, "pid_pred"].values).mean()
        near = self._nearest_baseline(d[has])
        f = self._run_fouls()
        fa = binary_report(f["PFF_foul"].values.astype(int), f["p_foul"].values).get("auc")
        nc = f["moment"] != "challenge"
        oa = binary_report(f.loc[nc, "PFF_offside"].values.astype(int), f.loc[nc, "p_offside"].values).get("auc")
        res = pd.DataFrame([
            ("Pressure type accuracy", acc, base, "always 'no pressure'"),
            ("Pressured or not (AUC)", auc, 0.5, "coin flip"),
            ("Who is pressing (top-1)", pres, near, "nearest opponent"),
            ("Foul given (AUC)", fa, 0.5, "coin flip"),
            ("Offside (AUC)", oa, 0.5, "coin flip"),
        ], columns=["question", "model", "baseline", "baseline is"])
        res["model"] = (res["model"] * 100).round(1).astype(str) + "%"
        res["baseline"] = (res["baseline"] * 100).round(1).astype(str) + "%"
        print(f"Match {self.gid}: {int(ok.sum())} pressure moments, {len(f)} events - never used in training")
        return res

    def _nearest_baseline(self, d):
        m = self.m
        hit = []
        for i, a, p in zip(d["idx"].values, d["actor"].values, d["presser"].values):
            dd = np.linalg.norm(m.pos[i] - m.pos[i, a], axis=1)
            dd[(m.team == m.team[a]) | ~np.isfinite(dd)] = np.inf
            hit.append(int(np.argmin(dd)) == p)
        return float(np.mean(hit)) if hit else np.nan

    def pressure_moments(self, only=None, n=30):
        """only: 'right', 'wrong', 'pressured' or None.  Use the row number with show_pressure()."""
        d = self._run_pressure()
        cols = ["time", "moment", "player", "PFF_type", "model_type", "model_confidence", "PFF_presser", "model_presser"]
        if only == "right":
            d = d[(d["PFF_type"] == d["model_type"]) & (d["PFF_type"] != "N")]
        elif only == "wrong":
            d = d[d["PFF_type"] != d["model_type"]]
        elif only == "pressured":
            d = d[d["PFF_type"] == "P"]
        return d[cols].head(n)

    def show_pressure(self, row):
        """Pitch picture of one pressure moment (row number from pressure_moments())."""
        d = self._run_pressure().loc[row]
        m = self.m
        i, a = int(d["idx"]), int(d["actor"])
        sign = float(m.attack[i] * (1 if m.team[a] == 0 else -1))
        P, V, B = m.pos[i] * sign, m.vel[i] * sign, m.ball[i, :2] * sign
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        draw_pitch(ax, m.half_len, m.half_wid)
        for s in range(m.n_players):
            if np.isfinite(P[s, 0]):
                c = BLUE if m.team[s] == m.team[a] else ORANGE
                ax.scatter(*P[s], s=150, color=c, edgecolor="white", linewidth=1.5, zorder=3)
                if np.isfinite(V[s, 0]):
                    ax.annotate("", xy=P[s] + V[s] * 0.6, xytext=P[s],
                                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, alpha=0.85), zorder=2)
        ax.scatter(*P[a], s=420, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
        if np.isfinite(B[0]):
            ax.scatter(*B, s=40, color="white", edgecolor=INK, zorder=5)
        p_true = int(d["presser"])
        if p_true >= 0:
            ax.scatter(*P[p_true], s=420, facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=4)
        lut = {int(pid): s for s, pid in enumerate(m.player_id)}
        p_pred = lut.get(int(d["pid_pred"]), -1) if d["pid_pred"] and d["pid_pred"] > 0 else -1
        if p_pred >= 0:
            ax.plot(*zip(P[a], P[p_pred]), color=INK, linewidth=1.4, linestyle="--", zorder=1)
        for s in {a, p_true, p_pred} - {-1}:
            ax.annotate(self.name(s), xy=P[s], xytext=P[s] + np.array([0, 1.8]), ha="center", fontsize=9, color=INK)
        cx, cy = P[a]
        ax.set_xlim(cx - 15, cx + 15)
        ax.set_ylim(cy - 9.5, cy + 9.5)
        right = "correct" if d["PFF_type"] == d["model_type"] else "wrong"
        ax.set_title(f"{d['time']}  {d['player']} - {d['moment']}   |   PFF: {PNAME.get(d['PFF_type'], d['PFF_type'])}   "
                     f"model: {PNAME.get(d['model_type'], d['model_type'])} ({d['model_confidence']:.0%}, {right})",
                     fontsize=10.5)
        fig.text(0.04, 0.04, f"PFF pressing player: {d['PFF_presser']}  (orange ring)     "
                             f"model's pressing player: {d['model_presser']}  (dashed line)\n"
                             "Black ring = player on the ball.  Blue = his team, orange = opponents.  "
                             "Arrows = movement in the next 0.6 s.", fontsize=9, color=INK2, va="top")
        plt.show()

    # ------------------------------------------------------------------ fouls / offside
    def _run_fouls(self):
        if self._fouls is not None:
            return self._fouls
        m = self.m
        fr = F.foul_rows(m)
        fo = self.m5.fouls(m, fr["idx"].values, m.player_id[np.clip(fr["actor"].values, 0, None)], fr["pe_type"].values)
        self._fouls = pd.DataFrame(dict(
            time=[self._clock(i) for i in fr["idx"].values], idx=fr["idx"].values,
            moment=[KIND.get(t, t) for t in fr["pe_type"].values], player=[self.name(s) for s in fr["actor"].values],
            PFF_foul=(fr["foul_type"].values == "I"), p_foul=fo["p_foul_given"].values.round(2),
            PFF_offside=(fr["offense"].values == "O"), p_offside=fo["p_offside"].values.round(2), actor=fr["actor"].values))
        return self._fouls

    def fouls_flagged(self, what="foul", n=20):
        """Events the model is most confident about.  what = 'foul' or 'offside'."""
        f = self._run_fouls()
        col = "p_foul" if what == "foul" else "p_offside"
        return f.sort_values(col, ascending=False)[["time", "moment", "player", "PFF_foul", "p_foul", "PFF_offside",
                                                     "p_offside"]].head(n)

    def show_event(self, row, what="offside"):
        """Pitch picture of a flagged event (row number from fouls_flagged())."""
        f = self._run_fouls().loc[row]
        m = self.m
        i, a = int(f["idx"]), int(f["actor"])
        sign = float(m.attack[i] * (1 if m.team[a] == 0 else -1))
        P, B = m.pos[i] * sign, m.ball[i, :2] * sign
        fig, ax = plt.subplots(figsize=(9.6, 6.4))
        draw_pitch(ax, m.half_len, m.half_wid)
        if what == "offside":
            opp = np.flatnonzero((m.team != m.team[a]) & np.isfinite(P[:, 0]))
            line = max(np.sort(P[opp, 0])[-2], 0.0)
            ax.axvline(line, color=ORANGE, linewidth=1.6)
            ax.text(line + 0.5, m.half_wid - 3, "offside line", color=INK2, fontsize=9)
        for s in range(m.n_players):
            if np.isfinite(P[s, 0]):
                ax.scatter(*P[s], s=130, color=BLUE if m.team[s] == m.team[a] else ORANGE, edgecolor="white", zorder=3)
        ax.scatter(*P[a], s=380, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
        j = m.shift(np.array([i]), 2.0)[0]
        B2 = m.ball[j, :2] * sign
        if np.isfinite(B[0]) and np.isfinite(B2[0]):
            ax.annotate("", xy=B2, xytext=B, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, linestyle="--"), zorder=5)
        p = f["p_offside"] if what == "offside" else f["p_foul"]
        truth = f["PFF_offside"] if what == "offside" else f["PFF_foul"]
        ax.set_title(f"{f['time']}  {f['player']} - {f['moment']}   |   model: {what} {p:.0%}   "
                     f"|   PFF: {'yes' if truth else 'no'}", fontsize=10.5)
        fig.text(0.04, 0.04, "Black ring = player on the ball.  Dashed arrow = ball path over the next 2 s.",
                 fontsize=9, color=INK2, va="top")
        plt.show()

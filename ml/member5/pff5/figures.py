"""
Figures for the Member 5 report.

    python -m pff5.figures --run runs/raw --tables data/tables --compact data/compact \
        --events data/ev_slim --out figures

Every number and every example comes from OUT-OF-FOLD predictions: the model that
produced it never saw the match. The pitch examples are re-scored with the fold
model that held that match out.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import features as F  # noqa: E402
from . import models as M  # noqa: E402
from .data import game_path, load_match  # noqa: E402
from .dataset import load_table, table_games  # noqa: E402

# reference palette (validated): categorical slots 1-3, text and surface tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BASE = "#c9c8c2"          # baseline bars: recessive neutral
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PITCH_LINE = "#b9b8b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 110,
})

TYPE_NAMES = {"A": "Aerial duel", "FIFTY": "50/50 ground duel", "D": "Dribble", "B": "Tackle from behind",
              "T": "Standing tackle", "L": "Slide tackle", "S": "Shoulder to shoulder", "H": "Shielding"}
P_NAMES = ["No pressure (N)", "Attempted (A)", "Passing lane (L)", "Pressured (P)"]


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _roc(y, s):
    y = np.asarray(y).astype(bool)
    o = np.argsort(-np.asarray(s, float), kind="stable")
    y = y[o]
    tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
    fpr = np.r_[0, np.cumsum(~y) / max((~y).sum(), 1)]
    return fpr, tpr, float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# --------------------------------------------------------------------------- #
# 1. headline results vs baselines
# --------------------------------------------------------------------------- #
def fig_results(R, out):
    p, a, f = R["pressure"], R["attributes"], R["fouls"]
    typ = a["possessor_unknown"]["with_true_participants"]["heads"]["type"]
    rows = [
        ("Pressure type (N / A / L / P)\naccuracy", p["pressure_type"]["accuracy"], p["pressure_type"]["majority_baseline_accuracy"],
         "always 'no pressure'"),
        ("Pressured or not\nAUC", p["pressured_vs_not"]["auc"], 0.5, "coin flip"),
        ("Who is pressing\ntop-1 accuracy", p["presser_pick"]["model"]["top1_accuracy"],
         p["presser_pick"]["baseline_nearest_opponent"]["top1_accuracy"], "nearest opponent"),
        ("Aerial duel\nF1", typ["per_class"]["A"]["f1"], None, ""),
        ("Foul given on an event\nAUC", f["foul_given_any_event"]["all_events"]["auc"], 0.5, "coin flip"),
        ("Offside on a pass\nAUC", f["offside"]["non_challenge_events"]["auc"], 0.5, "coin flip"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    y = np.arange(len(rows))[::-1] * 1.0
    h = 0.34
    for i, (name, mv, bv, bname) in enumerate(rows):
        yy = y[i]
        ax.barh(yy + h / 2 + 0.02, mv * 100, height=h, color=BLUE)
        ax.text(mv * 100 + 1, yy + h / 2 + 0.02, f"{mv * 100:.1f}" + ("%" if "F1" not in name else "") if "F1" not in name
                else f"{mv:.2f}", va="center", fontsize=9, color=INK)
        if bv is not None:
            ax.barh(yy - h / 2 - 0.02, bv * 100, height=h, color=BASE)
            ax.text(bv * 100 + 1, yy - h / 2 - 0.02, f"{bv * 100:.1f}%  {bname}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("score (%)  -  F1 shown on a 0-1 scale")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Member 5 model"), Patch(color=BASE, label="simple baseline")],
              loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=9)
    n = R["pressure"]["n_matches"]
    ax.set_title(f"Member 5 models vs simple baselines - {n} matches, scored only on unseen matches")
    _save(fig, out, "01_results_vs_baseline.png")


# --------------------------------------------------------------------------- #
# 2. pressure type confusion matrix
# --------------------------------------------------------------------------- #
def fig_pressure_confusion(run, out):
    o = pd.read_parquet(os.path.join(run, "oof", "pressure.parquet"))
    o = o[o["y"] >= 0]
    cm = np.zeros((4, 4))
    np.add.at(cm, (o["y"].values, o["pred"].values), 1)
    rn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rn[i, j] * 100:.0f}%\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8.5,
                    color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([s.replace(" (", "\n(") for s in P_NAMES], fontsize=8.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(P_NAMES, fontsize=8.5)
    ax.set_xlabel("model says")
    ax.set_ylabel("PFF annotator says")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Pressure type: where the model is right and wrong\n(each row sums to 100%)", fontsize=11)
    _save(fig, out, "02_pressure_type_confusion.png")


# --------------------------------------------------------------------------- #
# 3. ROC curves
# --------------------------------------------------------------------------- #
def fig_roc(run, tables, out):
    po = pd.read_parquet(os.path.join(run, "oof", "pressure.parquet"))
    po = po[po["y"] >= 0]
    fo = pd.read_parquet(os.path.join(run, "oof", "fouls.parquet"))
    fr = load_table(tables, "raw", "foul_rows", games=sorted(fo["gid"].unique()), columns=["gid", "ev_row", "pe_type"])
    fo = fo.merge(fr, on=["gid", "ev_row"], how="left")
    nc = fo["pe_type"] != "CH"
    curves = [("Pressured or not", po["y"].values > 0, 1 - po["p_N"].values, BLUE),
              ("Foul given", fo["y_given"].values, fo["p_given"].values, ORANGE),
              ("Offside (passes and other non-challenge events)", fo["y_off"].values[nc], fo["p_off"].values[nc], AQUA)]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
    ax.text(0.62, 0.55, "coin flip (AUC 50%)", color=INK2, fontsize=8.5, rotation=40)
    for name, y, s, c in curves:
        fpr, tpr, auc = _roc(y, s)
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name}: AUC {auc * 100:.1f}%")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("false alarms (share of normal moments flagged)")
    ax.set_ylabel("found (share of real cases flagged)")
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("How well each yes/no model separates real cases\nfrom normal moments (higher curve = better)", fontsize=11)
    _save(fig, out, "03_roc_curves.png")


# --------------------------------------------------------------------------- #
# 4. challenge type per class
# --------------------------------------------------------------------------- #
def fig_challenge_types(R, out):
    pc = R["attributes"]["possessor_unknown"]["with_true_participants"]["heads"]["type"]["per_class"]
    items = sorted(pc.items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    yy = np.arange(len(items))
    ax.barh(yy, [v["f1"] for _, v in items], height=0.62, color=BLUE)
    for i, (k, v) in enumerate(items):
        ax.text(v["f1"] + 0.01, i, f"{v['f1']:.2f}   (n={v['support']:,})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([TYPE_NAMES.get(k, k) for k, _ in items])
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("F1 score (0 = useless, 1 = perfect)")
    ax.set_title("Challenge type: aerial duels are clearly visible in tracking\n"
                 "(ball height); tackle technique is not", fontsize=11)
    _save(fig, out, "04_challenge_type_f1.png")


# --------------------------------------------------------------------------- #
# 5. what a foul looks like in tracking
# --------------------------------------------------------------------------- #
def fig_foul_signal(tables, games, out):
    cols = ["gid"] + [f"post_mean_speed_{t}" for t in (0, 1, 2, 3, 5)]
    fe = load_table(tables, "raw", "foul_ev", games=games, columns=cols)
    fr = load_table(tables, "raw", "foul_rows", games=games, columns=["gid", "foul_type"])
    is_f = (fr["foul_type"] == "I").values
    t = [0, 1, 2, 3, 5]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for mask, name, c in ((~is_f, f"no foul ({(~is_f).sum():,} events)", BLUE), (is_f, f"foul given ({is_f.sum():,} events)", ORANGE)):
        v = [np.nanmean(fe[f"post_mean_speed_{s}"].values[mask]) for s in t]
        ax.plot(t, v, color=c, linewidth=2, marker="o", markersize=6, label=name)
        ax.text(5.15, v[-1], name.split(" (")[0], va="center", color=INK2, fontsize=9)
    ax.set_xlim(-0.2, 6.3)
    ax.set_ylim(0, None)
    ax.set_xticks(t)
    ax.set_xlabel("seconds after the event")
    ax.set_ylabel("average speed of all 22 players (m/s)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Why the foul model works: after a whistle the players stop\n"
                 "(one of the signals the model learns from tracking)", fontsize=11)
    _save(fig, out, "05_foul_signal.png")


# --------------------------------------------------------------------------- #
# pitch helpers
# --------------------------------------------------------------------------- #
def draw_pitch(ax, hl=52.5, hw=34.0):
    kw = dict(color=PITCH_LINE, linewidth=1, zorder=0)
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], **kw)
    ax.plot([0, 0], [-hw, hw], **kw)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, **kw))
    for s in (-1, 1):
        x0 = s * hl
        ax.plot([x0, x0 - s * 16.5, x0 - s * 16.5, x0], [-20.16, -20.16, 20.16, 20.16], **kw)
        ax.plot([x0, x0 - s * 5.5, x0 - s * 5.5, x0], [-9.16, -9.16, 9.16, 9.16], **kw)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("#f4f6f1")


def _names(events_dir, gid):
    p = os.path.join(events_dir, f"{gid}_events.json.gz")
    names = {}
    if not os.path.exists(p):
        return names
    for e in json.load(gzip.open(p, "rt")):
        for part in ("gameEvents", "possessionEvents", "initialTouch", "fouls"):
            d = e.get(part) or {}
            for k, v in d.items():
                if k.endswith("PlayerId") and v is not None:
                    nm = d.get(k[:-2] + "Name")
                    if nm:
                        names[int(v)] = nm
    return names


def _fold_model(run, name, gid, games):
    import lightgbm as lgb
    folds = dict(zip(games, M.fold_of(np.array(games))))
    meta = json.load(open(os.path.join(run, "models", f"{name}.txt.json")))
    return lgb.Booster(model_file=os.path.join(run, "models", f"{name}.fold{folds[gid]}.txt")), meta


# --------------------------------------------------------------------------- #
# 6. pressure example on the pitch
# --------------------------------------------------------------------------- #
def fig_pressure_example(run, tables, compact, events_dir, games, out):
    o = pd.read_parquet(os.path.join(run, "oof", "pressure.parquet"))
    # a pass under pressure, model right, where the NEAREST opponent is not the presser
    good = o[(o["kind"] == 1) & (o["y"] == 3) & (o["pred"] == 3) & (o["presser"] >= 0) &
             (o["pred_presser"] == o["presser"]) & (o["p_P"] > 0.6)]
    best = None
    for gid in sorted(good["gid"].unique()):
        pc = load_table(tables, "raw", "press_cand", games=[gid])
        keys = set(pc[(pc["rank"] == 0) & (~pc["y"])]["key"]) & set(good.loc[good["gid"] == gid, "key"])
        if not keys:
            continue
        sub = pc[pc["key"].isin(keys)].copy()
        bst, meta = _fold_model(run, "pressure_presser", gid, games)
        sub["score"] = bst.predict(sub.reindex(columns=meta["features"]).to_numpy(np.float32))
        for key, g in sub.groupby("key"):
            sp = g.loc[g["y"], "score"].max()
            sn = g.loc[g["rank"] == 0, "score"].max()
            dn, dp = g.loc[g["rank"] == 0, "d_0"].max(), g.loc[g["y"], "d_0"].max()
            # clear margin, and the two opponents not on top of each other
            quality = (sp - sn) if (dp - dn) > 0.8 and dp < 5 else -1
            if best is None or quality > best[0]:
                best = (quality, gid, key, g)
        if best and best[0] > 0.5:
            break
    _, gid, key, pc = best
    pick = o[o["key"] == key].iloc[0]
    pe = load_table(tables, "raw", "press_ev", games=[gid])
    ev = pe[pe["key"] == key].iloc[0]
    score = pc["score"].to_numpy()
    m = load_match(game_path(compact, gid), with_events=True, ball_source="raw")
    names = _names(events_dir, gid)
    i, actor = int(ev["idx"]), int(ev["actor"])
    pe_type = m.events.loc[int(ev["ev_row"]), "pe_possessionEventType"]
    sign = float(m.attack[i] * (1 if m.team[actor] == 0 else -1))      # actor's team attacks to the right
    P, V, B = m.pos[i] * sign, m.vel[i] * sign, m.ball[i, :2] * sign
    cx, cy = P[actor]
    rx, ry = 15.0, 9.5
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    draw_pitch(ax, m.half_len, m.half_wid)
    team_a = m.team[actor]
    for s in range(m.n_players):
        if not np.isfinite(P[s, 0]):
            continue
        c = BLUE if m.team[s] == team_a else ORANGE
        ax.scatter(P[s, 0], P[s, 1], s=150, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        if np.isfinite(V[s, 0]):
            ax.annotate("", xy=(P[s, 0] + V[s, 0] * 0.6, P[s, 1] + V[s, 1] * 0.6), xytext=(P[s, 0], P[s, 1]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, alpha=0.85), zorder=2)
    ax.scatter(P[actor, 0], P[actor, 1], s=420, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
    true_p = int(ev["presser"])
    near = int(pc.loc[pc["rank"] == 0, "cand_slot"].iloc[0])
    ax.scatter(P[true_p, 0], P[true_p, 1], s=420, facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=4)
    nm = lambda s: names.get(int(m.player_id[s]), f"#{m.jersey[s]}")
    dist = lambda s: float(np.linalg.norm(P[s] - P[actor]))
    # label the candidates inside the view, pushed away from the player on the ball
    for k in np.argsort(-score)[:4]:
        s = int(pc["cand_slot"].iloc[k])
        x, y = P[s]
        if abs(x - cx) > rx or abs(y - cy) > ry:
            continue
        d = np.array([x - cx, y - cy])
        d = d / (np.linalg.norm(d) + 1e-6)
        tx, ty = x + 2.6 * d[0], y + 2.6 * d[1] + 0.6
        ax.annotate(f"{nm(s)}\nscore {score[k] * 100:.0f}%", xy=(x, y), xytext=(tx, ty), fontsize=8.5, ha="center",
                    color=INK, weight="bold" if s == true_p else "normal",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=6)
    ax.annotate(f"{nm(actor)}\n(on the ball)", xy=(cx, cy), xytext=(cx - 3.5, cy - 3.2), fontsize=8.5, ha="center",
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=6)
    ax.set_xlim(cx - rx, cx + rx)
    ax.set_ylim(cy - ry, cy + ry)
    probs = {c: float(pick[f"p_{c}"]) for c in "NALP"}
    kind = {"PA": "pass", "BC": "ball carry", "SH": "shot", "CR": "cross", "CL": "clearance", "TC": "touch",
            "RE": "rebound"}.get(pe_type, pe_type)
    ax.set_title(f"Pressure on {nm(actor)}'s {kind}  (match {gid}, a match this model never saw)", fontsize=11)
    note = (f"Model: pressure type = Pressured (P) with {probs['P'] * 100:.0f}%   "
            f"[N {probs['N'] * 100:.0f}%, A {probs['A'] * 100:.0f}%, L {probs['L'] * 100:.0f}%]  -  PFF says P.\n"
            f"Pressing player: the model picks {nm(true_p)} ({dist(true_p):.1f} m away) - correct.\n"
            f"The nearest opponent, {nm(near)} ({dist(near):.1f} m), was NOT the presser, so 'nearest opponent' gets this one wrong.\n"
            "Black ring = player on the ball.  Orange ring = pressing player according to PFF.  "
            "Arrows = where each player moves in the next 0.6 s.")
    fig.text(0.04, 0.04, note, fontsize=9, color=INK2, va="top")
    _save(fig, out, "06_pressure_example.png")
    return gid


# --------------------------------------------------------------------------- #
# 7. offside example on the pitch
# --------------------------------------------------------------------------- #
def fig_offside_example(run, tables, compact, events_dir, games, out):
    fo = pd.read_parquet(os.path.join(run, "oof", "fouls.parquet"))
    fr = load_table(tables, "raw", "foul_rows", games=sorted(fo["gid"].unique()),
                    columns=["gid", "ev_row", "idx", "actor", "pe_type"])
    fo = fo.merge(fr, on=["gid", "ev_row"])
    cand = fo[(fo["y_off"] == 1) & (fo["pe_type"] == "PA") & (fo["actor"] >= 0)].sort_values("p_off", ascending=False)
    row, m = None, None
    for _, r in cand.head(40).iterrows():          # most confident true offsides with a visible margin
        mm = load_match(game_path(compact, int(r["gid"])), with_events=False, ball_source="raw")
        ii, aa = int(r["idx"]), int(r["actor"])
        sg = float(mm.attack[ii] * (1 if mm.team[aa] == 0 else -1))
        PP = mm.pos[ii] * sg
        op = np.flatnonzero((mm.team != mm.team[aa]) & np.isfinite(PP[:, 0]))
        ln = max(np.sort(PP[op, 0])[-2], 0.0)
        jj = mm.shift(np.array([ii]), 2.0)[0]
        sn, _, tm = F._nearest_to_ball(mm, np.array([jj]))
        if sn[0] >= 0 and tm[0] == mm.team[aa] and 1.5 < PP[sn[0], 0] - ln < 8 and PP[aa, 0] < ln - 5:
            row, m = r, mm
            break
    if row is None:
        row = cand.iloc[0]
        m = load_match(game_path(compact, int(row["gid"])), with_events=False, ball_source="raw")
    gid, i, actor = int(row["gid"]), int(row["idx"]), int(row["actor"])
    names = _names(events_dir, gid)
    team_a = m.team[actor]
    sign = float(m.attack[i] * (1 if team_a == 0 else -1))
    P, B = m.pos[i] * sign, m.ball[i, :2] * sign
    opp = np.flatnonzero((m.team != team_a) & np.isfinite(P[:, 0]))
    xs = np.sort(P[opp, 0])
    line = max(xs[-2], B[0] if np.isfinite(B[0]) else P[actor, 0], 0.0)
    j = m.shift(np.array([i]), 2.0)[0]
    s_near, _, team = F._nearest_to_ball(m, np.array([j]))
    recv = int(s_near[0])
    B2 = m.ball[j, :2] * sign
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    draw_pitch(ax, m.half_len, m.half_wid)
    ax.axvspan(line, m.half_len, color=ORANGE, alpha=0.08, zorder=0)
    ax.axvline(line, color=ORANGE, linewidth=1.6, zorder=1)
    ax.text(line + 0.5, m.half_wid - 2.5, "offside line\n(2nd-last defender)", color=INK2, fontsize=9)
    for s in range(m.n_players):
        if np.isfinite(P[s, 0]):
            c = BLUE if m.team[s] == team_a else ORANGE
            ax.scatter(P[s, 0], P[s, 1], s=110, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter(P[actor, 0], P[actor, 1], s=320, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
    if recv >= 0 and m.team[recv] == team_a:
        ax.scatter(P[recv, 0], P[recv, 1], s=320, facecolor="none", edgecolor=BLUE, linewidth=2.2, zorder=4)
        ax.annotate(f"{P[recv, 0] - line:+.1f} m beyond", xy=(P[recv, 0], P[recv, 1]), xytext=(P[recv, 0] + 2, P[recv, 1] - 4),
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    if np.isfinite(B[0]) and np.isfinite(B2[0]):
        ax.annotate("", xy=B2, xytext=B, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, linestyle="--"), zorder=5)
    ax.set_xlim(min(-2, P[actor, 0] - 4), m.half_len + 2)
    ax.set_ylim(-m.half_wid - 2, m.half_wid + 2)
    nm = lambda s: names.get(int(m.player_id[s]), f"#{m.jersey[s]}")
    ax.set_title(f"Offside on {nm(actor)}'s pass\n(match {gid}, a match this model never saw)", fontsize=11)
    note = (f"Model: offside probability {row['p_off'] * 100:.0f}%  -  PFF annotators: offside.\n"
            "Black ring = passer.  Blue ring = team-mate nearest the ball 2 s later.  Dashed arrow = ball path over 2 s.\n"
            "Features the model uses: the offside line, how far each team-mate is beyond it, who receives the ball, "
            "and whether play stops.")
    fig.text(0.04, 0.06, note, fontsize=9, color=INK2, va="top")
    _save(fig, out, "07_offside_example.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="e.g. runs/raw")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--events", default=None, help="folder with <id>_events.json.gz (player names)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default="1,2,3,4,5,6,7")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    R = {t: json.load(open(os.path.join(a.run, "results", f"{t}.json"))) for t in ("pressure", "attributes", "fouls")}
    held = set(R["pressure"].get("held_out_matches") or [])
    games = [g for g in table_games(a.tables, "raw") if g not in held]
    ev_dir = a.events or a.compact
    todo = set(a.only.split(","))
    if "1" in todo:
        fig_results(R, a.out)
    if "2" in todo:
        fig_pressure_confusion(a.run, a.out)
    if "3" in todo:
        fig_roc(a.run, a.tables, a.out)
    if "4" in todo:
        fig_challenge_types(R, a.out)
    if "5" in todo:
        fig_foul_signal(a.tables, games, a.out)
    if "6" in todo:
        fig_pressure_example(a.run, a.tables, a.compact, ev_dir, games, a.out)
    if "7" in todo:
        fig_offside_example(a.run, a.tables, a.compact, ev_dir, games, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

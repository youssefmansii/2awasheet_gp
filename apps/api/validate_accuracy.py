"""
validate_accuracy.py  -  "are the numbers right?" test suite for ANY match
==========================================================================

Run (from apps/api, venv active):

    python validate_accuracy.py --match 3816
    python validate_accuracy.py --match 3816 3817 3820        # several matches, one summary at the end
    python validate_accuracy.py --match 3816 --quick          # skips the slowest checks

It complements validate_metrics.py (which checks the maths against the reference code and the predictive power).
This file asks the simple questions a reader of the dashboard would ask:

  1. Maths      do the models obey rules that must always hold?   (probabilities sum to 1, mirror the pitch -> same answer,
                                                                   a closer attacker never lowers his team's control)
  2. Outcomes   do the numbers predict what really happened?      (pitch control at the pass target vs. completed passes,
                                                                   pass model, pressure vs. success, shots vs. xG)
  3. Balance    is nothing systematically too high / too low?     (value of an average action ~ 0, penalty value, calibration)
  4. Physics    are speeds, distances and fatigue minutes plausible for football?
  5. Data       are events and tracking in step, are the real goals found?
  6. Repeat     does a second run give exactly the same numbers?
  7. Attacks    is every shot and goal inside exactly one attack, is the 1-100 danger score sensible and stable?

Every result is PASS / WARN / FAIL with the number behind it. Where it matters a 95 % interval (bootstrap) is printed, because
one match has few shots and few lost passes: a WARN there means "cannot be shown with this much data", not "wrong".

About match 3816: the pass-completion model (assets/pass_model_fitted.json) was fitted on 3816, so on 3816 its scores are
"in-sample" (optimistic). On ANY OTHER match the same checks are a genuine out-of-sample test - those are the numbers to trust.

Results are also saved to data/interim/accuracy_<match>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from data_loader import default_data_dir, load_match
from metrics_engine import PENALTY_XG, MetricsEngine, PitchControlParams, compute_pitch_control, xg_surface
from player_analytics import PlayerAnalytics, _wilson
from validate_metrics import KNOWN_GOALS, Report, check_facts, rank_auc

TRAINED_ON = {"3816"}  # matches the shipped pass model has seen


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def boot_ci(stat, n, rng, reps=400):
    """95 % bootstrap interval of stat(index_array) over n rows."""
    vals = []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        v = stat(ix)
        if np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (float("nan"), float("nan"))


def ece(p, y, bins=8):
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def live_frames(md, step_s, rng, n):
    """Random frames of live play with every player tracked."""
    ok = np.flatnonzero(np.isfinite(md.ball_x))
    ok = ok[:: max(1, int(step_s * md.fps))]
    ok = [i for i in ok if len(md.present_slots(int(i))) >= 20]
    return [int(i) for i in rng.choice(ok, size=min(n, len(ok)), replace=False)]


def frame_inputs(eng, idx):
    m = eng.m
    slots = m.present_slots(idx)
    is_home = m.players["side"].to_numpy()[slots] == "home"
    side = eng.possession_side(idx) or "home"
    is_att = is_home if side == "home" else ~is_home
    pos = np.column_stack([m.X[idx, slots], m.Y[idx, slots]]).astype(float)
    vel = np.column_stack([m.VX[idx, slots], m.VY[idx, slots]]).astype(float)
    acc = np.column_stack([m.AX[idx, slots], m.AY[idx, slots]]).astype(float)
    ball = eng.ball_xy(idx)
    return slots, is_home, is_att, pos, vel, acc, ball, eng._goalkeepers(idx, pos, is_home)


# --------------------------------------------------------------------------------------
# 1. maths
# --------------------------------------------------------------------------------------
def check_maths(md, eng, rep, rng, n_frames):
    M = "PitchCtl"
    P = eng.pc_params
    frames = live_frames(md, 20, rng, n_frames)
    worst_sum, worst_mirror, monotone_ok, monotone_n = 0.0, 0.0, 0, 0
    for idx in frames:
        slots, is_home, is_att, pos, vel, acc, ball, gk = frame_inputs(eng, idx)
        if ball is None or is_att.sum() < 3 or (~is_att).sum() < 3:
            continue
        pa, pdf, _ = compute_pitch_control(pos, vel, acc, is_att, ball, md.grid, P, is_gk=gk)
        worst_sum = max(worst_sum, float(np.abs(pa + pdf - 1).max()), float(max(0 - pa.min(), pa.max() - 1, 0)))
        # mirror the whole pitch through the centre spot: the picture must simply be turned round
        pa2, _, _ = compute_pitch_control(-pos, -vel, -acc, is_att, -ball, md.grid, P, is_gk=gk)
        worst_mirror = max(worst_mirror, float(np.abs(pa2 - pa[::-1, ::-1]).max()))
        # a closer attacker can never lower his team's control of a point: move the nearest attacker 2 m towards a point
        tgt = ball + np.array([10.0, 0.0])
        tgt = np.clip(tgt, [-50, -32], [50, 32])
        d = np.hypot(pos[:, 0] - tgt[0], pos[:, 1] - tgt[1])
        a_idx = np.flatnonzero(is_att)
        k = a_idx[int(np.argmin(d[a_idx]))]
        before, _, _ = compute_pitch_control(pos, vel, acc, is_att, ball, md.grid, P, points=tgt[None, :], is_gk=gk)
        pos2 = pos.copy()
        step = tgt - pos[k]
        pos2[k] += 2.0 * step / max(np.linalg.norm(step), 1e-6)
        after, _, _ = compute_pitch_control(pos2, vel, acc, is_att, ball, md.grid, P, points=tgt[None, :], is_gk=gk)
        monotone_n += 1
        monotone_ok += int(after[0] >= before[0] - 1e-9)
    rep.add(M, "attackers' + defenders' control = 1 and stays inside 0..1 (worst error)", "PASS" if worst_sum < 1e-6 else "FAIL", f"{worst_sum:.1e}", "< 1e-6")
    rep.add(M, "turn the pitch round 180 deg -> the map turns round with it (worst difference)", "PASS" if worst_mirror < 1e-9 else "FAIL", f"{worst_mirror:.1e}", "< 1e-9")
    rep.add(M, "moving an attacker 2 m closer never lowers his team's control there", "PASS" if monotone_ok == monotone_n and monotone_n else "FAIL",
            f"{monotone_ok}/{monotone_n} frames", "all frames")

    E = "EPV"
    xx, yy = md.grid.mesh()
    rep.add(E, "value at a cell centre equals the grid value (smooth interpolation keeps the table)",
            "PASS" if np.abs(eng.epv.value(xx, yy, 1) - eng.epv.surface(1)).max() < 1e-9 else "FAIL", "max diff 0", "< 1e-9")
    px, py = rng.uniform(-50, 50, 500), rng.uniform(-30, 30, 500)
    mir = float(np.abs(eng.epv.value(px, py, 1) - eng.epv.value(-px, -py, -1)).max())
    rep.add(E, "attacking left->right equals attacking right->left on the mirrored point (worst difference)", "PASS" if mir < 1e-9 else "FAIL", f"{mir:.1e}", "< 1e-9")
    g = eng.epv.surface(1)
    near = lambda x, y: g[md.grid.cell_index(x, y)]  # the old look-up
    jump_old = max(float(np.abs(near(px + 0.5, py) - near(px, py)).max()), 1e-12)
    jump_new = float(np.abs(eng.epv.value(px + 0.5, py, 1) - eng.epv.value(px, py, 1)).max())
    rep.add(E, "value changes smoothly: biggest jump when the ball moves 0.5 m (new vs. old table look-up)", "PASS" if jump_new < jump_old else "WARN",
            f"{jump_new:.4f} vs {jump_old:.4f}", "new < old")
    xg = xg_surface(md.grid, 1)
    c = lambda x, y: xg[md.grid.cell_index(x, y)]
    ok = c(46, 0) > c(36, 0) > c(26, 0) and c(46, 0) > c(46, 20)
    rep.add("xG", "closer to goal and more central is worth more", "PASS" if ok else "FAIL", f"{c(46, 0):.2f} > {c(36, 0):.2f} > {c(26, 0):.2f}", "monotone")
    rep.add("xG", "penalty value is in the published range (0.74 - 0.82)", "PASS" if 0.74 <= PENALTY_XG <= 0.82 else "FAIL", f"{PENALTY_XG:.2f}", "0.74 - 0.82")
    from metrics_engine import XG_V2, xg_v2_features, xg_v2_from_features
    if XG_V2 is not None:
        gk = np.array([-0.5, 0.0])  # keeper on his line
        f = lambda d, y=0.0, head=False, blk=0: xg_v2_from_features(xg_v2_features(np.array([-d, y]), np.array([[-d / 2, y / 2]] * blk).reshape(-1, 2), gk, head, False))
        ok2 = f(6) > f(11) > f(17) > f(25) and f(11) > f(11, 15) and f(11) > f(11, head=True) and f(11) > f(11, blk=2)
        rep.add("xG", "model v2 behaves like football: closer, central, foot (not head), free path to goal = higher xG", "PASS" if ok2 else "FAIL",
                f"6 m {f(6):.2f}, pen spot {f(11):.2f}, 17 m {f(17):.2f}, 25 m {f(25):.2f}; header {f(11, head=True):.2f}; 2 blockers {f(11, blk=2):.2f}", "monotone")


# --------------------------------------------------------------------------------------
# 2./3. outcomes and balance
# --------------------------------------------------------------------------------------
def pass_rows(md, eng):
    out = []
    ev = md.events
    for r in ev[(ev["poss_type"] == "PA") & (ev["frame_idx"] >= 0) & ev["side"].isin(["home", "away"])].itertuples():
        fi = int(r.frame_idx)
        ts = int(r.target_slot) if r.target_slot >= 0 else int(r.receiver_slot)
        if ts < 0 or not np.isfinite(md.X[fi, ts]) or not np.isfinite(r.ball_x) or not np.isfinite(r.ball_y):
            continue
        out.append((fi, r.side, ts, np.array([r.ball_x, r.ball_y], float), r.pass_outcome == "C"))
    return out


def check_outcomes(md, eng, rep, rng, quick, args_data_dir=None):
    rows = pass_rows(md, eng)
    y = np.array([r[4] for r in rows], float)
    n = len(y)
    if n < 100 or y.sum() < 20 or (n - y.sum()) < 20:
        rep.skip("PitchCtl", "pitch control at the pass target predicts completed passes", f"only {n} usable passes")
    else:
        pcs = []
        for fi, side, ts, ball, ok in rows:
            slots, is_home, is_att, pos, vel, acc, _, gk = frame_inputs(eng, fi)
            is_att = is_home if side == "home" else ~is_home
            adir = md.attack_direction(side, fi)
            if eng.pc_params.offsides and (~is_att).sum() >= 2:
                keep = eng._onside(pos, is_att, adir, ball) | (slots == ts)
                slots, is_att, pos, vel, acc, gk = slots[keep], is_att[keep], pos[keep], vel[keep], acc[keep], gk[keep]
            tp = np.array([[md.X[fi, ts], md.Y[fi, ts]]])
            pa, _, _ = compute_pitch_control(pos, vel, acc, is_att, ball, md.grid, eng.pc_params, points=tp, is_gk=gk)
            pcs.append(float(pa[0]))
        p = np.clip(np.array(pcs), 1e-4, 1 - 1e-4)
        lo, hi = boot_ci(lambda ix: rank_auc(y[ix], p[ix]), n, rng)
        a = rank_auc(y, p)
        rep.add("PitchCtl", "control at the pass target is higher for completed than for lost passes (AUC, 95% interval)",
                "PASS" if lo > 0.5 else ("WARN" if a > 0.5 else "FAIL"), f"{a:.2f} [{lo:.2f}, {hi:.2f}] n={n}", "interval above 0.5")
        print("        info   control is a good RANKER of passes (AUC) but only a weak raw probability (Brier skill near 0): read it as 'who is\n"
              "               likely to get there first', not as an exact completion chance. The pass model is the one that gives probabilities.")
        base = np.mean((y.mean() - y) ** 2)
        skill = 1 - np.mean((p - y) ** 2) / base
        slo, shi = boot_ci(lambda ix: 1 - np.mean((p[ix] - y[ix]) ** 2) / np.mean((y[ix].mean() - y[ix]) ** 2), n, rng)
        rep.add("PitchCtl", "as a probability it beats 'always say the average' (Brier skill, 95% interval)",
                "PASS" if slo > 0 else ("WARN" if skill > 0 else "FAIL"), f"{skill:+.3f} [{slo:+.3f}, {shi:+.3f}]", "interval above 0")

    # pass model
    ps = []
    ys = []
    for fi, side, ts, ball, ok in rows:
        tgt = np.array([[md.X[fi, ts], md.Y[fi, ts]]])
        ps.append(float(eng.pass_probabilities(fi, side, ball, tgt)[0]))
        ys.append(float(ok))
    ps, ys = np.clip(np.array(ps), 1e-4, 1 - 1e-4), np.array(ys)
    where = "in-sample, optimistic" if str(md.match_id) in TRAINED_ON else "OUT-OF-SAMPLE, the honest test"
    if len(ys) >= 100 and 20 <= ys.sum() <= len(ys) - 20:
        lo, hi = boot_ci(lambda ix: rank_auc(ys[ix], ps[ix]), len(ys), rng)
        a = rank_auc(ys, ps)
        rep.add("PassModel", f"pass model separates completed from lost passes (AUC; {where})", "PASS" if lo > 0.65 else ("WARN" if a > 0.6 else "FAIL"),
                f"{a:.2f} [{lo:.2f}, {hi:.2f}]", "interval above 0.65")
        e_ = ece(ps, ys)
        rep.add("PassModel", f"predicted pass-success rates match what happened (calibration error; {where})", "PASS" if e_ < 0.05 else ("WARN" if e_ < 0.08 else "FAIL"),
                f"{e_:.3f}", "< 0.05")
        bias = float((ys - ps).mean())
        se = float(np.std(ys - ps) / np.sqrt(len(ys)))
        rep.add("PassModel", "no systematic bias: mean(actual - predicted) within 2 standard errors of 0", "PASS" if abs(bias) <= 2 * se else "WARN",
                f"{bias:+.4f} (+-{se:.4f})", "|z| <= 2")

    # shots vs xG
    ev = md.events
    sh = ev[(ev["poss_type"] == "SH") & (ev["frame_idx"] >= 0) & ev["side"].isin(["home", "away"]) & np.isfinite(ev["ball_x"])]
    xg_sum, goals, var = 0.0, 0, 0.0
    for r in sh.itertuples():
        if getattr(r, "setpiece_type", None) == "P" or (r.shot_outcome == "G" and not bool(r.is_goal)):
            continue  # penalties and goals ruled out are not open-play chances
        q = float(eng.shot_xg(ev.index.get_loc(r.Index)))
        xg_sum += q
        var += q * (1 - q)
        goals += int(bool(r.is_goal))
    if len(sh) >= 10:
        z = (goals - xg_sum) / max(np.sqrt(var), 1e-6)
        rep.add("xG", "open-play goals vs. expected goals (z-score of the difference; few shots -> wide)", "PASS" if abs(z) < 2 else "WARN",
                f"{goals} goals, {xg_sum:.1f} expected, z={z:+.1f}", "|z| < 2")

    # xG model v2: its held-out test (from the model file) and, when available, agreement with StatsBomb's xG for this match
    from metrics_engine import XG_V2
    if XG_V2 is not None:
        v = XG_V2.get("validation", {})
        mine, old, sbx = v.get("this_model", {}), v.get("previous_distance_angle_model", {}), v.get("statsbomb_xg_reference", {})
        rep.add("xG", "xG model v2 on held-out World Cup 2022 shots (AUC; log loss) vs old model and StatsBomb", "PASS" if mine.get("auc", 0) > old.get("auc", 1) else "WARN",
                f"{mine.get('auc')}; {mine.get('log_loss')} | old {old.get('auc')} | SB {sbx.get('auc')}", "better than old")
    bench = (Path(args_data_dir) if args_data_dir else default_data_dir()) / "interim" / f"sb_{md.match_id}_shots.csv"
    if bench.exists():
        import unicodedata
        norm = lambda t: unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower().replace("-", " ")
        sbd = pd.read_csv(bench)
        sbd["n"] = sbd["player"].map(norm)
        pairs = []
        for r in sh.itertuples():
            if getattr(r, "setpiece_type", None) == "P":
                continue
            mn = float(md.match_minute[int(r.frame_idx)])
            team = md.teams[r.side]["name"]
            c = sbd[(sbd["team"] == team) & ((sbd["minute"] + sbd["second"] / 60 - mn).abs() < 1.6) & sbd["n"].str.contains(norm(r.player_name).split()[-1])]
            if len(c):
                pairs.append((float(eng.shot_xg(ev.index.get_loc(r.Index))), float(c["sb_xg"].iloc[0])))
        if len(pairs) >= 8:
            a_, b_ = np.array(pairs).T
            rho = float(np.corrcoef(a_, b_)[0, 1])
            rep.add("xG", "this match: our shot xG agrees with StatsBomb's xG for the same shots (correlation)", "PASS" if rho > 0.7 else "WARN",
                    f"r = {rho:.2f}, {len(pairs)} shots, mean diff {np.mean(np.abs(a_ - b_)):.3f}", "> 0.7")

    # value balance
    av = eng.action_values()
    if len(av):
        mean, se = float(av["vaep"].mean()), float(av["vaep"].std() / np.sqrt(len(av)))
        rep.add("VAEP", "an average action is worth about 0 (raw value change per action)", "PASS" if abs(mean) <= 2 * se else ("WARN" if abs(mean) < 0.01 else "FAIL"),
                f"{mean:+.4f} (+-{se:.4f}) n={len(av)}", "within 2 standard errors of 0")
        if abs(mean) > 2 * se:
            print("        info   raw VAEP is slightly negative on average because losing the ball always costs more than a short pass gains.\n"
                  "               That is why the app ranks players with 'VAEP above average' (centred at 0) - use that one for comparisons.")
        rep.add("VAEP", "above-average VAEP is centred (mean 0)", "PASS" if abs(float(av["vaep_oa"].mean())) < 1e-3 else "FAIL", f"{av['vaep_oa'].mean():+.5f}", "|mean| < 0.001")
        shots = av[av["action"] == "shot"]
        if len(shots):
            goals_v = shots[shots["success"]]
            if len(goals_v):
                rep.add("VAEP", "real goals get a large value (smallest of them)", "PASS" if goals_v["vaep"].min() > 0.5 else "WARN", f"{goals_v['vaep'].min():.2f}", "> 0.5")
            miss = shots[~shots["success"]]
            rep.add("VAEP", "shots that did not score never gain value (largest of them)", "PASS" if (len(miss) == 0 or miss["vaep"].max() <= 0.0) else "WARN",
                    f"{miss['vaep'].max():+.3f}" if len(miss) else "n/a", "<= 0")


# --------------------------------------------------------------------------------------
# pressure, physics, fatigue
# --------------------------------------------------------------------------------------
def check_players(md, eng, rep, rng):
    pa = PlayerAnalytics(eng)
    pr = pa._compute_press()
    if len(pr) > 50:
        rho = spearmanr(pr["nearest_opp_m"], pr["pressure"]).statistic
        rep.add("Pressure", "pressure score falls as the nearest opponent is further away (rank correlation)", "PASS" if rho < -0.6 else "FAIL", f"{rho:+.2f}", "< -0.6")
        rep.add("Pressure", "pressure score stays inside 0..1", "PASS" if pr["pressure"].between(0, 1).all() else "FAIL", f"{pr['pressure'].min():.2f}..{pr['pressure'].max():.2f}", "0..1")
        low, high = pr[pr["pressure"] < 0.25], pr[pr["pressure"] >= 0.6]
        if len(low) >= 20 and len(high) >= 20:
            p1, p2 = low["success"].mean(), high["success"].mean()
            se = np.sqrt(p1 * (1 - p1) / len(low) + p2 * (1 - p2) / len(high))
            z = (p1 - p2) / max(se, 1e-9)
            rep.add("Pressure", "actions under high pressure succeed less often than free ones (difference, z)", "PASS" if z > 1.64 else ("WARN" if p1 > p2 else "FAIL"),
                    f"{p1:.0%} free vs {p2:.0%} pressed, z={z:+.1f}", "z > 1.64")
        lbl = md.events["pressure_type"] if "pressure_type" in md.events.columns else None
        if lbl is not None:
            ev = md.events.loc[pr["event_idx"]]
            lab = ev["pressure_type"].astype(str).str.upper().isin(["P", "A", "L"]).to_numpy().astype(float)
            if 10 <= lab.sum() <= len(lab) - 10:
                a = rank_auc(lab, pr["pressure"].to_numpy())
                rep.add("Pressure", "agrees with the provider's own pressure label (AUC)", "PASS" if a > 0.65 else ("WARN" if a > 0.55 else "FAIL"), f"{a:.2f} n={int(lab.sum())} labelled", "> 0.65")

    # physics
    sp = np.where(md.SPEED <= 10.0, md.SPEED, np.nan)
    top = np.nanmax(sp) * 3.6
    rep.add("Physical", "fastest tracked speed is humanly possible (km/h)", "PASS" if top <= 38.5 else "FAIL", f"{top:.1f}", "<= 38.5")
    acc = np.hypot(md.AX, md.AY)
    q = float(np.nanpercentile(acc, 99.9))
    rep.add("Physical", "99.9 % of accelerations are below 8 m/s2", "PASS" if q < 8.0 else "WARN", f"{q:.2f}", "< 8")
    dist = np.nansum(sp, axis=0) / md.fps / 1000.0
    minutes = np.sum(~np.isnan(md.X), axis=0) / md.fps / 60.0
    is_gk = md.players["position"].astype(str).str.upper().eq("GK").to_numpy()
    full = (minutes > 80) & ~is_gk
    if full.any():
        per90 = dist[full] * 90.0 / minutes[full]
        med = float(np.median(per90))
        # published top-level outfield figures are about 9.5 - 12 km per 90 min. PFF's smoothed positions give less
        # (their smoothing removes part of the sideways movement), so a WARN here is expected and is a KNOWN limitation:
        # trust the numbers for comparing players / halves / matches, not as absolute kilometres.
        rep.add("Physical", "outfield distance per 90 min of tracked time is in the usual range (>= 9 km)", "PASS" if med >= 9.0 else "WARN",
                f"median {med:.1f} km (range {per90.min():.1f} - {per90.max():.1f})", ">= 9 km")
        if med < 9.0:
            print("        info   KNOWN LIMITATION: distance, high-speed-running and sprint totals computed from PFF's smoothed positions are LOWER than\n"
                  "               with lightly smoothed raw positions (about 20-25 % for distance, much more for sprints). Use them for comparisons\n"
                  "               (player vs player, half vs half, week vs week), not as absolute kilometres, until checked against the wearable GPS.")

    # fatigue bookkeeping
    bad = 0
    tot = 0
    for slot in range(md.n_players):
        if np.isnan(md.X[:, slot]).all():
            continue
        rows = pa.fatigue_report(slot)["bins"]
        tot += 1
        bad += int(any(r["live_minutes"] > r["minutes_on_pitch"] + 1e-6 for r in rows))
    rep.add("Fatigue", "live minutes never exceed minutes on the pitch (players failing)", "PASS" if bad == 0 else "FAIL", f"{bad} of {tot}", "0")


# --------------------------------------------------------------------------------------
# data and repeatability
# --------------------------------------------------------------------------------------
def check_data(md, eng, rep):
    ev = md.events
    e = ev[(ev["frame_idx"] >= 0) & (ev["actor_slot"] >= 0)] if "actor_slot" in ev.columns else ev.iloc[0:0]
    d = []
    for r in e.itertuples():
        fi, s = int(r.frame_idx), int(r.actor_slot)
        if np.isfinite(md.ball_x[fi]) and np.isfinite(md.X[fi, s]):
            d.append(np.hypot(md.ball_x[fi] - md.X[fi, s], md.ball_y[fi] - md.Y[fi, s]))
    if len(d) >= 50:
        d = np.array(d)
        rep.add("Events", "player and ball are together at the event frame (median / 90th percentile, m)", "PASS" if np.median(d) < 1.5 and np.percentile(d, 90) < 4 else "WARN",
                f"{np.median(d):.2f} / {np.percentile(d, 90):.2f} m (n={len(d)})", "< 1.5 / < 4")
    # possession after a duel must go to the team that WON it
    col = "possessionEvents.challengeWinnerPlayerId"
    if col in ev.columns:
        pid2side = {str(r.player_id): r.side for r in md.players.itertuples() if r.player_id}
        ch = ev[(ev["poss_type"] == "CH") & (ev["frame_idx"] >= 0)]
        wside = ch[col].astype(str).str.replace(r"\.0$", "", regex=True).map(pid2side)
        known = ch[wside.notna()]
        if len(known) >= 10:
            right = sum(eng.possession_side(int(r.frame_idx)) == wside[i] for i, r in known.iterrows())
            rep.add("Possession", "after a duel the possession goes to the winner's team (share of duels)", "PASS" if right / len(known) > 0.97 else "FAIL",
                    f"{right}/{len(known)}", "> 97 %")
    if str(md.match_id) in KNOWN_GOALS:
        check_facts(md, rep)
    else:
        goals = ev[(ev["poss_type"] == "SH") & ev["is_goal"].astype(bool)]
        print(f"        info   goals found for match {md.match_id}: {goals['side'].value_counts().to_dict()}  - compare with the real score")


def check_repeat(md, rep, eng):
    a = eng.action_values()["vaep"].to_numpy()
    b = MetricsEngine(md).action_values()["vaep"].to_numpy()
    same = len(a) == len(b) and np.allclose(a, b, equal_nan=True)
    rep.add("Repeat", "a second run gives exactly the same action values", "PASS" if same else "FAIL", f"{len(a)} actions", "identical")



# --------------------------------------------------------------------------------------
# 7. Top attacks (attacks.py)
# --------------------------------------------------------------------------------------
def check_attacks(md, eng, rep):
    from attacks import AttackFinder, danger_score
    M = "Attacks"
    at = AttackFinder(eng).attacks()
    if not at:
        rep.skip(M, "attacks found", "no events")
        return
    ev = md.events
    shots = ev[(ev["poss_type"] == "SH") & (ev["frame_idx"] >= 0) & np.isfinite(ev["ball_x"])]
    owner = {}
    for a in at:
        for s_ in a["shots"]:
            owner.setdefault(s_["event_idx"], []).append(a["id"])
    missing = [i for i in shots.index if i not in owner]
    double = [i for i, v in owner.items() if len(v) > 1]
    rep.add(M, "every shot belongs to exactly one attack", "PASS" if not missing and not double else "FAIL",
            f"{len(shots) - len(missing)}/{len(shots)} shots, {len(double)} doubled", "all, once")
    goals = ev[(ev["poss_type"] == "SH") & ev["is_goal"].astype(bool)]
    flagged = sum(1 for a in at if a["goal"])
    rep.add(M, "every real goal is an attack marked as a goal", "PASS" if flagged == len(goals) else "FAIL", f"{flagged}/{len(goals)}", "all")
    sides_ok = all(a["side"] in ("home", "away") for a in at)
    order_ok = all(a["start_frame"] <= a["end_frame"] for a in at)
    overl = sum(1 for p_, q in zip(at, at[1:]) if q["period"] == p_["period"] and q["start_frame"] < p_["end_frame"] - 1)
    rep.add(M, "attacks have one team, a valid time range and do not overlap", "PASS" if sides_ok and order_ok and overl == 0 else "FAIL",
            f"{overl} overlaps", "0")
    d = np.array([a["danger"] for a in at]); p = np.array([a["danger_p"] for a in at])
    mono = spearmanr(p, d).statistic
    rep.add(M, "danger score is 1..100 and follows the danger probability", "PASS" if d.min() >= 1 and d.max() <= 100 and mono > 0.99 else "FAIL",
            f"range {d.min()}-{d.max()}, rho {mono:.3f}", "1..100, rho > 0.99")
    pen = [a for a in at if a["pattern"] == "Penalty"]
    if pen:
        rep.add(M, "a penalty is rated (close to) the maximum", "PASS" if min(a["danger"] for a in pen) >= 95 else "FAIL",
                ", ".join(str(a["danger"]) for a in pen), ">= 95")
    dur = np.array([a["duration_s"] for a in at])
    r = spearmanr(dur, d).statistic
    rep.add(M, "danger is not just 'longer attack = more dangerous' (rank correlation with duration)", "PASS" if abs(r) < 0.5 else "WARN", f"{r:+.2f}", "|rho| < 0.5")
    # NOT circular: use only the best spot reached BEFORE the first shot, and ask if it predicts that a shot follows
    ys = np.array([a["n_shots"] > 0 for a in at], dtype=float)
    pre = np.array([a.get("pre_shot_xt", 0.0) for a in at])
    if 5 <= ys.sum() <= len(ys) - 5:
        auc = rank_auc(ys, pre)
        rep.add(M, "best spot reached BEFORE any shot predicts that the attack ends in a shot (AUC, no circularity)", "PASS" if auc > 0.65 else "WARN",
                f"{auc:.2f} ({int(ys.sum())} of {len(ys)} attacks shoot)", "> 0.65")
    # stability: a different reasonable way to combine shot quality and position (both instead of the larger) -> same top 10?
    alt = sorted(at, key=lambda a: 1 - (1 - a["xg"]) * (1 - a["peak_xt"]), reverse=True)[:10]  # union instead of max
    base = sorted(at, key=lambda a: a["danger_p"], reverse=True)[:10]
    same = len({a["id"] for a in alt} & {a["id"] for a in base})
    rep.add(M, "top 10 is stable when shot quality and position are combined differently", "PASS" if same >= 8 else "WARN", f"{same}/10 the same", ">= 8")
    if same < 8:
        print("        info   with xG and xT on the same ruler, 'larger of the two' and 'union 1-(1-xG)(1-xT)' disagree on a few attacks that")
        print("               have both a fair shot and a good position. We keep the larger: the union counts the same chance twice.")
    print(f"        info   {len(at)} attacks; scale check: 2% -> {danger_score(0.02)}, 10% -> {danger_score(0.10)}, 60% -> {danger_score(0.6)}")

# --------------------------------------------------------------------------------------
def run_match(match_id, args):
    t0 = time.time()
    md = load_match(match_id, data_dir=args.data_dir)
    eng = MetricsEngine(md)
    rng = np.random.default_rng(args.seed)
    rep = Report()
    home, away = md.teams["home"]["name"], md.teams["away"]["name"]
    print(f"\n=== Match {md.match_id}: {home} vs {away}  ({md.n_frames:,} frames) ===")
    print("--- 1. Maths rules ---")
    check_maths(md, eng, rep, rng, 12 if args.quick else 30)
    print("--- 2./3. Do the numbers match what happened? ---")
    check_outcomes(md, eng, rep, rng, args.quick, args.data_dir)
    print("--- 4. Players: pressure, physical, fatigue ---")
    check_players(md, eng, rep, rng)
    print("--- 5. Data ---")
    check_data(md, eng, rep)
    print("--- 7. Top attacks ---")
    check_attacks(md, eng, rep)
    if not args.quick:
        print("--- 6. Repeatability ---")
        check_repeat(md, rep, eng)
    counts = pd.Series([r["status"] for r in rep.rows]).value_counts().to_dict()
    print(f"\nSUMMARY match {md.match_id}: {counts}   ({time.time() - t0:.0f}s)")
    for r in rep.rows:
        if r["status"] in ("WARN", "FAIL"):
            print(f"  [{r['status']}] {r['metric']}: {r['check']}  ->  {r['value']}")
    out = (Path(args.data_dir) if args.data_dir else default_data_dir()) / "interim" / f"accuracy_{md.match_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"match_id": md.match_id, "summary": counts, "checks": rep.rows}, indent=2))
    print(f"saved: {out}")
    return md.match_id, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    results = [run_match(m, args) for m in args.match]
    if len(results) > 1:
        print("\n=== ALL MATCHES ===")
        for mid, c in results:
            print(f"  match {mid}: {c}")


if __name__ == "__main__":
    main()

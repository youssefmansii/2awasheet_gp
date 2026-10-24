"""
validate_metrics.py  -  automatic validation of the six core metrics
=====================================================================

Run (from apps/api, venv active):

    python validate_metrics.py --match 3816

It runs ~40 checks in seven families and prints PASS / WARN / FAIL for each:

  A. Maths       - is the code doing what the paper / reference code says?             (PC)
  B. Physics     - do the numbers behave like football?                                  (PC, EPV)
  C. Robustness  - do results survive reasonable changes of the model parameters?        (PC)
  D. Pass model  - does the completion probability predict real passes? (AUC, calibration)(OBV, Decision)
  E. Predictive  - higher BEFORE shots? and do DAS/OBSO add anything beyond ball position?(DAS, OBSO, EPV)
  F. Outcomes    - goals, disallowed goals, good/bad passes, bias of shot values         (VAEP, OBV)
  G. Facts       - goal counts / minutes vs. what really happened                         (events)

Options:  --fit-calibration   learn a 2-parameter correction of the pass model and save it in assets/
          --quick             skip the slow checks (C, E and decision efficiency)

A PASS means "no evidence of a problem", not "proven correct". WARN = look at it yourself.
Results are also saved to data/interim/validation_<match>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import mannwhitneyu, rankdata, spearmanr, ttest_1samp

from data_loader import default_data_dir, load_match
from metrics_engine import ASSETS_DIR, PASS_FEATURES, MetricsEngine, PitchControlParams, compute_pitch_control, xg_surface

KNOWN_GOALS = {  # real-world facts, used only for these matches: (side, minute)
    "3816": [("home", 10), ("away", 48), ("away", 53)],  # Messi (pen) 10', Al-Shehri 48', Al-Dawsari 53'
}


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, metric: str, name: str, status: str, value: str, rule: str):
        self.rows.append({"metric": metric, "check": name, "status": status, "value": value, "rule": rule})
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print(f"{icon} {metric:<9} {name:<52} {value:<26} {rule}")

    def gt(self, metric, name, val, thr, warn_floor=None, fmt="{:.3f}"):
        """PASS if val > thr; WARN if val > warn_floor; else FAIL."""
        if val is None or not np.isfinite(val):
            return self.add(metric, name, "SKIP", "n/a", f"> {thr}")
        st = "PASS" if val > thr else ("WARN" if warn_floor is not None and val > warn_floor else "FAIL")
        self.add(metric, name, st, fmt.format(val), f"> {thr}")

    def lt(self, metric, name, val, thr, warn_ceiling=None, fmt="{:.3f}"):
        if val is None or not np.isfinite(val):
            return self.add(metric, name, "SKIP", "n/a", f"< {thr}")
        st = "PASS" if val < thr else ("WARN" if warn_ceiling is not None and val < warn_ceiling else "FAIL")
        self.add(metric, name, st, fmt.format(val), f"< {thr}")

    def skip(self, metric, name, why):
        self.add(metric, name, "SKIP", why, "")


def auc_greater(a, b):
    """P(random value of a > random value of b) and the one-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return None, None
    u = mannwhitneyu(a, b, alternative="greater")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def rank_auc(y, score):
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logit(X, y, l2=1.0):
    """Small L2 logistic regression (no sklearn needed). Returns [intercept, w1, w2, ...]."""
    n, k = X.shape

    def nll(w):
        p = np.clip(expit(X @ w[1:] + w[0]), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + 0.5 * l2 * (w[1:] ** 2).sum()

    def grad(w):
        r = expit(X @ w[1:] + w[0]) - y
        return np.concatenate([[r.sum()], X.T @ r + l2 * w[1:]])

    return minimize(nll, np.zeros(k + 1), jac=grad, method="L-BFGS-B").x


def group_folds(groups, k=5, seed=0):
    ug = np.unique(groups)
    np.random.default_rng(seed).shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def cv_scores(X, y, groups, k=5, l2=1.0, seed=0, standardise=True):
    """Out-of-sample linear scores (logits) from grouped k-fold cross-validation."""
    f = group_folds(groups, k, seed)
    out = np.zeros(len(y))
    for i in range(k):
        tr, te = f != i, f == i
        mu = X[tr].mean(0) if standardise else 0.0
        sd = (X[tr].std(0) + 1e-9) if standardise else 1.0
        w = fit_logit((X[tr] - mu) / sd, y[tr], l2)
        out[te] = ((X[te] - mu) / sd) @ w[1:] + w[0]
    return out


# --------------------------------------------------------------------------------------
# reference implementation: a plain loop port of LaurieOnTracking's pitch control at ONE point
# --------------------------------------------------------------------------------------
def laurie_reference(target, pos, vel, is_att, ball, p: PitchControlParams):
    """Plain-loop port of LaurieOnTracking's pitch control at one point.
    Returns (attacking share after rescaling to a total of 1, raw total of his two numbers).
    His Euler step can overshoot 1 when many players could already reach the ball (raw total > 1)."""
    tr, sig = p.reaction_time, p.tti_sigma
    tti = tr + np.linalg.norm(target - (pos + vel * tr), axis=1) / p.max_player_speed
    bt = np.linalg.norm(target - ball) / p.average_ball_speed
    ta, td = tti[is_att].min(), tti[~is_att].min()
    if ta - max(bt, td) >= p.time_to_control_def:
        return 0.0, 1.0
    if td - max(bt, ta) >= p.time_to_control_att:
        return 1.0, 1.0
    att = [i for i in np.flatnonzero(is_att) if tti[i] - ta < p.time_to_control_att]
    dfn = [i for i in np.flatnonzero(~is_att) if tti[i] - td < p.time_to_control_def]
    T = np.arange(bt - p.int_dt, bt + p.max_int_time, p.int_dt)
    pa, pdf = np.zeros(len(T)), np.zeros(len(T))
    ppcf = np.zeros(len(pos))
    i, ptot = 1, 0.0
    while 1 - ptot > p.converge_tol and i < len(T):
        for j in att:
            prob = 1 / (1 + np.exp(-np.pi / np.sqrt(3.0) / sig * (T[i] - tti[j])))
            ppcf[j] += (1 - pa[i - 1] - pdf[i - 1]) * prob * p.lambda_att * p.int_dt
        for j in dfn:
            prob = 1 / (1 + np.exp(-np.pi / np.sqrt(3.0) / sig * (T[i] - tti[j])))
            ppcf[j] += (1 - pa[i - 1] - pdf[i - 1]) * prob * p.lambda_def * p.int_dt
        pa[i], pdf[i] = ppcf[att].sum(), ppcf[dfn].sum()
        ptot = pa[i] + pdf[i]
        i += 1
    tot = pa[i - 1] + pdf[i - 1]
    return float(pa[i - 1] / tot), float(tot)


def candidate_frames(md, eng, min_players=20):
    ok = ((~np.isnan(md.X)).sum(axis=1) >= min_players) & (md.last_event_idx >= 0) & md.observed & np.isfinite(md.ball_x)
    return np.flatnonzero(ok)


# --------------------------------------------------------------------------------------
def check_pitch_control(md, eng, rep, rng, n_frames, quick=False):
    M = "PC"
    cand = candidate_frames(md, eng)
    idxs = rng.choice(cand, size=min(n_frames, len(cand)), replace=False)

    # A. same maths as the reference code
    p = PitchControlParams(use_acceleration=False)
    worst, over, n_cells = 0.0, 0, 0
    for idx in idxs[:5]:
        slots = md.present_slots(int(idx))
        is_home = md.players["side"].to_numpy()[slots] == "home"
        pos = np.column_stack([md.X[idx, slots], md.Y[idx, slots]]).astype(float)
        vel = np.nan_to_num(np.column_stack([md.VX[idx, slots], md.VY[idx, slots]]).astype(float))
        ball = np.array([md.ball_x[idx], md.ball_y[idx]], float)
        pts = np.column_stack([rng.uniform(-50, 50, 24), rng.uniform(-32, 32, 24)])
        mine, _, _ = compute_pitch_control(pos, vel, np.zeros_like(vel), is_home, ball, md.grid, p, points=pts)
        ref = [laurie_reference(t, pos, vel, is_home, ball, p) for t in pts]
        worst = max(worst, float(np.max(np.abs(mine - np.array([r[0] for r in ref])))))
        over += sum(1 for r in ref if r[1] > 1.02)
        n_cells += len(ref)
    rep.lt(M, "vectorised code vs. Laurie's reference, rescaled (max diff, 120 cells)", worst, 0.02, 0.05)
    print(f"        info   Laurie's raw numbers add up to more than 1 in {over} of {n_cells} cells (Euler overshoot); both versions are rescaled to 1")

    # B. physics
    own_ok, own_n, ranges_ok, shares, ballx = 0, 0, True, [], []
    why = {"offside player excluded": 0, "opponent within 3 m": 0, "other": 0}
    sides = md.players["side"].to_numpy()
    for idx in idxs:
        idx = int(idx)
        pcr = eng.pitch_control(idx)
        pc = pcr["pc_home"]
        ranges_ok &= bool(np.isfinite(pc).all() and pc.min() >= -1e-9 and pc.max() <= 1 + 1e-9)
        for s_ in md.present_slots(idx):
            iy, ix = md.grid.cell_index(md.X[idx, s_], md.Y[idx, s_])
            home = sides[int(s_)] == "home"
            good = (pc[iy, ix] > 0.5) == home
            own_ok += int(good)
            own_n += 1
            if not good:
                others = md.present_slots(idx)
                others = others[sides[others] != sides[int(s_)]]
                dmin = np.min(np.hypot(md.X[idx, others] - md.X[idx, s_], md.Y[idx, others] - md.Y[idx, s_])) if len(others) else 99
                if int(s_) not in pcr["slots"]:
                    why["offside player excluded"] += 1
                elif dmin < 3.0:
                    why["opponent within 3 m"] += 1
                else:
                    why["other"] += 1
        shares.append(float(pc.mean()))
        ballx.append(float(md.ball_x[idx]) * md.attack_direction("home", idx))
    rep.add(M, "values are finite and inside [0, 1] (all sampled frames)", "PASS" if ranges_ok else "FAIL", str(ranges_ok), "True")
    rep.gt(M, "each player controls his own square (share of players)", own_ok / max(own_n, 1), 0.80, 0.70)
    print(f"        info   players NOT controlling their own cell ({own_n - own_ok} of {own_n}): {why}")
    rho = spearmanr(ballx, shares).statistic if len(shares) > 5 else None
    rep.gt(M, "home PC share rises as the ball goes towards the away goal (rho)", rho, 0.50, 0.30)

    firsts = []
    for per in sorted(set(md.period.tolist())):
        i0 = int(np.flatnonzero(md.period == per)[0])
        firsts.append(float(eng.pitch_control(i0)["pc_home"].mean()))
    d = max(abs(f - 0.5) for f in firsts)
    rep.lt(M, "kick-off frames are near 50/50 (max distance from 0.5)", d, 0.12, 0.2)

    if quick:
        return idxs
    # C. robustness to reasonable parameter changes (the literature uses different values)
    variants = {
        "reaction time 0.5 s": PitchControlParams(reaction_time=0.5),
        "reaction time 0.9 s": PitchControlParams(reaction_time=0.9),
        "max speed 4 m/s": PitchControlParams(max_player_speed=4.0),
        "max speed 7 m/s": PitchControlParams(max_player_speed=7.0),
        "ball control rate 3": PitchControlParams(lambda_att=3.0),
        "ball control rate 6": PitchControlParams(lambda_att=6.0),
        "no acceleration term": PitchControlParams(use_acceleration=False),
        "no offside rule": PitchControlParams(offsides=False),
    }
    sub = [int(i) for i in idxs[:12]]
    base = np.array([eng.pitch_control(i)["pc_home"].mean() for i in sub])
    shifts = {}
    for name, par in variants.items():
        e2 = MetricsEngine(md, par)
        shifts[name] = float(np.mean(np.abs(np.array([e2.pitch_control(i)["pc_home"].mean() for i in sub]) - base)))
    print("        info   mean change of the PC share per variant: " + ", ".join(f"{k}: {v:.3f}" for k, v in shifts.items()))
    rep.lt(M, "robust to parameter choices (largest mean change in PC share)", max(shifts.values()), 0.05, 0.10)
    return idxs


# --------------------------------------------------------------------------------------
def check_epv_shapes(md, eng, rep):
    M = "EPV"
    idxs = np.arange(0, md.n_frames, int(md.fps * 2))
    xs, vs = [], []
    for i in idxs:
        side = eng.possession_side(int(i))
        if side is None or not np.isfinite(md.ball_x[i]):
            continue
        d = md.attack_direction(side, int(i))
        xs.append(float(md.ball_x[i]) * d)
        vs.append(float(eng.epv.value(md.ball_x[i], md.ball_y[i], d)))
    xs, vs = np.array(xs), np.array(vs)
    if len(xs) < 50:
        return rep.skip(M, "EPV rises towards goal", "too few frames")
    thirds = [vs[xs < -17.5].mean(), vs[(xs >= -17.5) & (xs < 17.5)].mean(), vs[xs >= 17.5].mean()]
    ok = thirds[0] < thirds[1] < thirds[2]
    rep.add(M, "mean EPV: own third < middle < attacking third", "PASS" if ok else "FAIL", " < ".join(f"{t:.4f}" for t in thirds), "increasing")
    rho = spearmanr(xs, vs).statistic
    rep.gt(M, "EPV vs. distance to goal (rho)", rho, 0.60, 0.40)
    g = eng.epv.surface(1)
    iy, ix = np.unravel_index(int(g.argmax()), g.shape)
    px, py = md.grid.x_centers()[ix], md.grid.y_centers()[iy]
    ok = px > 40 and abs(py) < 8
    rep.add(M, "EPV grid peak is in front of the goal (x > 40 m, |y| < 8 m)", "PASS" if ok else "FAIL",
            f"peak {g.max():.2f} at ({px:.0f}, {py:.0f}) m", "in front of goal")


# --------------------------------------------------------------------------------------
def check_predictive(md, eng, rep, rng):
    """DAS, OBSO and EPV should be higher in the seconds BEFORE shots than at a normal moment of possession."""
    ev = md.events
    shots = ev[(ev["poss_type"] == "SH") & (ev["frame_idx"] >= 0) & ev["side"].isin(["home", "away"])]
    pre = {"das": [], "obso": [], "epv": []}
    for r in shots.itertuples():
        for off in (2.0, 4.0):
            i = int(r.frame_idx - round(off * md.fps))
            if i < 0 or md.period[i] != md.period[int(r.frame_idx)] or eng.possession_side(i) != r.side:
                continue
            fm = eng.frame_metrics(i)
            pre["das"].append(fm["das_value"])
            pre["obso"].append(fm["obso_total"])
            pre["epv"].append(fm["epv_ball"])
    cand = candidate_frames(md, eng)
    base_idx = rng.choice(cand, size=min(150, len(cand)), replace=False)
    base = {"das": [], "obso": [], "epv": []}
    for i in base_idx:
        fm = eng.frame_metrics(int(i))
        base["das"].append(fm["das_value"])
        base["obso"].append(fm["obso_total"])
        base["epv"].append(fm["epv_ball"])
    n_pre = len(pre["das"])
    for key, metric, label in (("das", "DAS", "DAS"), ("obso", "OBSO", "OBSO"), ("epv", "EPV", "EPV at ball")):
        a, p = auc_greater(pre[key], base[key])
        if a is None:
            rep.skip(metric, f"{label} higher before shots than normally", f"only {n_pre} pre-shot samples")
            continue
        st = "PASS" if (a > 0.65 and p < 0.05) else ("WARN" if a > 0.55 else "FAIL")
        rep.add(metric, f"{label} higher 2-4 s before shots than normally (AUC)", st, f"AUC {a:.2f}, p={p:.3f}, n={n_pre}", "AUC > 0.65, p < 0.05")
    # The two reference metrics are really the reference code
    import reference_metrics as refm
    fm_list = [eng.frame_metrics(int(i)) for i in base_idx[:30]]
    if refm.das_available():
        diffs = [abs(fm["das"].sum() - fm["das_value"]) / max(fm["das_value"], 1e-12) for fm in fm_list if np.isfinite(fm["das_value"])]
        rep.lt("DAS", f"drawn DAS surface adds up to the package's own DAS ({refm.das_source()}; max relative difference)",
               max(diffs) if diffs else float("nan"), 1e-4, 1e-2, fmt="{:.2e}")
    else:
        rep.add("DAS", "DAS comes from the accessible-space package", "WARN", "package not installed: own estimate in use",
                "pip install accessible-space")
    # OBSO recomputed with the C-OBSO repository's own indexing (obso_player.calc_obso) and compared with ours
    T = np.loadtxt(refm.ASSETS_DIR / "Transition_gauss.csv", delimiter=","); T = T / T.max()
    E = np.loadtxt(refm.ASSETS_DIR / "EPV_grid.csv", delimiter=","); E = E / E.max()
    worst = 0.0
    for fm in fm_list:
        i = fm["idx"]
        b = eng.ball_xy(i)
        if b is None:
            continue
        gx = min(max(int((b[0] + 52.5) // (105 / 50)), 0), 49)  # these four lines are C-OBSO's calc_obso
        gy = min(max(int((b[1] + 34) // (68 / 32)), 0), 31)
        trans = T[31 - gy:63 - gy, 49 - gx:99 - gx]
        score = E if fm["attack_dir"] == 1 else E[::-1, ::-1]
        pcr = eng.pitch_control(i)
        worst = max(worst, float(np.abs(pcr["pc_att"] * trans * score - fm["obso_grid"]).max()))
    rep.lt("OBSO", "grid equals the C-OBSO formula recomputed with its own indexing (max abs difference)", worst, 1e-9, 1e-6, fmt="{:.2e}")
    # a player's OBSO is the grid value at the cell he stands in
    bad = 0
    for fm in fm_list:
        for o in fm["obso_players"]:
            jy, jx = eng.grid.cell_index(o["target_x"], o["target_y"])
            bad += int(abs(fm["obso_grid"][int(jy), int(jx)] - o["obso"]) > 1e-12)
    rep.lt("OBSO", "each player's OBSO equals the grid value at his own cell (mismatches)", bad, 0.5, 0.5, fmt="{:.0f}")


# --------------------------------------------------------------------------------------
def pass_samples(md, eng):
    """One row per real pass: [completed, p_lane, p_simple, 30-s block, 6 features...]."""
    ev = md.events
    ps = ev[(ev["poss_type"] == "PA") & (ev["frame_idx"] >= 0) & ev["side"].isin(["home", "away"])]
    rows = []
    for r in ps.itertuples():
        fi = int(r.frame_idx)
        ts = int(r.target_slot) if r.target_slot >= 0 else int(r.receiver_slot)
        if ts < 0 or not np.isfinite(md.X[fi, ts]) or not np.isfinite(r.ball_x):
            continue
        origin = np.array([r.ball_x, r.ball_y])
        tgt = np.array([[md.X[fi, ts], md.Y[fi, ts]]])
        feats = eng.pass_features(fi, r.side, origin, tgt)[0]
        rows.append([int(r.pass_outcome == "C"),
                     float(eng.pass_probabilities(fi, r.side, origin, tgt, model="lane", calibrated=False)[0]),
                     float(eng.pass_probabilities(fi, r.side, origin, tgt, model="simple", calibrated=False)[0]),
                     fi // int(30 * md.fps), *feats])
    return np.array(rows, dtype=float)


def _quintiles(p, y):
    q = np.quantile(p, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    gaps, table = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p <= hi)
        gaps.append(abs(p[m].mean() - y[m].mean()))
        table.append(f"{p[m].mean():.2f}->{y[m].mean():.2f}")
    return max(gaps), "  ".join(table)


def check_pass_model(md, eng, rep, fit_calibration):
    M = "OBV/Dec"
    S = pass_samples(md, eng)
    if len(S) < 100:
        return rep.skip(M, "pass completion model", f"only {len(S)} passes")
    y, p_lane, p_simple, grp, F = S[:, 0], S[:, 1], S[:, 2], S[:, 3].astype(int), S[:, 4:]
    auc_l, auc_s = rank_auc(y, p_lane), rank_auc(y, p_simple)
    print(f"        info   {len(S)} real passes, completion rate {y.mean():.1%}. AUC: simple {auc_s:.3f} | lane {auc_l:.3f}  (mean predicted by lane model {p_lane.mean():.1%})")
    # honest out-of-sample test of the fitted model: 5-fold, folds are whole 30-second blocks of play
    aucs, cvp = [], np.zeros(len(y))
    for sd in range(3):
        z = cv_scores(F, y, grp, k=5, l2=3.0, seed=sd)
        aucs.append(rank_auc(y, z))
        cvp = cvp + expit(z) / 3
    auc_f = float(np.mean(aucs))
    rep.gt(M, f"fitted pass model predicts real completions, cross-validated (AUC, {len(S)} passes)", auc_f, 0.75, 0.68)
    rep.gt(M, "fitted model beats the hand-made lane model (AUC difference)", auc_f - auc_l, 0.01, -0.005)
    gap, table = _quintiles(cvp, y)
    brier0 = float(np.mean((y.mean() - y) ** 2))
    brier = float(np.mean((cvp - y) ** 2))
    print(f"        info   Brier score {brier:.3f} (always guessing the average: {brier0:.3f});  predicted->observed by quintile: {table}")
    rep.lt(M, "fitted pass probabilities match observed rates (worst quintile gap)", gap, 0.06, 0.12)
    w = fit_logit((F - F.mean(0)) / (F.std(0) + 1e-9), y, 3.0)
    print("        info   what drives completion (standardised weights): " + ", ".join(f"{n}: {c:+.2f}" for n, c in zip(PASS_FEATURES, w[1:])))
    if fit_calibration:
        mu, sd_ = F.mean(0), F.std(0) + 1e-9
        out = ASSETS_DIR / "pass_model_fitted.json"
        out.write_text(json.dumps({"features": PASS_FEATURES, "mu": mu.tolist(), "sd": sd_.tolist(), "w": w[1:].tolist(), "a": float(w[0]),
                                   "n_passes": int(len(S)), "cv_auc": auc_f}))
        z = logit(np.clip(p_lane, 0.01, 0.99))[:, None]
        wl = fit_logit(z, y, l2=1e-6)
        (ASSETS_DIR / "pass_calibration_lane.json").write_text(json.dumps({"model": "lane", "a": float(wl[0]), "b": float(wl[1]), "n_passes": int(len(S))}))
        print(f"        info   saved assets/pass_model_fitted.json (cross-validated AUC {auc_f:.3f}) and pass_calibration_lane.json; re-run the validation to use them")


def check_incremental(md, eng, rep, step_s):
    """Do DAS / OBSO / pitch control tell us anything about a coming shot that the BALL POSITION (EPV) does not?"""
    ev = md.events
    shots = ev[(ev["poss_type"] == "SH") & (ev["frame_idx"] >= 0) & ev["side"].isin(["home", "away"])]
    shot_frames = {s: shots[shots["side"] == s]["frame_idx"].to_numpy() for s in ("home", "away")}
    step, horizon = int(step_s * md.fps), int(6 * md.fps)
    rows, t0 = [], time.time()
    for i in range(0, md.n_frames, step):
        if not md.observed[i] or (~np.isnan(md.X[i])).sum() < 20 or not np.isfinite(md.ball_x[i]):
            continue
        side = eng.possession_side(i)
        if side is None:
            continue
        fm = eng.frame_metrics(i)
        sf = shot_frames[side]
        share = fm["pc_home"].mean() if side == "home" else 1 - fm["pc_home"].mean()
        rows.append((int(((sf > i) & (sf <= i + horizon)).any()), fm["epv_ball"], share, np.log(fm["das_value"] + 1e-3),
                     np.log(fm["obso_total"] + 1e-4), i // int(30 * md.fps)))
    T = np.array(rows, dtype=float)
    T = T[np.isfinite(T).all(axis=1)]
    y, grp = T[:, 0], T[:, 5].astype(int)
    print(f"        info   {len(T)} sampled moments of possession, {int(y.sum())} followed by a shot within 6 s ({time.time() - t0:.0f}s)")
    if y.sum() < 15:
        return rep.skip("DAS/OBSO", "incremental value beyond ball position", "too few shots")
    sets = {"EPV only": [1], "EPV + PC": [1, 2], "EPV + DAS": [1, 3], "EPV + OBSO": [1, 4], "EPV + PC + DAS + OBSO": [1, 2, 3, 4]}
    scores = {name: np.mean([cv_scores(T[:, cols], y, grp, seed=sd) for sd in range(3)], axis=0) for name, cols in sets.items()}
    aucs = {k: rank_auc(y, v) for k, v in scores.items()}
    print("        info   cross-validated AUC of 'shot in the next 6 s':  " + "  |  ".join(f"{k}: {v:.3f}" for k, v in aucs.items()))
    # bootstrap over 30-second blocks: is the AUC gain clearly above zero, clearly below, or just not measurable?
    ug = np.unique(grp)
    idx_of = {g: np.flatnonzero(grp == g) for g in ug}
    rng = np.random.default_rng(1)
    gains = {k: [] for k in ("EPV + PC", "EPV + DAS", "EPV + OBSO")}
    for _ in range(300):
        pick = np.concatenate([idx_of[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        if y[pick].sum() < 3:
            continue
        base = rank_auc(y[pick], scores["EPV only"][pick])
        for k in gains:
            gains[k].append(rank_auc(y[pick], scores[k][pick]) - base)
    for metric, key in (("PC", "EPV + PC"), ("DAS", "EPV + DAS"), ("OBSO", "EPV + OBSO")):
        lo, hi = np.percentile(gains[key], [2.5, 97.5])
        pt = aucs[key] - aucs["EPV only"]
        # "adds no information" is a finding about usefulness for predicting shots, not a calculation error; with ~24 shots, adding
        # features to a model also tends to overfit, which shows up as a slightly NEGATIVE gain. So the worst status is WARN.
        status = "PASS" if lo > 0 else "WARN"
        rep.add(metric, "adds information beyond ball position (AUC gain over EPV only, 95% interval)", status,
                f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]", "interval above 0")
    print(f"        info   WARN here means 'cannot be shown with {int(y.sum())} shots', not 'wrong': the interval is too wide. More matches would narrow it.")


# --------------------------------------------------------------------------------------
def check_actions(md, eng, rep):
    av = eng.action_values()
    ev = md.events
    if not len(av):
        return rep.skip("VAEP", "actions", "no on-ball actions")
    # --- goals
    g = ev[(ev["poss_type"] == "SH") & (ev["shot_outcome"] == "G")]
    real = g[g["is_goal"].astype(bool)]
    dis = g[~g["is_goal"].astype(bool)]
    vmap = av.set_index("event_idx")
    real_v = [float(vmap.loc[i, "vaep"]) for i in real.index if i in vmap.index]
    dis_v = [float(vmap.loc[i, "vaep"]) for i in dis.index if i in vmap.index]
    if real_v:
        rep.gt("VAEP", f"real goals get a large value (min of {len(real_v)} goals)", min(real_v), 0.5, 0.2)
    else:
        rep.skip("VAEP", "real goals get a large value", "no goals found")
    if dis_v:
        rep.lt("VAEP", f"disallowed 'G' shots are NOT rewarded (max of {len(dis_v)})", max(dis_v), 0.05, 0.2)
    # --- passes
    ps = av[av["action"] == "pass"].copy()
    ok_v, bad_v = ps[ps["success"]]["vaep"], ps[~ps["success"]]["vaep"]
    rep.gt("VAEP", "completed passes are worth more than lost ones (mean diff)", float(ok_v.mean() - bad_v.mean()), 0.02, 0.0)
    adir = np.array([md.attack_direction(s, int(i)) for s, i in zip(ps["side"], ps["frame_idx"])])
    dx = ((ps["end_x"] - ps["x"]) * adir).to_numpy()
    fwd, back = ps[(dx > 5) & ps["success"]]["vaep"], ps[(dx < -5) & ps["success"]]["vaep"]
    rep.gt("VAEP", "completed forward passes beat backward passes (mean diff)", float(fwd.mean() - back.mean()), 0.01, 0.0)
    # --- OBV vs VAEP
    # OBV (expected value of the intended pass) = progress if the pass is completed  -  the expected cost of losing the ball.
    # VAEP (what really happened) of a COMPLETED pass has no "cost of losing" part, so the two cannot agree on completed
    # passes, by design. What must agree is OBV's progress part: it is built from the same pitch states as VAEP, using the
    # receiver's position at the kick instead of where the ball ended (about 2 m apart for most passes).
    # Over all passes the agreement is modest because VAEP is dominated by pass luck, which OBV removes on purpose.
    done = ps[ps["success"]]
    rho_c = spearmanr(done["obv_progress"], done["vaep"]).statistic
    rep.gt("OBV", f"OBV's progress part agrees with VAEP on completed passes (rho, n={len(done)})", rho_c, 0.80, 0.60)
    rho_full = spearmanr(done["obv"], done["vaep"]).statistic
    print(f"        info   full OBV vs VAEP on completed passes: rho {rho_full:.2f} (lower by design: OBV also subtracts the risk of losing the ball)")
    diff = ps["vaep"] - ps["obv"]
    se_d = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    rep.add("OBV", "expected value is unbiased: mean realised minus mean expected (all passes)", "PASS" if abs(diff.mean()) <= 2 * se_d else "WARN",
            f"{diff.mean():+.4f} (+-{1.96 * se_d:.4f})", "within 2 standard errors of 0")
    rho_all = spearmanr(ps["obv"], ps["vaep"]).statistic
    print(f"        info   OBV vs VAEP over all {len(ps)} passes: rho {rho_all:.2f} (expected to be modest: VAEP mostly reflects pass luck)")
    d_vaep = float(ok_v.mean() - bad_v.mean())
    d_obv = float(ps[ps["success"]]["obv"].mean() - ps[~ps["success"]]["obv"].mean())
    rep.lt("OBV", "OBV depends less on pass luck than VAEP (outcome gap ratio)", abs(d_obv) / max(abs(d_vaep), 1e-9), 0.7, 1.0)
    # --- team totals, bias of shot values, EPV / xG calibration
    tot = av.groupby("side")["vaep"].sum().round(3).to_dict()
    print(f"        info   team VAEP totals: {tot}")
    mean_all = float(av["vaep"].mean())
    se_all = float(av["vaep"].std(ddof=1) / np.sqrt(len(av)))
    pas = av[av["action"] == "pass"]
    comp, lost = pas[pas["success"]], pas[~pas["success"]]
    print(f"        info   average VAEP per action {mean_all:+.4f} (+-{1.96 * se_all:.4f}); completed passes {comp['vaep'].mean():+.4f} (n={len(comp)}), "
          f"lost passes {lost['vaep'].mean():+.4f} (n={len(lost)}); break-even completed gain would be {-lost['vaep'].mean() * len(lost) / max(len(comp), 1):+.4f}")
    print("        info   raw VAEP sits slightly below 0 because the EPV grid was not fitted to this match; 'vaep_oa' (above average) removes that offset")
    rep.add("VAEP", "raw value function is balanced: average value per action is close to 0 (martingale check)",
            "PASS" if abs(mean_all) <= 2 * se_all else ("WARN" if abs(mean_all) <= 0.01 else "FAIL"),
            f"{mean_all:+.4f} +-{1.96 * se_all:.4f}", "within 2 standard errors of 0")
    oa_mean = float(av["vaep_oa"].mean())
    keep = float(spearmanr(av["vaep"], av["vaep_oa"]).statistic)
    rep.add("VAEP", "above-average VAEP is centred (mean 0) and keeps the ranking of raw VAEP (rho)", "PASS" if (abs(oa_mean) < 1e-3 and keep > 0.8) else "WARN",
            f"mean {oa_mean:+.5f}, rho {keep:.3f}", "|mean| < 0.001, rho > 0.8")
    tot_oa = av.groupby("side")["vaep_oa"].sum().round(3).to_dict()
    print(f"        info   team totals, above-average version: {tot_oa} (sum of both teams is 0 by construction; the difference between teams is what to read)")
    by = av.groupby("action")["vaep"].agg(["count", "mean", "sum"]).round(3)
    print("        info   VAEP by action type:  " + "  ".join(f"{a}: n={int(r['count'])} mean={r['mean']:+.3f} sum={r['sum']:+.2f}" for a, r in by.iterrows()))
    sh = av[av["action"] == "shot"]
    if len(sh) >= 10:
        z = float(sh["vaep"].mean() / (sh["vaep"].std(ddof=1) / np.sqrt(len(sh))))
        rep.lt("VAEP", f"shot values are not systematically biased (|z| of mean over {len(sh)} shots)", abs(z), 2.0, 3.0, fmt="{:.2f}")
        grid = md.grid
        xg_sum = sum(float(xg_surface(grid, md.attack_direction(r.side, int(r.frame_idx)))[grid.cell_index(r.x, r.y)]) for r in sh.itertuples())
        print(f"        info   expected goals from the {len(sh)} shots: EPV grid {sh['epv_before'].sum():.2f}, xG surface {xg_sum:.2f}; "
              f"real goals {int(sh['success'].sum())}; ball-in-net incl. disallowed {int(((ev['poss_type'] == 'SH') & (ev['shot_outcome'] == 'G')).sum())}")
    print("        info   top 5 actions by VAEP (do they look like real key moments?):")
    top = av.sort_values("vaep", ascending=False).head(5)
    for r in top.itertuples():
        nm = md.players.iloc[int(r.slot)]["name"] if r.slot >= 0 else "?"
        print(f"               min {r.match_minute:5.1f}  frame {r.frame_num:<7} {r.side:<5} {nm:<22} {r.action:<5} success={r.success}  vaep={r.vaep:+.3f}  obv={r.obv:+.3f}")
    return av


def check_decisions(md, eng, rep, data_dir):
    from player_analytics import PlayerAnalytics

    pa = PlayerAnalytics(eng, data_dir=data_dir)
    dec = pa._compute_decisions()
    if not len(dec):
        return rep.skip("Decision", "decision score", "no passes")
    M = "Decision"
    sc = dec.dropna(subset=["decision_score"])
    rep.lt(M, f"score is not saturated at 0 (share of {len(sc)} passes scored 0)", float((sc["decision_score"] <= 1e-9).mean()), 0.25, 0.40)
    t = ttest_1samp(sc["decision_score"], 0.5, alternative="greater")
    mean = float(sc["decision_score"].mean())
    rep.add(M, "players choose better than a random teammate (mean score vs 0.5)", "PASS" if (mean > 0.55 and t.pvalue < 0.05) else ("WARN" if mean > 0.5 else "FAIL"),
            f"{mean:.2f}, p={t.pvalue:.3f}", "> 0.55, p < 0.05")
    # Does the value the score is built on (expected dEPV of the chosen pass) predict what really happened?
    r = spearmanr(dec["chosen_exp_depv"], dec["realised_depv"])
    rep.add(M, "expected dEPV of the chosen pass predicts the realised dEPV (rho, p)", "PASS" if (r.statistic > 0.1 and r.pvalue < 0.05) else ("WARN" if r.statistic > 0 else "FAIL"),
            f"rho {r.statistic:.2f}, p={r.pvalue:.3f}", "rho > 0.10, p < 0.05")
    bias = float(dec["realised_depv"].mean() - dec["chosen_exp_depv"].mean())
    se = float(dec["realised_depv"].std(ddof=1) / np.sqrt(len(dec)))
    rep.add(M, "expected dEPV is unbiased: mean realised minus mean expected", "PASS" if abs(bias) <= 2 * se else "WARN",
            f"{bias:+.4f} (+-{1.96 * se:.4f})", "within 2 standard errors of 0")
    # is the player-level score a stable property or noise? split passes in odd / even and compare players
    d2 = sc.assign(half=np.arange(len(sc)) % 2)
    g = d2.groupby(["slot", "half"])["decision_score"].agg(["mean", "count"]).unstack("half").dropna()
    g = g[(g[("count", 0)] >= 5) & (g[("count", 1)] >= 5)]
    if len(g) >= 8:
        rr = spearmanr(g[("mean", 0)], g[("mean", 1)])
        print(f"        info   split-half reliability of a player's score (odd vs even passes, {len(g)} players): rho {rr.statistic:.2f}")
    print(f"        info   mean score {mean:.2f}; min-max version {sc['score_minmax'].mean():.2f}; mean regret {dec['regret'].mean():.4f} EPV; "
          f"best option chosen {(dec['chosen_slot'] == dec['best_slot']).mean():.0%}; legacy ratio scored 0 in {(dec['ratio_score'] <= 1e-9).mean():.0%} of passes")


def check_facts(md, rep):
    ev = md.events
    real = ev[(ev["poss_type"] == "SH") & ev["is_goal"].astype(bool)]
    print("        info   frames to open in the UI (3 s before each real goal) and judge with your own eyes:")
    for r in real.itertuples():
        if r.frame_idx >= 0:
            i = max(0, int(r.frame_idx - round(3 * md.fps)))
            print(f"               {r.side:<5} {str(r.player_name):<22} minute {md.match_minute[int(r.frame_idx)]:5.1f}   frameNum {int(md.frame_num[i])}")
    known = KNOWN_GOALS.get(str(md.match_id))
    if not known:
        print(f"        info   goals found: {real['side'].value_counts().to_dict()} (no reference list for match {md.match_id})")
        return
    found = [(r.side, float(md.match_minute[int(r.frame_idx)])) for r in real.itertuples() if r.frame_idx >= 0]
    hits = 0
    for side, minute in known:
        hit = any(s == side and abs(m - minute) <= 3 for s, m in found)
        hits += int(hit)
    rep.add("Facts", "real goals match the real match (side and minute +-3)", "PASS" if hits == len(known) and len(found) == len(known) else "FAIL",
            f"{hits}/{len(known)} matched, {len(found)} found", "all, no extras")
    print(f"        info   found goals: {[(s, round(m, 1)) for s, m in found]}  expected: {known}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--frames", type=int, default=60, help="random frames for the pitch-control checks")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-decisions", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip the slow checks")
    ap.add_argument("--fit-calibration", action="store_true", help="save the pass-model calibration to assets/")
    ap.add_argument("--step", type=float, default=3.0, help="seconds between sampled moments for the incremental-value test")
    args = ap.parse_args()

    t0 = time.time()
    md = load_match(args.match, data_dir=args.data_dir)
    eng = MetricsEngine(md)
    rng = np.random.default_rng(args.seed)
    rep = Report()
    print(f"\nValidating match {md.match_id}: {md.teams['home']['name']} vs {md.teams['away']['name']}  ({md.n_frames:,} frames)\n")
    print("--- Pitch Control (maths, physics, robustness) ---")
    check_pitch_control(md, eng, rep, rng, args.frames, quick=args.quick)
    print("--- EPV ---")
    check_epv_shapes(md, eng, rep)
    print("--- Pass completion model (feeds OBV and Decision efficiency) ---")
    check_pass_model(md, eng, rep, args.fit_calibration)
    print("--- Predictive checks: DAS / OBSO / EPV before shots ---")
    check_predictive(md, eng, rep, rng)
    if not args.quick:
        print("--- Do DAS / OBSO / PC add information beyond the ball position? (slow) ---")
        check_incremental(md, eng, rep, args.step)
    print("--- VAEP / OBV ---")
    check_actions(md, eng, rep)
    if not (args.skip_decisions or args.quick):
        print("--- Decision efficiency ---")
        check_decisions(md, eng, rep, args.data_dir)
    print("--- Facts ---")
    check_facts(md, rep)

    counts = pd.Series([r["status"] for r in rep.rows]).value_counts().to_dict()
    print(f"\nSUMMARY: {counts}   ({time.time() - t0:.0f}s)")
    todo = [r for r in rep.rows if r["status"] in ("WARN", "FAIL")]
    if todo:
        print("TO LOOK AT:")
        for r in todo:
            print(f"  [{r['status']}] {r['metric']}: {r['check']}  ->  {r['value']}")
    out = (Path(args.data_dir) if args.data_dir else default_data_dir()) / "interim" / f"validation_{md.match_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"match_id": md.match_id, "summary": counts, "checks": rep.rows}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

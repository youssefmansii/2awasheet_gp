"""
metrics_engine.py  -  Step 2 of the analytics engine
====================================================

The six metrics, computed from a ``MatchData`` object (see data_loader.py):

 1. Pitch Control (PC)   - Spearman (2018); follows Laurie Shaw's LaurieOnTracking
                           ``Metrica_PitchControl.py`` but vectorised with NumPy.
 2. DAS                  - Dangerous Attacking Space: attacking pitch control x goal threat,
                           weighted by how far the space is behind the defensive line.
 3. OBSO                 - Off-Ball Scoring Opportunity = PC x P(pass success) x xG,
                           computed for every off-ball attacker. The pass-success term is
                           normalised over the pitch (Spearman's transition-model convention),
                           so OBSO values are scoring probabilities (typically 0.001 - 0.05).
 4. EPV                  - Expected Possession Value from Laurie's 50x32 EPV grid.
 5. VAEP                 - EPV-style approximation of "change in P(scores) - P(concedes)".
 6. OBV                  - On-Ball Value: same state values, but credited by action type
                           (pass / carry / dribble / tackle) using EXPECTED outcomes.

Everything on the pitch grid is shaped (ny, nx) = (32, 50) with [row = y, column = x].
All surfaces are expressed in the real pitch frame (not flipped) so the front-end can draw
them directly. Internally "attack direction" (+1 / -1) tells which goal a team attacks.

IMPORTANT HONESTY NOTES (also shown in the README of the project):
 * The xG surface, pass-success model and the VAEP/OBV approximations are transparent
   heuristics, not models trained on many matches. They are meant to rank and compare, and
   can be replaced later (functions are isolated for that reason).
 * EPV uses Laurie Shaw's public grid (built from Metrica data).

Self-test:   python metrics_engine.py --match 3816
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import reference_metrics as ref
from data_loader import BIN_LABELS, MatchData, PitchGrid, load_match

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EPV_GRID_URL = "https://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/master/EPV_grid.csv"

GOAL_HALF_WIDTH = 3.66  # goal mouth is 7.32 m wide


# ===========================================================================
# 1. PITCH CONTROL  (Spearman 2018 / Laurie Shaw implementation)
# ===========================================================================
@dataclass
class PitchControlParams:
    reaction_time: float = 0.7  # T_react (s)
    max_player_speed: float = 5.0  # v_max (m/s) used for time-to-intercept
    average_ball_speed: float = 15.0  # m/s
    tti_sigma: float = 0.45  # uncertainty of interception time (s)
    lambda_att: float = 4.3  # ball control rate of attackers (1/s)
    kappa_def: float = 1.0  # defenders' lambda = kappa_def * lambda_att
    int_dt: float = 0.04  # integration step (s)
    max_int_time: float = 10.0  # integrate at most this many seconds
    converge_tol: float = 0.01
    use_acceleration: bool = True  # extension: also project the player with his acceleration
    max_accel_used: float = 6.0  # m/s^2 cap for that projection
    offsides: bool = True  # attackers in an offside position are ignored (as in Laurie's code)
    gk_lambda_factor: float = 3.0  # a defending goalkeeper controls the ball 3x faster (as in Laurie's code)

    @property
    def lambda_def(self) -> float:
        return self.kappa_def * self.lambda_att

    @property
    def time_to_control_att(self) -> float:
        return 3.0 * np.log(10.0) * (np.sqrt(3.0) * self.tti_sigma / np.pi + 1.0 / self.lambda_att)

    @property
    def time_to_control_def(self) -> float:
        return 3.0 * np.log(10.0) * (np.sqrt(3.0) * self.tti_sigma / np.pi + 1.0 / self.lambda_def)

    def key(self) -> tuple:
        return tuple(sorted(self.__dict__.items()))


@dataclass
class PitchControlResult:
    pc_att: np.ndarray  # (ny, nx) probability the ATTACKING (in-possession) team controls each cell
    pc_def: np.ndarray
    slots: np.ndarray  # (N,) player slots used
    is_att: np.ndarray  # (N,) bool
    ppcf_players: Optional[np.ndarray] = None  # (N, ny, nx) each player's own share of control


def compute_pitch_control(
    pos: np.ndarray,  # (N, 2) metres
    vel: np.ndarray,  # (N, 2)
    acc: np.ndarray,  # (N, 2)
    is_att: np.ndarray,  # (N,) bool
    ball_xy: Optional[np.ndarray],
    grid: PitchGrid,
    params: PitchControlParams,
    per_player: bool = False,
    points: Optional[np.ndarray] = None,
    is_gk: Optional[np.ndarray] = None,  # (N,) bool, goalkeepers (faster ball control when defending)
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Vectorised Spearman pitch control over all grid cells at once.

    For every cell g and player n:
        r_reaction = p + v*T_react (+ 0.5*a*T_react^2)
        tti[g,n]   = T_react + |g - r_reaction| / v_max
        P_n(T)     = 1 / (1 + exp(-pi/sqrt(3)/sigma * (T - tti)))      (can he get there by time T?)
        dPPCF_n/dT = (1 - sum PPCF) * P_n(T) * lambda                   (integrated over T)
    """
    ny, nx = grid.ny, grid.nx
    G = grid.points() if points is None else np.asarray(points, dtype=float).reshape(-1, 2)  # (M, 2)
    M = G.shape[0]
    flat_out = points is not None  # custom points -> return flat (M,) arrays instead of (ny, nx) grids
    N = pos.shape[0]
    if N == 0 or is_att.sum() == 0 or (~is_att).sum() == 0:
        pc = np.full(M, 0.5)
        ones_att = 1.0 if is_att.sum() > 0 else 0.0
        pc[:] = ones_att
        if flat_out:
            return pc, 1 - pc, None
        return pc.reshape(ny, nx), (1 - pc).reshape(ny, nx), None

    vel = np.nan_to_num(vel)
    acc = np.nan_to_num(acc)
    if params.use_acceleration:
        a_mag = np.linalg.norm(acc, axis=1, keepdims=True)
        scale = np.minimum(1.0, params.max_accel_used / np.maximum(a_mag, 1e-9))
        acc = acc * scale
        r_react = pos + vel * params.reaction_time + 0.5 * acc * params.reaction_time ** 2
    else:
        r_react = pos + vel * params.reaction_time

    # time to intercept, shape (M, N)
    dist = np.sqrt(((G[:, None, :] - r_react[None, :, :]) ** 2).sum(axis=2))
    tti = params.reaction_time + dist / params.max_player_speed

    if ball_xy is None or not np.all(np.isfinite(ball_xy)):
        ball_time = np.zeros(M)
    else:
        ball_time = np.sqrt(((G - ball_xy[None, :]) ** 2).sum(axis=1)) / params.average_ball_speed

    inf = np.inf
    tau_att = np.where(is_att[None, :], tti, inf).min(axis=1)
    tau_def = np.where(~is_att[None, :], tti, inf).min(axis=1)

    # shortcuts of the original algorithm: one team clearly gets there first
    def_wins = tau_att - np.maximum(ball_time, tau_def) >= params.time_to_control_def
    att_wins = (tau_def - np.maximum(ball_time, tau_att) >= params.time_to_control_att) & ~def_wins

    pc_att = np.zeros(M)
    pc_def = np.zeros(M)
    pc_att[att_wins] = 1.0
    pc_def[def_wins] = 1.0
    ppcf_players = np.zeros((M, N)) if per_player else None

    todo = np.flatnonzero(~(att_wins | def_wins))
    if todo.size:
        tti_t = tti[todo]  # (m, N)
        # only players who can plausibly matter in each cell
        act_att = is_att[None, :] & ((tti_t - tau_att[todo, None]) < params.time_to_control_att)
        act_def = (~is_att)[None, :] & ((tti_t - tau_def[todo, None]) < params.time_to_control_def)
        gk = np.zeros(N, dtype=bool) if is_gk is None else np.asarray(is_gk, dtype=bool)
        lam_def_each = params.lambda_def * np.where(gk, params.gk_lambda_factor, 1.0)
        lam = np.where(is_att, params.lambda_att, lam_def_each)[None, :]
        active = act_att | act_def

        T = ball_time[todo] - params.int_dt  # start time per cell
        pa = np.zeros(todo.size)
        pd_ = np.zeros(todo.size)
        pl = np.zeros((todo.size, N)) if per_player else None
        alive = np.ones(todo.size, dtype=bool)
        k_const = np.pi / np.sqrt(3.0) / params.tti_sigma
        n_steps = int((params.max_int_time + params.int_dt) / params.int_dt)

        for _ in range(1, n_steps):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            T[idx] += params.int_dt
            # P(player n reaches the ball by time T)
            z = -k_const * (T[idx, None] - tti_t[idx])
            prob = 1.0 / (1.0 + np.exp(np.clip(z, -50, 50)))
            prob = prob * active[idx]
            free = (1.0 - pa[idx] - pd_[idx])[:, None]
            d = free * prob * lam * params.int_dt  # (n_idx, N)
            if per_player:
                pl[idx] += d
            pa[idx] += (d * is_att[None, :]).sum(axis=1)
            pd_[idx] += (d * (~is_att)[None, :]).sum(axis=1)
            tot = pa[idx] + pd_[idx]
            alive[idx[tot > 1.0 - params.converge_tol]] = False

        # rescale so pc_att + pc_def == 1 exactly. (Laurie's Euler step can overshoot 1 by up to ~6% where many
        # players could already reach the ball; the rescaled result equals his rescaled output.)
        tot = np.maximum(pa + pd_, 1e-9)
        pc_att[todo] = pa / tot
        pc_def[todo] = pd_ / tot
        if per_player:
            ppcf_players[todo] = pl / tot[:, None]

    if flat_out:
        return pc_att, pc_def, (ppcf_players.T if per_player else None)
    out_pl = None
    if per_player:
        out_pl = ppcf_players.T.reshape(N, ny, nx)
        # cells decided by the shortcut belong (fully) to the nearest player of the winning team
        sc = np.flatnonzero(att_wins | def_wins)
        if sc.size:
            near_att = np.where(is_att[None, :], tti[sc], inf).argmin(axis=1)
            near_def = np.where(~is_att[None, :], tti[sc], inf).argmin(axis=1)
            winner = np.where(att_wins[sc], near_att, near_def)
            flat = out_pl.reshape(N, -1)
            flat[winner, sc] = 1.0
            out_pl = flat.reshape(N, ny, nx)
    return pc_att.reshape(ny, nx), pc_def.reshape(ny, nx), out_pl


# ===========================================================================
# Static surfaces: xG, pass success, EPV grid
# ===========================================================================
PENALTY_XG = 0.79  # penalty conversion rate in top-level football (published event-data models put it at about 0.78-0.79)


def xg_surface(grid: PitchGrid, attack_dir: int = 1) -> np.ndarray:
    """Open-play shot value of each cell, for a team attacking towards +x (attack_dir=1) or -x.

    Logistic model on distance d and the angle the goal mouth covers:
        logit(xG) = -2.884 + 2.39 * angle - 0.0577 * d
    Hand-calibrated to typical open-play values (6-yard centre ~0.40, penalty spot ~0.12,
    20 m ~0.04, 30 m ~0.017). Swap this function for a fitted model later if you wish.
    """
    xx, yy = grid.mesh()
    gx = grid.length / 2
    dx = gx - xx
    dist = np.sqrt(dx ** 2 + yy ** 2)
    a1 = np.arctan2(GOAL_HALF_WIDTH - yy, np.maximum(dx, 0.01))
    a2 = np.arctan2(-GOAL_HALF_WIDTH - yy, np.maximum(dx, 0.01))
    angle = np.abs(a1 - a2)
    xg = 1.0 / (1.0 + np.exp(-(-2.884 + 2.39 * angle - 0.0577 * np.maximum(dist, 1.0))))
    xg = np.where(dx < 0.5, 0.0, xg)
    return xg if attack_dir == 1 else xg[::-1, ::-1]


def xg_at(x: float, y: float, attack_dir: int = 1) -> float:
    """Open-play xG of a shot taken exactly at (x, y) metres, same model as xg_surface but without rounding the spot to a
    2 m grid cell (the cell can be off by up to ~0.03 xG close to goal)."""
    x, y = float(x) * attack_dir, float(y) * attack_dir
    dx = 52.5 - x
    if not (np.isfinite(dx) and np.isfinite(y)) or dx < 0.5:
        return 0.0
    dist = max(float(np.hypot(dx, y)), 1.0)
    angle = abs(np.arctan2(GOAL_HALF_WIDTH - y, max(dx, 0.01)) - np.arctan2(-GOAL_HALF_WIDTH - y, max(dx, 0.01)))
    return float(1.0 / (1.0 + np.exp(-(-2.884 + 2.39 * angle - 0.0577 * dist))))


# ---------------------------------------------------------------------------------------------------------------------
# xG model v2: logistic regression fitted on StatsBomb open data (about 10,700 shots with freeze frames, World Cup 2022
# held out for testing). Besides distance and angle it knows headers, direct free kicks, defenders between the ball and
# the goal, where the goalkeeper stands, and opponents close to the shooter - all read here from the PFF tracking data.
# Held-out World Cup 2022: AUC 0.815 / log loss 0.271 (old distance+angle model 0.768 / 0.296; StatsBomb's xG 0.820 / 0.263).
# ---------------------------------------------------------------------------------------------------------------------
def _load_xg_v2() -> Optional[dict]:
    path = ASSETS_DIR / "xg_model_v2.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


XG_V2 = _load_xg_v2()


def _in_triangle(p, a, b, c) -> bool:
    def s_(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s_(p, a, b), s_(p, b, c), s_(p, c, a)
    return not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and ((d1 > 0) or (d2 > 0) or (d3 > 0)))


def xg_v2_features(ball: np.ndarray, opponents: np.ndarray, gk: Optional[np.ndarray], header: bool, free_kick: bool) -> dict:
    """Features in a goal-centred frame (metres): goal-line centre at (0, 0), posts at (0, +-3.66), pitch at x < 0."""
    dx, dy = -float(ball[0]), float(ball[1])
    dist = float(np.hypot(dx, dy))
    angle = float(abs(np.arctan2(GOAL_HALF_WIDTH - dy, max(dx, 0.01)) - np.arctan2(-GOAL_HALF_WIDTH - dy, max(dx, 0.01))))
    p1, p2 = np.array([0.0, GOAL_HALF_WIDTH]), np.array([0.0, -GOAL_HALF_WIDTH])
    n_block = sum(1 for o in opponents if _in_triangle(o, ball, p1, p2))
    n_close = sum(1 for o in opponents if np.hypot(*(o - ball)) < 3.0)
    if gk is not None:
        n_close += int(np.hypot(*(gk - ball)) < 3.0)
        gk_dist = float(np.hypot(*gk))
        v = -ball / max(float(np.linalg.norm(ball)), 1e-6)
        rel = gk - ball
        gk_off = float(abs(rel[0] * v[1] - rel[1] * v[0]))
    else:  # no keeper found: assume he is on his line, in the middle
        gk_dist, gk_off = 0.5, 0.0
    return {"distance": dist, "log_distance": float(np.log(max(dist, 1.0))), "angle": angle, "header": float(header),
            "free_kick": float(free_kick), "n_block": float(n_block), "gk_dist": gk_dist, "gk_off_line": gk_off, "n_close": float(n_close)}


def xg_v2_from_features(f: dict, model: Optional[dict] = None) -> float:
    mdl = model or XG_V2
    z = mdl["intercept"]
    for name in mdl["features"]:
        v = f[name]
        cap = mdl.get("caps", {}).get(name)
        if cap is not None:
            v = min(v, cap)
        z += mdl["coef"][name] * v
    return float(1.0 / (1.0 + np.exp(-z)))


def pass_success_surface(grid: PitchGrid, ball_xy: np.ndarray) -> np.ndarray:
    """P(a pass from the ball to each cell is completed): 0.97 * exp(-distance / 70 m)."""
    xx, yy = grid.mesh()
    if ball_xy is None or not np.all(np.isfinite(ball_xy)):
        return np.full_like(xx, 0.5)
    d = np.sqrt((xx - ball_xy[0]) ** 2 + (yy - ball_xy[1]) ** 2)
    return 0.97 * np.exp(-d / 70.0)


def pass_success_probability(origin: np.ndarray, dest: np.ndarray, pc_att_at_dest: float) -> float:
    """Heuristic completion probability of one pass: distance decay x how much the team controls the target spot."""
    d = np.hypot(np.asarray(dest)[..., 0] - origin[0], np.asarray(dest)[..., 1] - origin[1])
    out = np.clip(0.97 * np.exp(-d / 70.0) * (0.35 + 0.65 * np.asarray(pc_att_at_dest)), 0.01, 0.99)
    return float(out) if np.ndim(out) == 0 else out


def lane_free_probability(origin: np.ndarray, targets: np.ndarray, opp_pos: np.ndarray, opp_vel: np.ndarray,
                          params: "PitchControlParams", n_points: int = 7) -> np.ndarray:
    """P(no opponent gets to the ball on its way from `origin` to each target)  -> shape (K,).

    The ball is sampled at `n_points` spots along each pass lane. An opponent can cut the pass at a spot if he
    can be there before the ball (same time-to-intercept logistic as pitch control). Per opponent we keep the
    worst spot on the lane; opponents are then combined as independent chances."""
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(opp_pos) == 0:
        return np.ones(len(targets))
    u = np.linspace(0.15, 0.85, n_points)
    pts = origin[None, None, :] + (targets[:, None, :] - origin[None, None, :]) * u[None, :, None]  # (K, S, 2)
    t_ball = np.linalg.norm(pts - origin[None, None, :], axis=2) / params.average_ball_speed  # (K, S)
    r_react = opp_pos + np.nan_to_num(opp_vel) * params.reaction_time  # (J, 2)
    dist = np.linalg.norm(pts[:, :, None, :] - r_react[None, None, :, :], axis=3)  # (K, S, J)
    t_opp = params.reaction_time + dist / params.max_player_speed
    k = np.pi / np.sqrt(3.0) / params.tti_sigma
    p_arrive = 1.0 / (1.0 + np.exp(np.clip(-k * (t_ball[:, :, None] - t_opp), -50, 50)))  # (K, S, J)
    return np.prod(1.0 - p_arrive.max(axis=1), axis=1)


def _load_pass_calibration(model: str) -> tuple[float, float]:
    """Optional Platt calibration learned by validate_metrics.py --fit-calibration: p' = sigmoid(a + b*logit(p))."""
    f = ASSETS_DIR / f"pass_calibration_{model}.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            return float(d["a"]), float(d["b"])
        except Exception:
            pass
    return 0.0, 1.0


PASS_FEATURES = ["log_distance", "forward_m", "pc_target", "log_lane_free", "opp_near_target", "opp_near_passer"]


def _load_fitted_pass_model() -> Optional[dict]:
    """Logistic pass-completion model learned by validate_metrics.py --fit-calibration (assets/pass_model_fitted.json)."""
    f = ASSETS_DIR / "pass_model_fitted.json"
    if f.exists():
        try:
            d = json.loads(f.read_text())
            if d.get("features") == PASS_FEATURES:
                for k in ("mu", "sd", "w"):
                    d[k] = np.asarray(d[k], dtype=float)
                d["a"] = float(d["a"])
                return d
        except Exception:
            pass
    return None


class EPVModel:
    """Laurie Shaw's 32x50 Expected Possession Value grid (valid for a team attacking +x)."""

    def __init__(self, grid: PitchGrid, path: Optional[Path] = None):
        self.grid = grid
        path = Path(path) if path else ASSETS_DIR / "EPV_grid.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[epv] downloading EPV grid to {path} ...")
            urllib.request.urlretrieve(EPV_GRID_URL, path)
        g = np.loadtxt(path, delimiter=",")
        if g.shape != (grid.ny, grid.nx):
            raise ValueError(f"EPV grid has shape {g.shape}, expected {(grid.ny, grid.nx)}")
        self._g = g

    def surface(self, attack_dir: int = 1) -> np.ndarray:
        return self._g if attack_dir == 1 else self._g[::-1, ::-1]

    def value(self, x, y, attack_dir: int = 1):
        """EPV at one point (or arrays of points). NaN positions return NaN.

        The grid is coarse (about 2.1 m x 2.1 m cells). Looking up only the cell that contains the ball makes the value jump
        when the ball crosses a cell border, and a pass of a few metres can show a "gain" that is only a border effect.
        So the value is interpolated bilinearly between the four nearest cell centres (v2). Away from cell borders nothing changes.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        g = self.surface(attack_dir)
        gr = self.grid
        # position in "cell units" measured from the centre of cell 0
        fx = np.clip((np.nan_to_num(x) + gr.length / 2) / gr.dx - 0.5, 0.0, gr.nx - 1.0)
        fy = np.clip((np.nan_to_num(y) + gr.width / 2) / gr.dy - 0.5, 0.0, gr.ny - 1.0)
        x0 = np.minimum(np.floor(fx).astype(int), gr.nx - 2)
        y0 = np.minimum(np.floor(fy).astype(int), gr.ny - 2)
        tx, ty = fx - x0, fy - y0
        v = ((1 - ty) * ((1 - tx) * g[y0, x0] + tx * g[y0, x0 + 1])
             + ty * ((1 - tx) * g[y0 + 1, x0] + tx * g[y0 + 1, x0 + 1]))
        return np.where(np.isnan(x) | np.isnan(y), np.nan, v)


# ===========================================================================
# 2./3. DAS and OBSO surfaces
# ===========================================================================
def defensive_line_x(pos_def: np.ndarray, attack_dir: int) -> float:
    """x of the defensive line in the ATTACKING team's frame (attacked goal at +x).
    Uses the second-last defender, goalkeeper included (the offside-line idea)."""
    if pos_def.shape[0] < 3:
        return 0.0
    xs_att = np.sort(pos_def[:, 0] * attack_dir)[::-1]  # deepest defender (the GK) has the largest value
    return float(xs_att[1])


def das_surface(pc_att: np.ndarray, grid: PitchGrid, attack_dir: int, def_line_x_att: float) -> np.ndarray:
    """Dangerous Attacking Space per cell = PC_att x xG x line-penetration weight.

    line weight is a smooth step: ~1 for space BEHIND the defensive line, ~0.4 well in front of it.
    """
    xx, _ = grid.mesh()
    x_att = xx * attack_dir  # attacker frame: attacking goal at +x
    w = 0.4 + 0.6 / (1.0 + np.exp(-(x_att - def_line_x_att) / 2.0))
    return pc_att * xg_surface(grid, attack_dir) * w


# ===========================================================================
# The engine (one instance per match; caches everything it computes)
# ===========================================================================
_ACTION_GROUP = {"PA": "pass", "CR": "pass", "SH": "shot", "BC": "carry", "CH": "duel", "CL": "clearance", "RE": "rebound"}


class MetricsEngine:
    def __init__(self, match: MatchData, pc_params: Optional[PitchControlParams] = None,
                 epv_path: Optional[Path] = None, cache_size: int = 256, pass_model: str = "auto"):
        self.m = match
        # "fitted": logistic model on 6 pass features (used automatically when assets/pass_model_fitted.json exists)
        # "lane"  : distance x target control x open passing lane (+ optional Platt calibration)
        # "simple": distance x target control
        self._pass_fit = _load_fitted_pass_model()
        if pass_model == "auto":
            pass_model = "fitted" if self._pass_fit is not None else "lane"
        elif pass_model == "fitted" and self._pass_fit is None:
            pass_model = "lane"
        self.pass_model = pass_model
        self._pass_cal = _load_pass_calibration("lane")
        self.grid = match.grid
        self.pc_params = pc_params or PitchControlParams()
        self.epv = EPVModel(self.grid, epv_path)
        self._cache: OrderedDict = OrderedDict()
        self._clock = threading.RLock()  # the API and the warm-up thread share this cache
        self._cache_size = cache_size
        ev = self.m.events
        # team of the latest event that HAS a team (some events, e.g. stoppages, carry none)
        # A challenge (duel) event carries the team of the challenger, but the ball goes to whoever WON it: in 27 % of the duels
        # of match 3816 the winner is the other team. So for duels with a known winner the possession goes to the winner's team.
        poss = ev["side"].where(ev["side"].isin(["home", "away"])) if len(ev) else None
        if len(ev) and "possessionEvents.challengeWinnerPlayerId" in ev.columns:
            pid2side = {str(r.player_id): r.side for r in self.m.players.itertuples() if r.player_id}
            wid = ev["possessionEvents.challengeWinnerPlayerId"].astype(str).str.replace(r"\.0$", "", regex=True)
            wside = wid.map(pid2side)
            poss = poss.where(~((ev["poss_type"] == "CH") & wside.notna()), wside)
        self._side_ff = poss.ffill().to_numpy(dtype=object) if len(ev) else np.array([], dtype=object)
        self._actions: Optional[pd.DataFrame] = None
        self._timeline: dict = {}

    # ------------------------------------------------------------------ helpers
    def _remember(self, key, value):
        with self._clock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return value

    def _lookup(self, key):
        with self._clock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def possession_side(self, idx: int) -> Optional[str]:
        """'home' / 'away' = team of the latest event that has a team, at or before this frame (None before the first)."""
        ei = int(self.m.last_event_idx[idx]) if len(self.m.events) else -1
        if ei < 0:
            return None
        s = self._side_ff[ei]
        return s if s in ("home", "away") else None

    def shot_xg(self, ei: int, detail: bool = False):
        """xG of the shot in event row `ei` (model v2 when its file is present, else the distance-and-angle model).
        Penalties = PENALTY_XG. With detail=True returns (xg, features) for explanations."""
        m, ev = self.m, self.m.events
        e = ev.iloc[int(ei)]
        fi, side = int(e["frame_idx"]), e["side"]
        x, y = float(e["ball_x"]), float(e["ball_y"])
        if e.get("setpiece_type") == "P":
            return (PENALTY_XG, {"penalty": True}) if detail else PENALTY_XG
        if fi < 0 or side not in ("home", "away") or not (np.isfinite(x) and np.isfinite(y)):
            return (0.0, {}) if detail else 0.0
        adir = m.attack_direction(side, fi)
        if XG_V2 is None:
            q = xg_at(x, y, adir)
            return (q, {"model": "distance+angle"}) if detail else q
        half = self.grid.length / 2
        ball = np.array([x * adir - half, y * adir])
        sides = m.players["side"].to_numpy()
        opp = [s for s in m.present_slots(fi) if sides[s] != side and np.isfinite(m.X[fi, s])]
        pts = np.array([[m.X[fi, s] * adir - half, m.Y[fi, s] * adir] for s in opp]) if opp else np.zeros((0, 2))
        gk = None
        if len(pts):
            # the goalkeeper: the opponent listed as GK in the line-up; if none is on the pitch, the deepest defender
            pos_col = m.players["position"].astype(str).str.upper().to_numpy() if "position" in m.players.columns else None
            gk_rows = [j for j, s_ in enumerate(opp) if pos_col is not None and pos_col[s_] == "GK"]
            k = gk_rows[0] if gk_rows else int(np.argmax(pts[:, 0]))
            gk = pts[k]
            pts = np.delete(pts, k, axis=0)
        body = str(e.get("possessionEvents.bodyType") or "")
        f = xg_v2_features(ball, pts, gk, header=(body == "HE"), free_kick=(e.get("setpiece_type") == "F"))
        q = xg_v2_from_features(f)
        return (q, {**f, "model": "v2"}) if detail else q

    def ball_xy(self, idx: int) -> Optional[np.ndarray]:
        b = np.array([self.m.ball_x[idx], self.m.ball_y[idx]], dtype=float)
        return b if np.all(np.isfinite(b)) else None

    def _goalkeepers(self, idx: int, pos: np.ndarray, is_home: np.ndarray) -> np.ndarray:
        """The goalkeeper of each side = the player closest to his own goal line."""
        gk = np.zeros(len(pos), dtype=bool)
        for side, mask in (("home", is_home), ("away", ~is_home)):
            if mask.any():
                own = self.m.attack_direction(side, idx)  # +1: attacks +x, so his own goal is at -x
                depth = pos[:, 0] * own
                depth = np.where(mask, depth, np.inf)
                gk[int(depth.argmin())] = True
        return gk

    @staticmethod
    def _onside(pos: np.ndarray, is_att: np.ndarray, adir: int, ball: Optional[np.ndarray], tol: float = 0.2) -> np.ndarray:
        """Boolean mask of players to KEEP: everyone except attackers beyond the offside line
        (second-last defender incl. GK, the ball, or the halfway line - whichever is furthest)."""
        xs_def = np.sort(pos[~is_att, 0] * adir)[::-1]
        line = max(float(xs_def[1]), (ball[0] * adir) if ball is not None else 0.0, 0.0) + tol
        return ~(is_att & (pos[:, 0] * adir > line))

    # ------------------------------------------------------------------ 1. pitch control
    def pitch_control(self, idx: int, per_player: bool = False) -> dict:
        """Pitch control at frame row `idx`. 'attacking' team = team in possession (home if unknown)."""
        key = ("pc", idx, per_player, self.pc_params.key())
        hit = self._lookup(key)
        if hit is not None:
            return hit
        m = self.m
        slots = m.present_slots(idx)
        is_home = (m.players["side"].to_numpy()[slots] == "home")
        poss = self.possession_side(idx)
        att_side = poss or "home"  # only a calculation convention when nobody has touched the ball yet
        is_att = is_home if att_side == "home" else ~is_home
        adir = m.attack_direction(att_side, idx)
        pos = np.column_stack([m.X[idx, slots], m.Y[idx, slots]]).astype(float)
        ball = self.ball_xy(idx)
        is_gk = self._goalkeepers(idx, pos, is_home)
        if self.pc_params.offsides and (~is_att).sum() >= 2:
            keep = self._onside(pos, is_att, adir, ball)
            slots, is_att, pos, is_gk = slots[keep], is_att[keep], pos[keep], is_gk[keep]
        vel = np.column_stack([m.VX[idx, slots], m.VY[idx, slots]]).astype(float)
        acc = np.column_stack([m.AX[idx, slots], m.AY[idx, slots]]).astype(float)
        pc_att, pc_def, ppcf = compute_pitch_control(pos, vel, acc, is_att, ball, self.grid,
                                                    self.pc_params, per_player=per_player, is_gk=is_gk)
        pc_home = pc_att if att_side == "home" else pc_def
        res = {"attacking_side": att_side, "possession_known": poss is not None, "slots": slots, "is_att": is_att, "pc_att": pc_att, "pc_home": pc_home,
               "ppcf_players": ppcf}
        return self._remember(key, res)

    # ------------------------------------------------------------------ 2.-4. frame metrics
    def frame_metrics(self, idx: int, include_obso: bool = True) -> dict:
        """Everything the pitch view needs for one frame (cached)."""
        key = ("frame", idx, include_obso, self.pc_params.key())
        hit = self._lookup(key)
        if hit is not None:
            return hit
        m, grid = self.m, self.grid
        pcr = self.pitch_control(idx)
        att_side = pcr["attacking_side"]
        def_side = "away" if att_side == "home" else "home"
        adir = m.attack_direction(att_side, idx)
        slots, is_att = pcr["slots"], pcr["is_att"]
        pos = np.column_stack([m.X[idx, slots], m.Y[idx, slots]]).astype(float)
        ball = self.ball_xy(idx)

        # DAS: the accessible-space package (reference). Own estimate only if the package is missing.
        def_pos = pos[~is_att]
        dl = defensive_line_x(def_pos, adir) if len(def_pos) else 0.0
        das_src = ref.das_source()
        as_value = float("nan")
        if ref.das_available():
            r = ref.das_frames(m, [(idx, att_side, adir, -1)], want_surface=True)[0]
            if r is not None:
                das, das_value, as_value = r["surface"], r["das"], r["acc"]  # surface sums to das_value (danger-weighted m2)
            else:  # no ball position -> cannot simulate passes
                das, das_value = np.zeros((grid.ny, grid.nx)), float("nan")
                das_src = "unavailable for this frame (no ball position)"
        else:
            das = das_surface(pcr["pc_att"], grid, adir, dl)  # dl is already in the attacker frame
            das_value = float(das.sum() * grid.cell_area)  # "danger-weighted square metres"

        # EPV at the ball (value for the team in possession) and for home (signed)
        epv_ball = float(self.epv.value(ball[0], ball[1], adir)) if ball is not None else float("nan")
        epv_home = (epv_ball if att_side == "home" else -epv_ball) if np.isfinite(epv_ball) else float("nan")

        # OBSO as in C-OBSO: grid = PC(attackers) x transition(ball-centred) x EPV/max; a player's OBSO is the grid value
        # at his own cell, the team value is the sum of the grid (offside attackers are not in `slots`; the ball carrier
        # is left out because OBSO is defined for off-ball players).
        obso_players = []
        obso_total = 0.0
        obso_surface = np.zeros((grid.ny, grid.nx))
        if include_obso and ball is not None:
            obso_surface = ref.cobso_grid(pcr["pc_att"], ball, self.epv.surface(adir), grid)
            obso_total = float(obso_surface.sum())
            carrier = None
            att_idx = np.flatnonzero(is_att)
            if att_idx.size:
                dd = np.hypot(pos[att_idx, 0] - ball[0], pos[att_idx, 1] - ball[1])
                if dd.min() < 3.0:
                    carrier = int(att_idx[dd.argmin()])
            for n in att_idx:
                if n == carrier:
                    continue
                jy, jx = grid.cell_index(float(pos[n, 0]), float(pos[n, 1]))
                val = float(obso_surface[int(jy), int(jx)])
                obso_players.append({
                    "slot": int(slots[n]), "obso": val,
                    "target_x": float(pos[n, 0]), "target_y": float(pos[n, 1]), "target_value": val,
                })

        res = {
            "idx": idx, "frame_num": int(m.frame_num[idx]), "period": int(m.period[idx]),
            "time_s": float(m.time_s[idx]), "match_minute": float(m.match_minute[idx]),
            "attacking_side": att_side, "attack_dir": adir, "defending_side": def_side,
            "possession_known": pcr["possession_known"], "ball_tracked": ball is not None,
            "pc_home": pcr["pc_home"], "das": das, "das_value": das_value, "as_value": as_value, "das_source": das_src,
            "defensive_line_x": dl,
            "epv_ball": epv_ball, "epv_home": epv_home, "obso_players": obso_players, "obso_total": obso_total,
            "obso_grid": obso_surface,
        }
        return self._remember(key, res)

    # ------------------------------------------------------------------ player share of space (used for fatigue)
    def player_control_area(self, idx: int) -> dict:
        """slot -> area (m^2) the player controls ON HIS OWN (sum of his PPCF x cell area)."""
        pcr = self.pitch_control(idx, per_player=True)
        areas = pcr["ppcf_players"].reshape(len(pcr["slots"]), -1).sum(axis=1) * self.grid.cell_area
        return {int(s): float(a) for s, a in zip(pcr["slots"], areas)}

    # ------------------------------------------------------------------ 4. EPV progression
    def epv_timeline(self, step_s: float = 1.0) -> pd.DataFrame:
        """EPV of the ball location over the match, signed from the HOME team's view.
        One row every `step_s` seconds: positive = home has the more valuable possession."""
        if step_s in self._timeline:
            return self._timeline[step_s]
        m = self.m
        step = max(1, int(round(step_s * m.fps)))
        idxs = np.arange(0, m.n_frames, step)
        rows = []
        for i in idxs:
            side = self.possession_side(int(i))
            bx, by = m.ball_x[i], m.ball_y[i]
            if side is None or not np.isfinite(bx):
                continue
            adir = m.attack_direction(side, int(i))
            v = float(self.epv.value(bx, by, adir))
            rows.append((int(m.frame_num[i]), int(m.period[i]), float(m.time_s[i]), float(m.match_minute[i]),
                         side, v if side == "home" else -v))
        df = pd.DataFrame(rows, columns=["frame_num", "period", "time_s", "match_minute", "possession", "epv_home"])
        self._timeline[step_s] = df
        return df

    # ------------------------------------------------------------------ 5./6. VAEP and OBV
    def action_values(self) -> pd.DataFrame:
        """One row per on-ball event with its EPV change, VAEP and OBV approximations.

        State value V = what the team in possession is worth at the ball's location (EPV grid).
        Before an action:  V_before = EPV(origin, actor's attacking direction).
        After the action:
           * success (pass completed, carry/dribble kept, tackle won):   V_after = EPV(end location)
           * failure (pass lost, dispossessed):                           V_after = -EPV(end location, opponent's view)
           * shot: goal -> 1.0, anything else -> 0.0
        VAEP  = V_after - V_before   (realised outcome - the "what actually happened" rating)
        OBV   = same states but with the EXPECTED outcome of the action (pass success probability
                from the pitch-control model), so luck is smoothed out. Credited by action type.
        """
        if self._actions is not None:
            return self._actions
        m, ev = self.m, self.m.events
        if not len(ev):
            self._actions = pd.DataFrame()
            return self._actions
        rows = []
        for i, e in ev.iterrows():
            ptype = e["poss_type"]
            if ptype not in _ACTION_GROUP or e["frame_idx"] < 0 or e["side"] not in ("home", "away"):
                continue
            fi, nf = int(e["frame_idx"]), int(e["next_frame_idx"])
            side = e["side"]
            opp = "away" if side == "home" else "home"
            adir, odir = m.attack_direction(side, fi), m.attack_direction(opp, fi)
            ox, oy = e["ball_x"], e["ball_y"]
            ex, ey = e["end_ball_x"], e["end_ball_y"]
            if not (np.isfinite(ox) and np.isfinite(oy)):
                continue
            v_before = float(self.epv.value(ox, oy, adir))
            group = _ACTION_GROUP[ptype]
            success, v_after, obv_after = True, np.nan, np.nan
            obv_progress = np.nan  # OBV's "if the pass is completed" part: value at the intended target minus value now

            if group == "shot":
                goal = bool(e["is_goal"]) if "is_goal" in ev.columns else (e["shot_outcome"] == "G")
                success = bool(goal)
                v_after = 1.0 if goal else 0.0
                # expected value of the shot: xG of the spot. A penalty is NOT an open-play shot: the spot-based model would
                # say ~0.12, but penalties are scored about 3 times in 4, so use the usual conversion rate.
                is_pen = "setpiece_type" in ev.columns and e["setpiece_type"] == "P"
                obv_after = self.shot_xg(ev.index.get_loc(i))
            elif group in ("pass",):
                outcome = e["pass_outcome"] if ptype == "PA" else e["cross_outcome"]
                success = outcome == "C"
                if np.isfinite(ex) and np.isfinite(ey):
                    v_succ = float(self.epv.value(ex, ey, adir))
                    v_fail = -float(self.epv.value(ex, ey, odir))
                    v_after = v_succ if success else v_fail
                    # expected value of the INTENDED pass: the player he aimed at (known even if the pass failed)
                    ts = int(e["target_slot"]) if e["target_slot"] >= 0 else int(e["receiver_slot"])
                    tx, ty = ((m.X[fi, ts], m.Y[fi, ts]) if ts >= 0 and np.isfinite(m.X[fi, ts]) else (ex, ey))
                    p = float(self.pass_probabilities(fi, side, np.array([ox, oy]), np.array([[tx, ty]]))[0])
                    v_t_succ = float(self.epv.value(tx, ty, adir))
                    v_t_fail = -float(self.epv.value(tx, ty, odir))
                    obv_after = p * v_t_succ + (1 - p) * v_t_fail  # progress if completed, minus the expected cost of losing the ball
                    obv_progress = v_t_succ - v_before
            elif group in ("carry", "rebound"):
                if np.isfinite(ex) and np.isfinite(ey):
                    v_after = float(self.epv.value(ex, ey, adir))
                    obv_after = v_after
            elif group == "duel":
                # a challenge: valued neutrally (small credit by ball progress); "success" = this player won the duel when the
                # data says who won (challengeWinnerPlayerId), otherwise it stays True
                w = e.get("possessionEvents.challengeWinnerPlayerId") if hasattr(e, "get") else None
                try:
                    if w is not None and np.isfinite(float(w)) and e["player_id"]:
                        success = str(int(float(w))) == str(e["player_id"])
                except (TypeError, ValueError):
                    pass
                if np.isfinite(ex) and np.isfinite(ey):
                    v_after = float(self.epv.value(ex, ey, adir))
                    obv_after = 0.5 * (v_after + v_before)
            elif group == "clearance":
                if np.isfinite(ex) and np.isfinite(ey):
                    v_after = -0.5 * float(self.epv.value(ex, ey, odir))  # ball is contested: half credit to the opponent
                    obv_after = v_after
            if not np.isfinite(v_after):
                continue
            rows.append({
                "event_idx": int(i), "frame_idx": fi, "frame_num": int(e["frame_num"]), "period": int(e["period"]) if pd.notna(e["period"]) else 0,
                "time_s": float(m.time_s[fi]), "match_minute": float(m.match_minute[fi]), "bin_id": int(m.bin_id[fi]),
                "side": side, "player_id": e["player_id"], "slot": int(e["actor_slot"]),
                "ax": float(ox * adir),
                "action": group, "poss_type": ptype, "success": success,
                "x": float(ox), "y": float(oy), "end_x": float(ex) if np.isfinite(ex) else np.nan, "end_y": float(ey) if np.isfinite(ey) else np.nan,
                "epv_before": v_before, "epv_after": float(v_after), "delta_epv": float(v_after - v_before),
                "vaep": float(v_after - v_before),
                "obv": float(obv_after - v_before) if np.isfinite(obv_after) else float(v_after - v_before),
                "obv_progress": float(obv_progress) if np.isfinite(obv_progress) else (float(obv_after - v_before) if np.isfinite(obv_after) else float(v_after - v_before)),
                "receiver_slot": int(e["receiver_slot"]) if group == "pass" else -1,
            })
        df = pd.DataFrame(rows)
        if len(df):
            # "VAEP above average": the raw value minus what a typical action of the SAME kind from the SAME third of
            # the pitch is worth in this match. Raw VAEP is built from an EPV grid whose scale is not balanced for every
            # match (losing the ball always costs more than a short pass gains), so raw totals are not "goals". The
            # centred value is zero for an average action and is the one to use for averages, rankings and comparisons.
            zone = np.select([df["ax"] < -17.5, df["ax"] > 17.5], [0, 2], default=1)
            grp = df.groupby(["action", zone])["vaep"]
            n_cell, mu_cell = grp.transform("size"), grp.transform("mean")
            mu_act = df.groupby("action")["vaep"].transform("mean")
            df["vaep_oa"] = df["vaep"] - np.where(n_cell >= 8, mu_cell, mu_act)
        self._actions = df
        return self._actions

    def pass_features(self, idx: int, side: str, origin: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Six numbers describing each pass option (rows = targets), see PASS_FEATURES:
        log(1+distance), metres gained towards goal, pitch control at the target, log(open-lane chance),
        distance from the target to the nearest opponent, distance from the passer to the nearest opponent."""
        m = self.m
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        origin = np.asarray(origin, dtype=float)
        slots = m.present_slots(idx)
        opp = slots[m.players["side"].to_numpy()[slots] != side]
        opp_pos = np.column_stack([m.X[idx, opp], m.Y[idx, opp]]).astype(float)
        opp_vel = np.column_stack([m.VX[idx, opp], m.VY[idx, opp]]).astype(float)
        d = np.hypot(targets[:, 0] - origin[0], targets[:, 1] - origin[1])
        fwd = np.clip((targets[:, 0] - origin[0]) * m.attack_direction(side, idx), -40.0, 60.0)
        pc = self.pc_side_at_points(idx, side, targets)
        lane = np.clip(lane_free_probability(origin, targets, opp_pos, opp_vel, self.pc_params), 0.01, 1.0)
        if len(opp_pos):
            near_t = np.clip(np.min(np.linalg.norm(targets[:, None, :] - opp_pos[None, :, :], axis=2), axis=1), 0.0, 20.0)
            near_p = float(np.clip(np.min(np.linalg.norm(opp_pos - origin[None, :], axis=1)), 0.0, 20.0))
        else:
            near_t, near_p = np.full(len(targets), 20.0), 20.0
        return np.column_stack([np.log1p(d), fwd, pc, np.log(lane), near_t, np.full(len(targets), near_p)])

    def pass_probabilities(self, idx: int, side: str, origin: np.ndarray, targets: np.ndarray,
                           model: Optional[str] = None, calibrated: bool = True) -> np.ndarray:
        """Completion probability of passes from `origin` to each target point, frame `idx`, team `side`.

        simple:  0.97 * exp(-d/70) * (0.35 + 0.65 * pitch control of the passing team at the target)
        lane  :  simple  *  P(no opponent can cut the passing lane), optional Platt recalibration
        fitted:  logistic regression on pass_features(), learned from the real passes of the match(es)"""
        model = model or self.pass_model
        m = self.m
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        origin = np.asarray(origin, dtype=float)
        if model == "fitted" and self._pass_fit is not None:
            f = self._pass_fit
            z = f["a"] + ((self.pass_features(idx, side, origin, targets) - f["mu"]) / f["sd"]) @ f["w"]
            return np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.99)
        pc = self.pc_side_at_points(idx, side, targets)
        d = np.hypot(targets[:, 0] - origin[0], targets[:, 1] - origin[1])
        p = 0.97 * np.exp(-d / 70.0) * (0.35 + 0.65 * pc)
        if model == "lane":
            slots = m.present_slots(idx)
            opp = slots[m.players["side"].to_numpy()[slots] != side]
            opp_pos = np.column_stack([m.X[idx, opp], m.Y[idx, opp]]).astype(float)
            opp_vel = np.column_stack([m.VX[idx, opp], m.VY[idx, opp]]).astype(float)
            p = p * lane_free_probability(origin, targets, opp_pos, opp_vel, self.pc_params)
        p = np.clip(p, 0.01, 0.99)
        if calibrated and model == "lane" and self._pass_cal != (0.0, 1.0):
            a, b = self._pass_cal
            logit = np.log(p / (1 - p))
            p = np.clip(1.0 / (1.0 + np.exp(-(a + b * logit))), 0.01, 0.99)
        return p

    def pc_side_at_points(self, idx: int, side: str, points: np.ndarray) -> np.ndarray:
        """Pitch control of `side` ('home'/'away') at a few chosen points of frame `idx`.
        Much cheaper than a full grid (used for pass options and action values)."""
        m = self.m
        slots = m.present_slots(idx)
        is_side = (m.players["side"].to_numpy()[slots] == side)
        pos = np.column_stack([m.X[idx, slots], m.Y[idx, slots]]).astype(float)
        vel = np.column_stack([m.VX[idx, slots], m.VY[idx, slots]]).astype(float)
        acc = np.column_stack([m.AX[idx, slots], m.AY[idx, slots]]).astype(float)
        is_gk = self._goalkeepers(idx, pos, (m.players["side"].to_numpy()[slots] == "home"))
        pc, _, _ = compute_pitch_control(pos, vel, acc, is_side, self.ball_xy(idx), self.grid, self.pc_params,
                                         per_player=False, points=np.asarray(points, dtype=float).reshape(-1, 2),
                                         is_gk=is_gk)
        return pc

    def _pc_att_at(self, idx: int, side: str, x: float, y: float) -> float:
        return float(self.pc_side_at_points(idx, side, np.array([[x, y]]))[0])

    # ------------------------------------------------------------------ JSON helpers for the API
    def frame_payload(self, idx: int, round_to: int = 3) -> dict:
        """JSON-ready dictionary for one frame: players, ball, grids and scalar metrics."""
        m = self.m
        fm = self.frame_metrics(idx, include_obso=True)
        slots = m.present_slots(idx)
        pl = m.players
        players = []
        for s in slots:
            r = pl.iloc[int(s)]
            players.append({
                "slot": int(s), "side": r["side"], "jersey": r["jersey"], "player_id": r["player_id"], "name": r["name"],
                "x": round(float(m.X[idx, s]), 2), "y": round(float(m.Y[idx, s]), 2),
                "speed": None if np.isnan(m.SPEED[idx, s]) else round(float(m.SPEED[idx, s]), 2),
            })
        obso = {o["slot"]: o for o in fm["obso_players"]}
        for p in players:
            if p["slot"] in obso:
                p["obso"] = round(obso[p["slot"]]["obso"], 5)  # C-OBSO value at the player's own cell
                p["obso_target"] = [round(obso[p["slot"]]["target_x"], 2), round(obso[p["slot"]]["target_y"], 2)]
        return {
            "match_id": m.match_id, "frame_num": fm["frame_num"], "idx": idx, "period": fm["period"],
            "time_s": round(fm["time_s"], 2), "match_minute": round(fm["match_minute"], 2),
            "attacking_side": fm["attacking_side"], "attack_dir": fm["attack_dir"],
            "possession_known": fm["possession_known"], "ball_tracked": fm["ball_tracked"],
            "ball": None if not np.isfinite(m.ball_x[idx]) else {"x": round(float(m.ball_x[idx]), 2), "y": round(float(m.ball_y[idx]), 2)},
            "players": players,
            "grid": {"nx": self.grid.nx, "ny": self.grid.ny, "length": self.grid.length, "width": self.grid.width},
            "pitch_control_home": np.round(fm["pc_home"], round_to).tolist(),
            "das": np.round(fm["das"], 6).tolist(),
            "obso_grid": np.round(fm["obso_grid"], 5).tolist(),
            "das_source": fm["das_source"],
            "metrics": {
                "das_value": None if not np.isfinite(fm["das_value"]) else round(fm["das_value"], 4),
                "as_value": None if not np.isfinite(fm["as_value"]) else round(fm["as_value"], 1),
                "obso_total": round(fm["obso_total"], 4),
                "epv_ball": None if not np.isfinite(fm["epv_ball"]) else round(fm["epv_ball"], 5),
                "epv_home": None if not np.isfinite(fm["epv_home"]) else round(fm["epv_home"], 5),
                "pc_home_share": round(float(fm["pc_home"].mean()), 4),
                "defensive_line_x": round(fm["defensive_line_x"], 2),
            },
        }


# ===========================================================================
# Self-test:  python metrics_engine.py --match 3816
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    md = load_match(args.match, data_dir=args.data_dir, verbose=True)
    eng = MetricsEngine(md)
    print(f"\nMatch {md.match_id}: {md.n_frames:,} frames, {md.fps:.2f} fps")

    i = md.n_frames // 3
    t = time.time()
    fm = eng.frame_metrics(i)
    print(f"\nFrame row {i} (frameNum {md.frame_num[i]}): computed in {time.time() - t:.2f}s (1st call), "
          f"possession: {fm['attacking_side']}")
    t = time.time()
    eng.frame_metrics(i)
    print(f"  cached call: {(time.time() - t) * 1000:.2f} ms")
    pc = fm["pc_home"]
    print(f"  pitch-control grid shape {pc.shape}, min {pc.min():.3f}, max {pc.max():.3f}, home share of pitch {pc.mean():.3f}")
    print(f"  DAS {fm['das_value']:.4f} m2 [{fm['das_source']}], OBSO total {fm['obso_total']:.4f} [C-OBSO], EPV at ball {fm['epv_ball']:.4f}")
    print("  top OBSO attackers:")
    for o in sorted(fm["obso_players"], key=lambda o: -o["obso"])[:3]:
        r = md.players.iloc[o["slot"]]
        print(f"    {r['name']:<26} obso={o['obso']:.4f} at ({o['target_x']:.1f}, {o['target_y']:.1f})")

    t = time.time()
    tl = eng.epv_timeline(1.0)
    print(f"\nEPV timeline: {len(tl)} points in {time.time() - t:.1f}s, range {tl['epv_home'].min():.3f} .. {tl['epv_home'].max():.3f}")

    t = time.time()
    av = eng.action_values()
    print(f"\nAction values: {len(av)} on-ball actions in {time.time() - t:.1f}s")
    if len(av):
        print(av.groupby("action")[["delta_epv", "vaep", "obv"]].agg(["count", "mean"]).round(4).to_string())
        print("\nTop 5 actions by VAEP:")
        cols = ["match_minute", "side", "slot", "action", "success", "vaep", "obv"]
        print(av.sort_values("vaep", ascending=False).head(5)[cols].round(3).to_string(index=False))
    print("\nOK - engine self-test finished.")

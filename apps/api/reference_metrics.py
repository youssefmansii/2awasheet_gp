"""Reference implementations of the two metrics that have public repositories.

DAS  - Dangerous Accessible Space: the `accessible-space` package (J. Bischofberger, MIT licence,
       https://github.com/jonas-bischofberger/accessible-space). Passes are simulated from the ball in 30 directions
       and 15 speeds; the attackers' accessible area is weighted by a small shot-chance model. We call the package
       unchanged and only turn its polar result into the same 50 x 32 grid the rest of the app uses.

OBSO - Off-Ball Scoring Opportunity as computed by C-OBSO (K. Fujii, MIT licence,
       https://github.com/keisuke198619/C-OBSO), after Spearman (2018):
           OBSO grid = pitch control of the attackers  x  transition (a fixed pattern centred on the ball)  x  EPV / max(EPV)
       A player's OBSO is the value of that grid at the cell where he stands; the team value is the sum of the grid.
       `assets/Transition_gauss.csv` is the file from the C-OBSO repository (see assets/C-OBSO_LICENSE.txt).

If `accessible-space` is not installed the engine falls back to its own DAS estimate and says so (das_source).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

try:  # optional dependency
    import accessible_space as _asp
    from accessible_space.interface import get_dangerous_accessible_space as _get_das

    try:
        from importlib.metadata import version as _version

        _ASP_VERSION = _version("accessible-space")
    except Exception:  # pragma: no cover
        _ASP_VERSION = "?"
except Exception:  # pragma: no cover - not installed
    _asp = None
    _get_das = None
    _ASP_VERSION = None


def das_available() -> bool:
    return _get_das is not None


def das_source() -> str:
    return f"accessible-space {_ASP_VERSION}" if das_available() else "own estimate (accessible-space is not installed)"


# ===========================================================================
# OBSO (C-OBSO)
# ===========================================================================
_TRANSITION: Optional[np.ndarray] = None


def _transition() -> np.ndarray:
    """C-OBSO's ball-centred transition pattern, scaled so that its maximum is 1 (as C-OBSO does)."""
    global _TRANSITION
    if _TRANSITION is None:
        t = np.loadtxt(ASSETS_DIR / "Transition_gauss.csv", delimiter=",")  # (2*ny, 2*nx) = (64, 100)
        _TRANSITION = t / np.max(t)
    return _TRANSITION


def cobso_grid(pc_att: np.ndarray, ball_xy: np.ndarray, epv_att: np.ndarray, grid) -> np.ndarray:
    """OBSO of every cell for the team in possession (shape (ny, nx)).

    pc_att   attackers' pitch control, (ny, nx)
    ball_xy  ball position in metres
    epv_att  EPV surface for the attacking direction, (ny, nx)
    """
    T = _transition()
    iy, ix = grid.cell_index(float(ball_xy[0]), float(ball_xy[1]))
    iy, ix = int(iy), int(ix)
    trans = T[grid.ny - 1 - iy: 2 * grid.ny - 1 - iy, grid.nx - 1 - ix: 2 * grid.nx - 1 - ix]  # cropped around the ball cell
    score = epv_att / float(np.max(epv_att))
    return pc_att * trans * score


# ===========================================================================
# DAS (accessible-space)
# ===========================================================================
def _area_elements(sim) -> tuple[np.ndarray, np.ndarray]:
    """dr (T,) and dA (F, PHI, T) exactly as accessible_space.core.integrate_surfaces builds them."""
    r = np.asarray(sim.r_grid, dtype=float)
    lo = np.zeros_like(r)
    lo[1:] = (r[:-1] + r[1:]) / 2
    lo[0] = r[0]
    up = np.zeros_like(r)
    up[:-1] = (r[:-1] + r[1:]) / 2
    up[-1] = r[-1]
    dr = up - lo
    phi = np.asarray(sim.phi_grid, dtype=float)  # (F, PHI)
    plo = np.zeros_like(phi)
    plo[:, 1:] = (phi[:, :-1] + phi[:, 1:]) / 2
    plo[:, 0] = phi[:, 0]
    pup = np.zeros_like(phi)
    pup[:, :-1] = (phi[:, :-1] + phi[:, 1:]) / 2
    pup[:, -1] = phi[:, -1]
    dphi = pup - plo
    dA = dphi[:, :, None] / (2 * np.pi) * (np.pi * up[None, None, :] ** 2) - dphi[:, :, None] / (2 * np.pi) * (np.pi * lo[None, None, :] ** 2)
    return dr, dA


def das_frames(m, items: list[tuple[int, str, int, int]], want_surface: bool = False) -> list[Optional[dict]]:
    """Dangerous accessible space for several frames in ONE call of the library.

    items: (frame_row, attacking_side, attack_direction, drop_slot) - drop_slot >= 0 removes that player from the frame
           (used for the decoy-run index: the runner's own space is taken out, the defenders stay where they are).
    returns one dict per item (or None when the frame cannot be simulated, e.g. no ball):
        das      danger-weighted accessible area, m2 (the package's DAS)
        acc      accessible area without the danger weight, m2 (the package's AS)
        surface  (ny, nx) array whose sum equals `das`  (only when want_surface)
    """
    if not das_available() or not items:
        return [None] * len(items)
    grid = m.grid
    sides_all = m.players["side"].to_numpy()
    recs = []
    keep: dict[int, int] = {}  # unique frame id -> position in items
    for k, (idx, side, adir, drop) in enumerate(items):
        bx, by = float(m.ball_x[idx]), float(m.ball_y[idx])
        if not (np.isfinite(bx) and np.isfinite(by)):
            continue
        slots = m.present_slots(idx)
        sd = sides_all[slots]
        d = np.hypot(m.X[idx, slots] - bx, m.Y[idx, slots] - by)
        carrier = int(slots[int(np.argmin(np.where(sd == side, d, np.inf)))])  # attacker closest to the ball
        if drop >= 0:
            if drop == carrier:  # the ball carrier cannot be taken out as a "receiver"
                continue
            slots = slots[slots != drop]
            sd = sides_all[slots]
        if (sd == side).sum() < 2 or (sd != side).sum() < 2:
            continue
        fid = len(keep)
        keep[fid] = k
        pid_carrier = f"{side}_{carrier}"
        for s, who in zip(slots, sd):
            recs.append((fid, int(m.period[idx]), f"{who}_{int(s)}", who, float(m.X[idx, s]), float(m.Y[idx, s]),
                         float(m.VX[idx, s]), float(m.VY[idx, s]), side, pid_carrier, int(adir)))
        j0, j1 = max(idx - 1, 0), min(idx + 1, m.n_frames - 1)
        dt = max((j1 - j0) / m.fps, 1e-6)
        bvx = (m.ball_x[j1] - m.ball_x[j0]) / dt
        bvy = (m.ball_y[j1] - m.ball_y[j0]) / dt
        recs.append((fid, int(m.period[idx]), "ball", None, bx, by, float(np.nan_to_num(bvx)), float(np.nan_to_num(bvy)),
                     side, pid_carrier, int(adir)))
    out: list[Optional[dict]] = [None] * len(items)
    if not recs:
        return out
    df = pd.DataFrame(recs, columns=["frame_id", "period_id", "player_id", "team_id", "x", "y", "vx", "vy",
                                     "team_in_possession", "player_in_possession", "adir"])
    df = df.dropna(subset=["x", "y"]).copy()
    df[["vx", "vy"]] = df[["vx", "vy"]].fillna(0.0)
    # plain numpy object columns: newer pandas turns text into its own string dtype, which the package's array maths rejects
    for col in ("player_id", "team_id", "team_in_possession", "player_in_possession"):
        df[col] = pd.Series(df[col].to_numpy(dtype=object), index=df.index, dtype=object)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _get_das(df, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id",
                           x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession",
                           player_in_possession_col="player_in_possession", attacking_direction_col="adir",
                           infer_attacking_direction=False, chunk_size=50, use_progress_bar=False)
    except Exception:
        return out
    fr_das = pd.Series(res.das.to_numpy(), index=df["frame_id"].to_numpy())
    fr_as = pd.Series(res.acc_space.to_numpy(), index=df["frame_id"].to_numpy())
    fr_idx = pd.Series(res.frame_index.to_numpy(), index=df["frame_id"].to_numpy())
    fr_das = fr_das[~fr_das.index.duplicated()]
    fr_as = fr_as[~fr_as.index.duplicated()]
    fr_idx = fr_idx[~fr_idx.index.duplicated()]

    surfaces = None
    if want_surface:
        sim = res.simulation_result
        dens = res.dangerous_result.attack_poss_density  # (F, PHI, T), danger already applied
        dr, dA = _area_elements(sim)
        contrib = np.asarray(dens, dtype=float) * dr[None, None, :] * dA  # danger-weighted m2 of every polar cell
        x, y = np.asarray(sim.x_grid, dtype=float), np.asarray(sim.y_grid, dtype=float)
        onp = (np.abs(x) <= grid.length / 2) & (np.abs(y) <= grid.width / 2)  # outside the pitch counts for nothing
        contrib = np.where(onp & np.isfinite(contrib), contrib, 0.0)
        # The package samples space along 30 directions x 15 distances, so only a few hundred points carry value. Dropping each
        # point into the single cell it falls in leaves most cells empty and the picture looks like scattered pixels. Instead every
        # sample is spread over the cells around it (a Gaussian whose width follows the gap between neighbouring samples, i.e.
        # it grows with distance from the ball). Each sample's weights add up to 1, so the surface still sums to the package's DAS.
        x, y = np.broadcast_to(x, contrib.shape), np.broadcast_to(y, contrib.shape)  # the package shares one grid when frames agree
        r = np.asarray(sim.r_grid, dtype=float)
        dphi = 2 * np.pi / max(x.shape[1], 1)
        arc = r[None, None, :] * dphi  # gap between neighbouring directions at each distance
        radial = np.broadcast_to(dr[None, None, :], arc.shape)
        sigma = np.broadcast_to(np.clip(0.6 * np.maximum(arc, radial), 1.2, 9.0), contrib.shape)
        gx, gy = grid.mesh()
        gx, gy = gx.ravel(), gy.ravel()
        surfaces = np.zeros((contrib.shape[0], grid.ny * grid.nx))
        for f in range(contrib.shape[0]):
            m_ = contrib[f] > 0
            if not m_.any():
                continue
            sx, sy, sv, ss = x[f][m_], y[f][m_], contrib[f][m_], sigma[f][m_]
            w = np.exp(-((gx[None, :] - sx[:, None]) ** 2 + (gy[None, :] - sy[:, None]) ** 2) / (2 * ss[:, None] ** 2))
            tot = w.sum(axis=1, keepdims=True)
            ok = tot[:, 0] > 1e-12
            surfaces[f] = (sv[ok, None] * w[ok] / tot[ok]).sum(axis=0)
        surfaces = surfaces.reshape(-1, grid.ny, grid.nx)

    for fid, k in keep.items():
        if fid not in fr_das.index or not np.isfinite(fr_das[fid]):
            continue
        r = {"das": float(fr_das[fid]), "acc": float(fr_as[fid])}
        if surfaces is not None:
            r["surface"] = surfaces[int(fr_idx[fid])]
        out[k] = r
    return out

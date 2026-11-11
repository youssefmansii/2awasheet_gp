"""
ball_physics.py - clean the RAW ball track and find every kick from the ball's own movement
==========================================================================================

1. clean_ball():  raw broadcast ball -> usable ball path
     * height z clipped to [0, 25] m (small negative values are noise)
     * spike removal: a sample far (> SPIKE_M) from the median of its neighbours is dropped
     * short holes (<= FILL_MAX frames) filled by straight lines; longer holes stay missing
     * Savitzky-Golay smoothing of position and its derivative (velocity) on every unbroken run
2. kick_candidates():  a kick / touch = the ball's velocity VECTOR changes suddenly.
     For every frame t we compare the average velocity just before t with just after t
     (DV = |v_after - v_before|, so a speed-up AND a change of direction both count).
     Candidates are the local peaks of DV above DV_MIN.  Moments where the ball disappears and
     reappears with a different velocity (gap kicks) are added too.
No player positions are used here, and nothing from the event file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .tracking_reader import Tracking

PHYS_VERSION = "1.0"


@dataclass
class BallParams:
    z_max: float = 25.0
    spike_m: float = 1.5        # distance from the neighbours' median that marks a spike
    spike_win: int = 7          # frames in the median window
    fill_max: int = 6           # longest hole filled by a straight line (frames, 0.2 s)
    sg_win: int = 7             # Savitzky-Golay window (frames, 0.23 s)
    sg_order: int = 2
    side: int = 5               # frames averaged before / after t for the velocity change
    dv_min: float = 3.0         # m/s, smallest velocity change kept as a candidate
    peak_half: int = 6          # a candidate must be the biggest DV within +-peak_half frames
    gap_max: int = 45           # gap kicks: holes up to 1.5 s are bridged by the gap rule


def _runs(mask: np.ndarray):
    """(start, end) of every run of True values, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def clean_ball(tr: Tracking, p: BallParams = BallParams()) -> dict:
    fps = tr.fps
    pos = tr.ball_raw.astype(np.float64).copy()
    pos[:, 2] = np.clip(pos[:, 2], 0.0, p.z_max)
    ok = np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])
    pos[~ok] = np.nan
    n_raw = int(ok.sum())
    # period boundaries must break every run
    brk = np.zeros(len(pos), bool); brk[1:] = tr.period[1:] != tr.period[:-1]
    # 1) spike removal (per coordinate, against the median of the valid neighbours)
    spikes = np.zeros(len(pos), bool)
    for a, b in _runs(ok):
        if b - a < 3:
            continue
        seg = pos[a:b]
        med = median_filter(seg, size=(p.spike_win, 1), mode="nearest")
        dev = np.linalg.norm(seg[:, :2] - med[:, :2], axis=1)
        spikes[a:b] = dev > p.spike_m
    pos[spikes] = np.nan
    ok = np.isfinite(pos[:, 0])
    # 2) fill short holes linearly (never across a period change)
    filled = np.zeros(len(pos), bool)
    for a, b in _runs(~ok):
        if a == 0 or b >= len(pos) or b - a > p.fill_max or brk[a:b + 1].any():
            continue
        for k in range(3):
            pos[a:b, k] = np.interp(np.arange(a, b), [a - 1, b], [pos[a - 1, k], pos[b, k]])
        filled[a:b] = True
    ok = np.isfinite(pos[:, 0])
    run_id = np.cumsum(brk | np.concatenate([[True], ok[1:] != ok[:-1]]))
    # 3) Savitzky-Golay position + velocity on every run long enough
    sm = np.full_like(pos, np.nan); vel = np.full_like(pos, np.nan)
    for a, b in _runs(ok):
        # split at period changes
        cuts = [a] + [i for i in range(a + 1, b) if brk[i]] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            n = e - s
            if n >= p.sg_win:
                sm[s:e] = savgol_filter(pos[s:e], p.sg_win, p.sg_order, axis=0, mode="interp")
                vel[s:e] = savgol_filter(pos[s:e], p.sg_win, p.sg_order, deriv=1, delta=1 / fps, axis=0, mode="interp")
            elif n >= 2:
                sm[s:e] = pos[s:e]
                vel[s:e] = np.gradient(pos[s:e], 1 / fps, axis=0)
    good = np.isfinite(sm[:, 0])
    return {"pos": sm, "vel": vel, "ok": good, "filled": filled, "spikes": spikes,
            "speed2d": np.hypot(vel[:, 0], vel[:, 1]), "n_raw": n_raw, "run_id": run_id}


def _side_mean(vel: np.ndarray, ok: np.ndarray, t: np.ndarray, side: int, before: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean 2-D velocity over `side` frames before (t-side..t-1) or after (t+1..t+side) every t."""
    T = len(vel)
    v = np.where(ok[:, None], vel[:, :2], 0.0)
    c = np.cumsum(np.vstack([np.zeros((1, 2)), v]), axis=0)
    n = np.cumsum(np.concatenate([[0], ok.astype(np.int64)]))
    if before:
        lo, hi = np.clip(t - side, 0, T), np.clip(t, 0, T)
    else:
        lo, hi = np.clip(t + 1, 0, T), np.clip(t + 1 + side, 0, T)
    cnt = n[hi] - n[lo]
    m = (c[hi] - c[lo]) / np.maximum(cnt, 1)[:, None]
    return m, cnt


def kick_candidates(tr: Tracking, ball: dict, p: BallParams = BallParams()) -> pd.DataFrame:
    vel, ok = ball["vel"], ball["ok"]
    T = len(vel)
    t = np.arange(T)
    vb, nb = _side_mean(vel, ok, t, p.side, True)
    va, na = _side_mean(vel, ok, t, p.side, False)
    enough = (nb >= 3) & (na >= 3)
    dv = np.where(enough, np.linalg.norm(va - vb, axis=1), 0.0)
    # local peaks of DV
    from scipy.ndimage import maximum_filter1d
    peak = (dv >= p.dv_min) & (dv == maximum_filter1d(dv, size=2 * p.peak_half + 1))
    idx = np.flatnonzero(peak)
    # de-duplicate flat peaks (keep the first frame of equal neighbours)
    if idx.size:
        idx = idx[np.concatenate([[True], np.diff(idx) > 1])]
    kind = ["motion"] * len(idx)
    # gap kicks: the ball disappears and comes back moving differently
    gaps = []
    for a, b in _runs(~ok):
        if a < p.side or b + p.side >= T or b - a > p.gap_max or tr.period[a - 1] != tr.period[min(b, T - 1)]:
            continue
        v0 = np.nanmean(vel[a - 3:a, :2], axis=0); v1 = np.nanmean(vel[b:b + 3, :2], axis=0)
        # straight-line check: where would the ball be if nothing touched it?
        pred = ball["pos"][a - 1, :2] + v0 * (b - a + 1) / tr.fps
        miss = np.linalg.norm(ball["pos"][b, :2] - pred)
        jump = np.linalg.norm(v1 - v0)
        if jump >= p.dv_min and miss > 1.0:
            gaps.append(((a + b) // 2, jump, b - a))
    near = set(idx.tolist())
    for g, j, L in gaps:
        if not any(abs(g - k) <= p.peak_half for k in near):
            idx = np.append(idx, g); kind.append("gap")
    o = np.argsort(idx); idx = idx[o]; kind = [kind[i] for i in o]
    df = pd.DataFrame({"idx": idx, "kind": kind})
    df["frame"] = tr.frame[idx]; df["period"] = tr.period[idx]; df["t"] = tr.t[idx]
    df["dv"] = dv[idx]
    gi = df.kind == "gap"
    if gi.any():
        gdv = {g: j for g, j, _ in gaps}
        df.loc[gi, "dv"] = df.loc[gi, "idx"].map(gdv)
    df["speed_before"] = np.linalg.norm(vb[idx], axis=1)
    df["speed_after"] = np.linalg.norm(va[idx], axis=1)
    cosang = np.einsum("ij,ij->i", va[idx], vb[idx]) / np.maximum(df.speed_before * df.speed_after, 1e-6)
    df["turn_deg"] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    df["vx_after"], df["vy_after"] = va[idx, 0], va[idx, 1]
    pos = ball["pos"]
    nearest_ok = np.where(ok)[0]
    j = nearest_ok[np.clip(np.searchsorted(nearest_ok, idx), 0, len(nearest_ok) - 1)]
    df["bx"], df["by"], df["bz"] = pos[j, 0], pos[j, 1], pos[j, 2]
    return df.reset_index(drop=True)

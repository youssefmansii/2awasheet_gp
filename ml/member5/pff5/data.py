"""
Loading compact matches (see preprocess.py) into analysis-ready arrays.

A :class:`Match` holds, for every tracking frame:

* ``pos``  (N, P, 2)  player positions in metres, NaN when the player is not on the pitch
* ``vel``  (N, P, 2)  player velocities in m/s
* ``ball`` (N, 3)     ball x, y, z in metres (short gaps interpolated), ``ball_vel`` (N, 3)
* ``attack`` (N,)     +1 when the HOME team attacks towards +x in that frame, else -1

and one flat ``events`` DataFrame (one row per PFF game-event x possession-event)
with every event already aligned to a tracking frame index.

Pitch coordinates are PFF's: origin at the centre spot, x along the length
(-52.5 .. 52.5), y along the width (-34 .. 34).
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .preprocess import NA, delta_decode

YARD = 0.9144


# --------------------------------------------------------------------------- #
@dataclass
class Match:
    gid: int
    meta: dict
    fps: float
    frame: np.ndarray        # (N,) original PFF frame numbers
    period: np.ndarray       # (N,)
    t: np.ndarray            # (N,) video time, seconds
    pos: np.ndarray          # (N, P, 2)
    vel: np.ndarray          # (N, P, 2)
    ball: np.ndarray         # (N, 3)
    ball_vel: np.ndarray     # (N, 3)
    ball_obs: np.ndarray     # (N,) 1/2 measured, 4 interpolated, 0 missing
    vis: np.ndarray          # (N, P) 0 absent, 1 seen by the camera, 2 position estimated by PFF
    conf: np.ndarray         # (N, P) tracking confidence 0 unknown, 1 low, 2 medium, 3 high
    attack: np.ndarray       # (N,) +1 home attacks +x
    team: np.ndarray         # (P,) 0 home, 1 away
    jersey: np.ndarray       # (P,)
    player_id: np.ndarray    # (P,) PFF player id, -1 if unknown
    is_gk: np.ndarray        # (P,) bool
    pos_group: list = field(default_factory=list)
    events: pd.DataFrame | None = None
    half_len: float = 52.5
    half_wid: float = 34.0

    # ---- helpers ---------------------------------------------------------- #
    @property
    def n_frames(self):
        return len(self.frame)

    @property
    def n_players(self):
        return len(self.team)

    def slot_of(self, player_ids) -> np.ndarray:
        """Vectorised player id -> slot index (-1 when unknown / null)."""
        lut = {int(p): i for i, p in enumerate(self.player_id) if p >= 0}
        out = np.full(len(player_ids), -1, np.int64)
        for k, p in enumerate(player_ids):
            if p is None or (isinstance(p, float) and np.isnan(p)) or pd.isna(p):
                continue
            out[k] = lut.get(int(p), -1)
        return out

    def time_to_index(self, t_sec) -> np.ndarray:
        """Nearest tracking frame index for video time(s) in seconds."""
        t_sec = np.atleast_1d(np.asarray(t_sec, np.float64))
        i = np.searchsorted(self.t, t_sec)
        i = np.clip(i, 1, len(self.t) - 1)
        left = (t_sec - self.t[i - 1]) < (self.t[i] - t_sec)
        return (i - left).astype(np.int64)

    def shift(self, idx, seconds) -> np.ndarray:
        """Frame index ``seconds`` away from ``idx`` without leaving the period."""
        idx = np.asarray(idx, np.int64)
        j = np.clip(idx + int(round(seconds * self.fps)), 0, self.n_frames - 1)
        bad = self.period[j] != self.period[idx]
        if bad.any():
            # clamp to the period boundary
            lo, hi = self._period_bounds()
            p = self.period[idx[bad]]
            j[bad] = np.clip(j[bad], lo[p], hi[p])
        return j

    def _period_bounds(self):
        if not hasattr(self, "_pb"):
            lo = np.zeros(16, np.int64)
            hi = np.zeros(16, np.int64)
            for p in np.unique(self.period):
                w = np.flatnonzero(self.period == p)
                lo[p], hi[p] = w[0], w[-1]
            self._pb = (lo, hi)
        return self._pb


# --------------------------------------------------------------------------- #
def _velocity(x: np.ndarray, period: np.ndarray, t: np.ndarray, k: int = 2) -> np.ndarray:
    """Central difference over +-k frames, never across a period break or a time jump."""
    n = len(x)
    v = np.full_like(x, np.nan, dtype=np.float32)
    if n <= 2 * k:
        return v
    dt = (t[2 * k:] - t[:-2 * k]).astype(np.float32)
    ok = (period[2 * k:] == period[:-2 * k]) & (dt > 0) & (dt < (2 * k + 2) / 25.0)
    d = (x[2 * k:] - x[:-2 * k]) / dt.reshape((-1,) + (1,) * (x.ndim - 1))
    d[~ok] = np.nan
    v[k:-k] = d
    # one-sided fill at the edges of valid runs (keeps NaN only where data is missing)
    for s in (1, -1):
        nb = np.roll(v, s, axis=0)
        fill = np.isnan(v) & ~np.isnan(nb) & ~np.isnan(x)
        fill[0 if s == 1 else -1] = False
        v[fill] = nb[fill]
    return v


def _interp_gaps(ball: np.ndarray, obs: np.ndarray, period: np.ndarray, max_gap: int):
    """Linear interpolation of short runs of missing ball positions."""
    ball = ball.copy()
    obs = obs.copy()
    miss = np.isnan(ball[:, 0])
    if not miss.any() or miss.all():
        return ball, obs
    idx = np.arange(len(ball))
    good = np.flatnonzero(~miss)
    prev = good[np.clip(np.searchsorted(good, idx, side="right") - 1, 0, len(good) - 1)]
    nxt = good[np.clip(np.searchsorted(good, idx, side="left"), 0, len(good) - 1)]
    fill = miss & (prev < idx) & (nxt > idx) & ((nxt - prev) <= max_gap) & (period[prev] == period[nxt])
    w = ((idx - prev) / np.maximum(nxt - prev, 1)).astype(np.float32)[:, None]
    interp = ball[prev] * (1 - w) + ball[nxt] * w
    ball[fill] = interp[fill]
    obs[fill] = 4
    return ball, obs


def clean_raw_ball(raw, pos_raw, pos_sm, period, fps, max_gap_s=1.0):
    """Honest ball signal built from the RAW broadcast ball only.

    PFF's ``ballsSmoothed`` is snapped onto the player PFF's event annotators say
    has the ball, so it leaks the labels we want to predict.  The raw ball has no
    such leak but is noisy and lives in the raw (un-smoothed) coordinate frame.
    Steps: (1) move it into the smoothed-player frame using the local raw->smoothed
    displacement of the nearest players, (2) drop teleporting outliers,
    (3) interpolate short gaps, (4) light Savitzky-Golay smoothing.
    Returns (ball (N,3), obs (N,) 1 = measured, 4 = interpolated, 0 = missing).
    """
    from scipy.signal import savgol_filter

    n = len(raw)
    ball = raw.copy()
    have = np.isfinite(ball[:, 0])
    # (1) frame correction: inverse-distance weighted offset of the 3 nearest players
    off = pos_sm - pos_raw                                   # (N,P,2)
    d = np.linalg.norm(pos_raw - ball[:, None, :2], axis=2)  # (N,P)
    d[~np.isfinite(d) | ~np.isfinite(off[..., 0])] = np.inf
    k = np.argsort(d, axis=1)[:, :3]
    dk = np.take_along_axis(d, k, 1)
    w = 1.0 / (dk + 0.5) ** 2
    w[~np.isfinite(dk) | (dk > 15)] = 0.0
    ok = np.take_along_axis(off, k[..., None], 1)
    ok[~np.isfinite(ok)] = 0.0
    ws = w.sum(1, keepdims=True)
    corr = (ok * w[..., None]).sum(1) / np.maximum(ws, 1e-6)
    corr[ws[:, 0] == 0] = 0.0
    ball[:, :2] += corr.astype(np.float32)
    # (2) outliers: far from the local median
    idx = np.arange(n)
    med = np.full_like(ball[:, :2], np.nan)
    stack = np.stack([ball[np.clip(idx + o, 0, n - 1), :2] for o in (-3, -2, -1, 0, 1, 2, 3)], 0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(stack, axis=0)
    bad = have & (np.linalg.norm(ball[:, :2] - med, axis=1) > 4.0)
    ball[bad] = np.nan
    obs = np.isfinite(ball[:, 0]).astype(np.uint8)
    # (3) gaps
    ball, obs = _interp_gaps(ball, obs, period, int(round(max_gap_s * fps)))
    # (4) smoothing on contiguous runs
    good = np.isfinite(ball[:, 0])
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(np.int8), 0]) != 0)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 7:
            ball[a:b, :2] = savgol_filter(ball[a:b, :2], 7, 2, axis=0)
            ball[a:b, 2] = np.maximum(savgol_filter(ball[a:b, 2], 5, 2), 0.0)
    return ball, obs


def _infer_attack(pos, team, period, is_gk, meta):
    """+1 where the home team attacks +x. Uses the goalkeepers (robust, data driven)."""
    attack = np.ones(len(period), np.int8)
    for p in np.unique(period):
        w = np.flatnonzero(period == p)
        sub = w[:: max(1, len(w) // 2000)]
        hx = None
        if is_gk.any():
            hk = np.flatnonzero((team == 0) & is_gk)
            ak = np.flatnonzero((team == 1) & is_gk)
            if len(hk) and len(ak):
                h = np.nanmean(pos[sub][:, hk, 0])
                a = np.nanmean(pos[sub][:, ak, 0])
                if np.isfinite(h) and np.isfinite(a) and abs(h - a) > 20:
                    hx = h - a
        if hx is None:  # fall back to team centroid at the start of the period
            first = w[: 30]
            hx = np.nanmean(pos[first][:, team == 0, 0]) - np.nanmean(pos[first][:, team == 1, 0])
        if not np.isfinite(hx):
            start_left = meta.get("homeTeamStartLeft", True)
            hx = -1 if (start_left == (p % 2 == 1)) else 1
        attack[w] = 1 if hx < 0 else -1
    return attack


# --------------------------------------------------------------------------- #
def flatten_events(events: list[dict]) -> pd.DataFrame:
    """One row per event with prefixed columns: ge_ / it_ / pe_ / foul_."""
    rows = []
    for e in events:
        r = {k: e.get(k) for k in ("gameId", "gameEventId", "possessionEventId", "startTime", "endTime",
                                   "duration", "eventTime", "sequence")}
        for key, pre in (("gameEvents", "ge_"), ("initialTouch", "it_"), ("possessionEvents", "pe_"), ("fouls", "foul_")):
            for k, v in (e.get(key) or {}).items():
                if k.endswith("Name") or "ideo" in k:
                    continue
                r[pre + k] = v
        sm = e.get("stadiumMetadata") or {}
        r["attack_dir"] = sm.get("teamAttackingDirection")
        rows.append(r)
    df = pd.DataFrame(rows)
    if "foul_onFieldCulpritPlayerID" in df:  # spec spelling vs data spelling
        df = df.rename(columns={"foul_onFieldCulpritPlayerID": "foul_onFieldCulpritPlayerId"})
    return df


def load_match(path_npz: str, with_events: bool = True, ball_source: str = "raw", ball_gap_s: float = 1.0) -> Match:
    """``ball_source``: ``"raw"`` (honest, default) or ``"smoothed"`` (PFF's event-informed ball)."""
    z = delta_decode(np.load(path_npz, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    fps = float(meta.get("fps") or 29.97)
    period = z["period"].astype(np.int64)
    t = z["t_video"]

    def to_m(a):
        out = a.astype(np.float32) / 100.0
        out[a == NA] = np.nan
        return out

    hxy, axy = to_m(z["home_xy"]), to_m(z["away_xy"])
    pos = np.concatenate([hxy, axy], axis=1)
    vflag = np.concatenate([z["home_vis"], z["away_vis"]], axis=1)
    team = np.r_[np.zeros(hxy.shape[1], np.int8), np.ones(axy.shape[1], np.int8)]
    jersey = np.r_[z["home_jerseys"], z["away_jerseys"]].astype(np.int64)
    roster = {(bool(r["home"]), int(r["jerseyNum"])): r for r in meta.get("roster", []) if r.get("jerseyNum") is not None}
    pid = np.full(len(team), -1, np.int64)
    gk = np.zeros(len(team), bool)
    grp = [""] * len(team)
    for i in range(len(team)):
        r = roster.get((team[i] == 0, int(jersey[i])))
        if r:
            pid[i] = int(r["playerId"])
            grp[i] = r.get("positionGroupType") or ""
            gk[i] = grp[i] == "GK"
    if ball_source == "smoothed":
        ball = to_m(z["ball"])
        obs = z["ball_vis"].astype(np.uint8)
        ball, obs = _interp_gaps(ball, obs, period, int(round(ball_gap_s * fps)))
    elif ball_source == "raw":
        pos_raw = np.concatenate([to_m(z["home_raw_xy"]), to_m(z["away_raw_xy"])], axis=1)
        ball, obs = clean_raw_ball(to_m(z["ball_raw"]), pos_raw, pos, period, fps, ball_gap_s)
        del pos_raw
    else:
        raise ValueError("ball_source must be 'raw' or 'smoothed'")
    m = Match(
        gid=int(meta["gameId"]), meta=meta, fps=fps, frame=z["frame"], period=period, t=t,
        pos=pos, vel=_velocity(pos, period, t), ball=ball, ball_vel=_velocity(ball, period, t),
        ball_obs=obs, vis=(vflag & 3).astype(np.uint8), conf=((vflag >> 2) & 3).astype(np.uint8), attack=_infer_attack(pos, team, period, gk, meta), team=team, jersey=jersey,
        player_id=pid, is_gk=gk, pos_group=grp,
        half_len=float(meta.get("pitchLength") or 105.0) / 2, half_wid=float(meta.get("pitchWidth") or 68.0) / 2,
    )
    ev_path = path_npz[:-4] + "_events.json.gz"
    if with_events and os.path.exists(ev_path):
        with gzip.open(ev_path, "rt", encoding="utf-8") as f:
            df = flatten_events(json.load(f))
        df["i_start"] = m.time_to_index(df["startTime"].fillna(df["eventTime"]).to_numpy(float))
        df["i_event"] = m.time_to_index(df["eventTime"].fillna(df["startTime"]).to_numpy(float))
        df["i_end"] = m.time_to_index(df["endTime"].fillna(df["eventTime"]).fillna(df["startTime"]).to_numpy(float))
        # some matches have no tracking for extra time: such events cannot be used
        t_ev = df["eventTime"].fillna(df["startTime"]).to_numpy(float)
        t_st = df["startTime"].fillna(df["eventTime"]).to_numpy(float)
        df["has_trk"] = (np.abs(m.t[df["i_event"].values] - t_ev) <= 0.25) & (np.abs(m.t[df["i_start"].values] - t_st) <= 0.25)
        df["gid"] = m.gid
        m.events = df
    return m


def list_games(compact_dir: str) -> list[int]:
    ids = []
    for p in glob.glob(os.path.join(compact_dir, "*.npz")):
        b = os.path.basename(p)[:-4]
        if b.isdigit() and os.path.exists(os.path.join(compact_dir, b + "_events.json.gz")):
            ids.append(int(b))
    return sorted(ids)


def game_path(compact_dir: str, gid: int) -> str:
    return os.path.join(compact_dir, f"{gid}.npz")

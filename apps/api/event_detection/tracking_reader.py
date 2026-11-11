"""
tracking_reader.py - read ONE match's tracking file WITHOUT any of PFF's hand-made events
==========================================================================================

Why a separate reader?  PFF's tracking lines also carry the hand-made events
(`game_event`, `possession_event`).  If a model could see them it would be copying the answer.
This reader deletes those fields from every line BEFORE looking at anything else, so they
can never reach a feature.  (test_leakage.py proves it.)

What it keeps (per frame):
  * frame number, period, video time, time in period
  * ball: RAW x, y, z + visibility   (PFF's smoothed ball is dropped: it is snapped to players
    at the annotated events, i.e. it contains the answer)
  * every player: RAW x, y + visibility, mapped to the roster (player id, team, position)
Missing frames inside a period become NaN rows, so time steps are always 1 frame.

Run (from apps/api):   python -m event_detection.tracking_reader --match 3830
"""
from __future__ import annotations

import argparse
import bz2
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

READER_VERSION = "1.1"
# Fields that hold PFF's manual event annotations.  They are removed before parsing.
FORBIDDEN_KEYS = ("game_event", "possession_event", "game_event_id", "possession_event_id")
# PFF's post-processed positions.  Step-1 check: the smoothed ball sits EXACTLY on the shooter at
# PFF's shot frames (0.00 m in 9 of 17 shots of 3830) and exactly on a player in 43 % of random
# frames, so it was built with the event annotations.  Only the raw broadcast detections are used.
DERIVED_KEYS = ("ballsSmoothed", "homePlayersSmoothed", "awayPlayersSmoothed")
_VIS = {"VISIBLE": 1, "ESTIMATED": 0}


def data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def cache_dir() -> Path:
    d = data_dir() / "interim" / "m4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _open(path: Path):
    return bz2.open(path, "rb") if str(path).endswith(".bz2") else open(path, "rb")


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def find_files(match_id, ddir: Optional[Path] = None) -> dict:
    ddir = Path(ddir) if ddir else data_dir()
    mid = str(match_id)

    def first(names):
        for n in names:
            if (ddir / n).exists():
                return ddir / n
        return None

    return {
        "tracking": first([f"tracking_{mid}.jsonl.bz2", f"tracking_{mid}.jsonl"]),
        "metadata": first([f"metadata_{mid}.json"]),
        "roster": first([f"roster_{mid}.json", f"rosters_{mid}.json"]),
        "events": first([f"events_{mid}.json", f"event_{mid}.json"]),  # answer key only
    }


@dataclass
class Tracking:
    match_id: str
    fps: float
    frame: np.ndarray        # (T,) int64 frame numbers on a continuous ruler per period
    period: np.ndarray       # (T,) int8
    t: np.ndarray            # (T,) video time in seconds (interpolated in missing rows)
    observed: np.ndarray     # (T,) bool, the line existed in the file
    ball_raw: np.ndarray     # (T,3) float32 raw ball x,y,z (NaN when missing)
    ball_raw_vis: np.ndarray # (T,) int8 1 visible, 0 estimated, -1 missing
    px: np.ndarray           # (T,P) float32 raw player x
    py: np.ndarray           # (T,P) float32 raw player y
    pvis: np.ndarray         # (T,P) int8 1 visible, 0 estimated, -1 absent
    players: pd.DataFrame    # one row per slot: slot, side, jersey, player_id, name, position, team_id, started
    meta: dict               # metadata (teams, pitch, periods, start sides)

    @property
    def n_frames(self) -> int:
        return int(self.frame.size)

    def attack_sign(self, side: str, period: int) -> int:
        """+1 if `side` ('home'/'away') attacks toward +x in this period, else -1."""
        left = self.meta.get("homeTeamStartLeft")
        left = True if left is None else bool(left)
        if period >= 3 and self.meta.get("homeTeamStartLeftExtraTime") is not None:
            left = bool(self.meta["homeTeamStartLeftExtraTime"])
            home_sign = 1 if left else -1
            if period == 4:
                home_sign = -home_sign
        else:
            home_sign = 1 if left else -1
            if period == 2:
                home_sign = -home_sign
        return home_sign if side == "home" else -home_sign

    def gk_slots(self, side: str) -> list[int]:
        p = self.players
        return p.loc[(p.side == side) & (p.position == "GK"), "slot"].tolist()


def _read_meta(path: Optional[Path]) -> dict:
    if not path:
        return {}
    d = json.load(open(path, encoding="utf-8"))
    d = d[0] if isinstance(d, list) else d
    pitch = ((d.get("stadium") or {}).get("pitches") or [{}])[0]
    return {
        "homeTeam": d.get("homeTeam") or {}, "awayTeam": d.get("awayTeam") or {},
        "homeTeamStartLeft": d.get("homeTeamStartLeft"),
        "homeTeamStartLeftExtraTime": d.get("homeTeamStartLeftExtraTime"),
        "fps": _num(d.get("fps")),
        "pitch_length": _num(pitch.get("length")) if pitch else 105.0,
        "pitch_width": _num(pitch.get("width")) if pitch else 68.0,
        "date": d.get("date"),
    }


def _read_roster(path: Optional[Path]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["player_id", "name", "jersey", "team_id", "position", "started"])
    rows = []
    for r in json.load(open(path, encoding="utf-8")):
        p, t = r.get("player") or {}, r.get("team") or {}
        rows.append({"player_id": str(p.get("id")), "name": p.get("nickname"),
                     "jersey": str(r.get("shirtNumber")), "team_id": str(t.get("id")),
                     "position": r.get("positionGroupType"), "started": bool(r.get("started"))})
    return pd.DataFrame(rows)


def parse_lines(lines, max_slots: int = 80) -> dict:
    """Parse an iterable of raw JSON lines (bytes or str). Event fields are deleted first."""
    slot_of: dict[tuple[str, str], int] = {}
    meta_rows, braw, bvis = [], [], []
    xs, ys, vs = [], [], []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = _loads(line)
        except Exception:
            continue
        for k in FORBIDDEN_KEYS + DERIVED_KEYS:   # <- answer key + event-based smoothing thrown away here
            obj.pop(k, None)
        fn = obj.get("frameNum")
        if fn is None:
            continue
        meta_rows.append((fn, obj.get("period") or 0, _num(obj.get("videoTimeMs"))))
        b = obj.get("balls")
        b = (b[0] if b else None) if isinstance(b, list) else b
        if b and b.get("x") is not None:
            braw.append((_num(b.get("x")), _num(b.get("y")), _num(b.get("z"))))
            bvis.append(_VIS.get(b.get("visibility"), 0))
        else:
            braw.append((np.nan, np.nan, np.nan)); bvis.append(-1)
        rx = np.full(max_slots, np.nan, np.float32)
        ry = np.full(max_slots, np.nan, np.float32)
        rv = np.full(max_slots, -1, np.int8)
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for p in obj.get(key) or []:
                k = (side, str(p.get("jerseyNum")))
                s = slot_of.get(k)
                if s is None:
                    if len(slot_of) >= max_slots:
                        continue
                    s = slot_of[k] = len(slot_of)
                if p.get("x") is not None and p.get("y") is not None:
                    rx[s], ry[s] = p["x"], p["y"]
                    rv[s] = _VIS.get(p.get("visibility"), 0)
        xs.append(rx); ys.append(ry); vs.append(rv)
    n = len(slot_of)
    return {
        "meta": np.array(meta_rows, dtype=np.float64).reshape(-1, 3),
        "ball_raw": np.array(braw, np.float32).reshape(-1, 3), "ball_raw_vis": np.array(bvis, np.int8),
        "px": np.array(xs, np.float32)[:, :n] if xs else np.zeros((0, 0), np.float32),
        "py": np.array(ys, np.float32)[:, :n] if ys else np.zeros((0, 0), np.float32),
        "pvis": np.array(vs, np.int8)[:, :n] if vs else np.zeros((0, 0), np.int8),
        "keys": sorted(slot_of, key=slot_of.get),
    }


def _build(match_id: str, files: dict) -> Tracking:
    meta = _read_meta(files["metadata"])
    roster = _read_roster(files["roster"])
    with _open(files["tracking"]) as f:
        raw = parse_lines(f)
    m = raw["meta"]
    keep = m[:, 1] > 0                      # frames inside a match period
    fn = m[keep, 0].astype(np.int64)
    fn, uniq = np.unique(fn, return_index=True)
    sel = np.flatnonzero(keep)[uniq]
    per = m[sel, 1].astype(np.int8)
    vt = m[sel, 2] / 1000.0
    d = np.diff(vt)[np.diff(fn) == 1]
    d = d[np.isfinite(d) & (d > 0)]
    fps = float(1.0 / np.median(d)) if d.size else 29.97
    if np.isfinite(meta.get("fps", np.nan)) and abs(meta["fps"] - fps) < 1.0:
        fps = float(meta["fps"])
    # continuous frame ruler per period
    ruler, rper = [], []
    for p in np.unique(per):
        r = np.arange(fn[per == p].min(), fn[per == p].max() + 1)
        ruler.append(r); rper.append(np.full(r.size, p, np.int8))
    ruler, rper = np.concatenate(ruler), np.concatenate(rper)
    o = np.argsort(ruler, kind="stable"); ruler, rper = ruler[o], rper[o]
    pos = np.searchsorted(ruler, fn)
    T, P = ruler.size, raw["px"].shape[1]

    def place(a, fill):
        out = np.full((T,) + a.shape[1:], fill, dtype=a.dtype)
        out[pos] = a[sel]
        return out

    observed = np.zeros(T, bool); observed[pos] = True
    t = np.interp(ruler, fn, vt)
    # roster mapping: (side, jersey) -> player
    home_id = str((meta.get("homeTeam") or {}).get("id"))
    rows = []
    for s, (side, jersey) in enumerate(raw["keys"]):
        team_rows = roster[(roster.jersey == jersey) &
                           ((roster.team_id == home_id) if side == "home" else (roster.team_id != home_id))]
        r = team_rows.iloc[0] if len(team_rows) else None
        rows.append({"slot": s, "side": side, "jersey": jersey,
                     "player_id": None if r is None else r.player_id,
                     "name": None if r is None else r["name"],
                     "position": None if r is None else r.position,
                     "team_id": None if r is None else r.team_id,
                     "started": None if r is None else bool(r.started)})
    return Tracking(match_id=match_id, fps=fps, frame=ruler, period=rper, t=t, observed=observed,
                    ball_raw=place(raw["ball_raw"], np.nan), ball_raw_vis=place(raw["ball_raw_vis"], -1),
                    px=place(raw["px"], np.nan), py=place(raw["py"], np.nan),
                    pvis=place(raw["pvis"], -1), players=pd.DataFrame(rows), meta=meta)


def load_tracking(match_id, use_cache: bool = True, verbose: bool = False) -> Tracking:
    mid = str(match_id)
    files = find_files(mid)
    if not files["tracking"]:
        raise FileNotFoundError(f"no tracking file for match {mid} in {data_dir()}")
    cpath = cache_dir() / f"tracking_{mid}_v{READER_VERSION}.npz"
    ppath = cache_dir() / f"tracking_{mid}_v{READER_VERSION}.json"
    if use_cache and cpath.exists() and ppath.exists():
        z = np.load(cpath, allow_pickle=False)
        side = json.load(open(ppath, encoding="utf-8"))
        return Tracking(match_id=mid, fps=float(z["fps"]), frame=z["frame"], period=z["period"], t=z["t"],
                        observed=z["observed"], ball_raw=z["ball_raw"], ball_raw_vis=z["ball_raw_vis"],
                        px=z["px"], py=z["py"], pvis=z["pvis"],
                        players=pd.DataFrame(side["players"]), meta=side["meta"])
    t0 = time.time()
    tr = _build(mid, files)
    np.savez(cpath, fps=tr.fps, frame=tr.frame, period=tr.period, t=tr.t, observed=tr.observed,
             ball_raw=tr.ball_raw, ball_raw_vis=tr.ball_raw_vis,
             px=tr.px, py=tr.py, pvis=tr.pvis)
    json.dump({"players": tr.players.to_dict("records"), "meta": tr.meta}, open(ppath, "w", encoding="utf-8"))
    if verbose:
        print(f"[reader] match {mid}: {tr.n_frames} frames, {tr.px.shape[1]} player slots, "
              f"{time.time() - t0:.0f}s")
    return tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    for m in a.match:
        tr = load_tracking(m, use_cache=not a.no_cache, verbose=True)
        unmapped = int(tr.players.player_id.isna().sum())
        print(f"  fps {tr.fps:.2f} | ball raw visible {np.mean(tr.ball_raw_vis >= 0):.1%} | "
              f"unmapped player slots {unmapped}")

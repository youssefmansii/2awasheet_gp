"""
data_loader.py  -  Step 1 of the analytics engine
=================================================

What this file does (in plain words)
------------------------------------
1. Finds the files of ONE match in the local ``data/`` folder
   (tracking .jsonl, events .json, and optionally metadata / roster .json).
2. Reads the huge tracking file line by line (never all at once).
3. Fills small holes in the data with a cubic spline.
4. Smooths speed and acceleration with a 6 Hz Butterworth low-pass filter.
5. Puts everything into one clean ``MatchData`` object made of NumPy arrays.
6. Saves a cache file in ``data/interim/`` so the next start takes ~1 second.

It is written for ANY match: nothing here is hard-coded to match 3816.
Try it from the ``apps/api`` folder:

    python data_loader.py --match 3816
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import pickle
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

try:  # orjson is ~3x faster than json; it is optional.
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

LOADER_VERSION = "1.2"

# ---------------------------------------------------------------------------
# Standard pitch + grid (spec: 105 x 68 m, 50 x 32 cells)
# ---------------------------------------------------------------------------
STD_LENGTH = 105.0
STD_WIDTH = 68.0
GRID_NX = 50
GRID_NY = 32

# 15-minute blocks used for fatigue curves. Stoppage time is folded into the
# last block of each half so the two halves never overlap.
BIN_LABELS = ["0-15'", "15-30'", "30-45'", "45-60'", "60-75'", "75-90'", "90-105' (ET)", "105-120' (ET)"]
_PERIOD_MINUTE_BASE = {1: 0.0, 2: 45.0, 3: 90.0, 4: 105.0}


@dataclass(frozen=True)
class PitchGrid:
    """A 105 x 68 m pitch cut into 50 (length) x 32 (width) cells.

    Arrays shaped (ny, nx) = (32, 50) follow the convention [row = y, column = x],
    the same one used by Laurie Shaw's EPV grid and pitch-control code.
    """

    length: float = STD_LENGTH
    width: float = STD_WIDTH
    nx: int = GRID_NX
    ny: int = GRID_NY

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.width / self.ny

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    def x_centers(self) -> np.ndarray:
        return -self.length / 2 + self.dx * (np.arange(self.nx) + 0.5)

    def y_centers(self) -> np.ndarray:
        return -self.width / 2 + self.dy * (np.arange(self.ny) + 0.5)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(XX, YY) both shaped (ny, nx) with the metre position of every cell centre."""
        return np.meshgrid(self.x_centers(), self.y_centers())

    def points(self) -> np.ndarray:
        """All cell centres as an (ny*nx, 2) array (row-major, y changes slowest)."""
        xx, yy = self.mesh()
        return np.column_stack([xx.ravel(), yy.ravel()])

    def cell_index(self, x, y):
        """Metres -> (iy, ix) cell indices (clipped to the pitch)."""
        ix = np.clip(np.floor((np.asarray(x) + self.length / 2) / self.dx).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((np.asarray(y) + self.width / 2) / self.dy).astype(int), 0, self.ny - 1)
        return iy, ix


@dataclass
class LoaderParams:
    position_source: str = "smoothed"  # "smoothed" (PFF smoothed, same as the web app) or "raw"
    cutoff_hz: float = 6.0  # Butterworth low-pass cut-off (spec)
    filter_order: int = 2  # applied forwards+backwards (filtfilt) -> effective order 4
    max_gap_s: float = 1.0  # holes longer than this are NOT spline-filled
    max_valid_speed: float = 12.5  # m/s, faster = tracking error (same limit as the web app)
    attack_window_frames: int = 30  # frames at the start of a period used to find who attacks which way

    def fingerprint(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------
@dataclass
class MatchData:
    match_id: str
    fps: float
    grid: PitchGrid
    source_pitch: tuple[float, float]  # (length, width) found in the files before standardising

    # --- one entry per frame (T,) ---
    frame_num: np.ndarray  # int32, PFF frameNum (the index used by the API)
    period: np.ndarray  # int8
    time_s: np.ndarray  # float32, seconds since the start of the period
    match_minute: np.ndarray  # float32 (period-aware: 2nd half starts at 45)
    bin_id: np.ndarray  # int8 index into BIN_LABELS
    video_time_ms: np.ndarray  # float64
    observed: np.ndarray  # bool, True if the frame really existed in the file

    # --- (T, P) player arrays in standard metres, NOT flipped ---
    X: np.ndarray
    Y: np.ndarray
    VX: np.ndarray
    VY: np.ndarray
    AX: np.ndarray
    AY: np.ndarray
    SPEED: np.ndarray

    # --- ball (T,) ---
    ball_x: np.ndarray
    ball_y: np.ndarray
    ball_z: np.ndarray

    players: pd.DataFrame  # one row per tracked player (column "slot" = column index in X/Y/...)
    teams: dict  # {"home": {"id":..., "name":...}, "away": {...}}
    attack_dir: dict  # {period: {"home": +1/-1, "away": -1/+1}}  (+1 = attacks towards +x)
    events: pd.DataFrame
    last_event_idx: np.ndarray  # (T,) int32: row of the latest event at/before each frame (-1 = none)
    quality: dict = field(default_factory=dict)

    # ---------------- small helpers ----------------
    @property
    def n_frames(self) -> int:
        return int(self.frame_num.shape[0])

    @property
    def n_players(self) -> int:
        return int(self.X.shape[1])

    def frame_to_idx(self, frame_num: int) -> int:
        """PFF frameNum -> row index in all arrays. Raises KeyError if absent."""
        i = int(np.searchsorted(self.frame_num, frame_num))
        if i >= self.n_frames or int(self.frame_num[i]) != int(frame_num):
            raise KeyError(f"frame {frame_num} not in match {self.match_id}")
        return i

    def nearest_idx(self, frame_num: int) -> int:
        i = int(np.searchsorted(self.frame_num, frame_num))
        i = min(max(i, 0), self.n_frames - 1)
        if i > 0 and abs(int(self.frame_num[i - 1]) - frame_num) < abs(int(self.frame_num[i]) - frame_num):
            i -= 1
        return i

    def time_to_idx(self, video_time_s: float) -> int:
        i = int(np.searchsorted(self.video_time_ms, video_time_s * 1000.0))
        return min(max(i, 0), self.n_frames - 1)

    def attack_direction(self, side: str, idx: int) -> int:
        """+1 if `side` ('home'/'away') attacks towards +x at frame row `idx`."""
        p = int(self.period[idx])
        d = self.attack_dir.get(p) or next(iter(self.attack_dir.values()))
        return int(d[side])

    def slot_for_player_id(self, player_id) -> int:
        row = self.players[self.players["player_id"] == _idstr(player_id)]
        return int(row["slot"].iloc[0]) if len(row) else -1

    def present_slots(self, idx: int) -> np.ndarray:
        return np.flatnonzero(~np.isnan(self.X[idx]) & ~np.isnan(self.Y[idx]))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def default_data_dir() -> Path:
    env = os.environ.get("MATCH_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"  # <repo>/data


_TRACKING_RE = re.compile(r"^(?:tracking_)?(\d+)\.jsonl(?:\.bz2)?$")


def list_available_matches(data_dir: Optional[Path] = None) -> list[str]:
    """Match ids that have a tracking file in data_dir (used later by the API)."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    ids = []
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            m = _TRACKING_RE.match(f.name)
            if m:
                ids.append(m.group(1))
    return sorted(set(ids), key=int)


def _first_existing(data_dir: Path, names: list[str]) -> Optional[Path]:
    for n in names:
        p = data_dir / n
        if p.exists():
            return p
    return None


def find_match_files(match_id, data_dir: Optional[Path] = None) -> dict:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    mid = str(match_id)
    return {
        "tracking": _first_existing(data_dir, [f"tracking_{mid}.jsonl", f"tracking_{mid}.jsonl.bz2", f"{mid}.jsonl", f"{mid}.jsonl.bz2"]),
        "events": _first_existing(data_dir, [f"events_{mid}.json", f"event_{mid}.json", f"{mid}.json"]),
        "metadata": _first_existing(data_dir, [f"metadata_{mid}.json", f"{mid}_metadata.json"]),
        "roster": _first_existing(data_dir, [f"rosters_{mid}.json", f"roster_{mid}.json", f"{mid}_roster.json"]),
        "data_dir": data_dir,
    }


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _idstr(v) -> Optional[str]:
    """Player / team ids come as 14011, 14011.0, '14011' ... -> always '14011' (or None)."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
        return str(int(float(v)))
    except (TypeError, ValueError):
        s = str(v).strip()
        return s or None


def _num(v) -> float:
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def _open_binary(path: Path):
    return bz2.open(path, "rb") if str(path).endswith(".bz2") else open(path, "rb")


def _read_json(path: Path):
    with _open_binary(path) as f:
        return _loads(f.read())


# ---------------------------------------------------------------------------
# Tracking: stream the JSONL file
# ---------------------------------------------------------------------------
_MAX_SLOTS = 64
_CHUNK = 8192


def _parse_tracking(path: Path, source: str) -> dict:
    """Read the .jsonl line by line and keep only what we need, in compact arrays."""
    smoothed = source == "smoothed"
    hkey = "homePlayersSmoothed" if smoothed else "homePlayers"
    akey = "awayPlayersSmoothed" if smoothed else "awayPlayers"
    bkey = "ballsSmoothed" if smoothed else "balls"

    slot_of: dict[tuple[int, str], int] = {}  # (side, jersey) -> column
    actor_map: dict[tuple[int, str], tuple[Optional[str], Optional[str]]] = {}  # (side, jersey) -> (player_id, name)
    team_names: dict[int, tuple[Optional[str], Optional[str]]] = {}

    xs_chunks, ys_chunks = [], []
    cx = np.full((_CHUNK, _MAX_SLOTS), np.nan, dtype=np.float32)
    cy = np.full((_CHUNK, _MAX_SLOTS), np.nan, dtype=np.float32)
    r = 0
    malformed = 0
    cur_meta = np.full((_CHUNK, 7), np.nan, dtype=np.float64)
    meta_chunks = []

    with _open_binary(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _loads(line)
                fn = obj["frameNum"]
            except Exception:
                malformed += 1
                continue

            per = obj.get("period")
            ball = obj.get(bkey)
            if isinstance(ball, list):
                ball = ball[0] if ball else None
            row = cur_meta[r]
            row[0] = fn
            row[1] = per if per is not None else 0
            row[2] = _num(obj.get("videoTimeMs"))
            row[3] = _num(obj.get("periodElapsedTime"))
            if ball:
                row[4] = _num(ball.get("x"))
                row[5] = _num(ball.get("y"))
                row[6] = _num(ball.get("z"))

            for side, key in ((0, hkey), (1, akey)):
                plist = obj.get(key)
                if not plist:
                    continue
                for p in plist:
                    jersey = str(p.get("jerseyNum"))
                    k = (side, jersey)
                    s = slot_of.get(k)
                    if s is None:
                        if len(slot_of) >= _MAX_SLOTS:
                            continue
                        s = len(slot_of)
                        slot_of[k] = s
                    px, py = p.get("x"), p.get("y")
                    if px is not None and py is not None:
                        cx[r, s] = px
                        cy[r, s] = py

            ge = obj.get("game_event")
            if ge:  # the tracking file also tells us which player (id + name) is which shirt
                hs = ge.get("home_team")
                side = 0 if hs in (1, True, "1") else 1
                jersey = ge.get("shirt_number")
                if jersey is not None and ge.get("player_id") is not None:
                    actor_map[(side, str(jersey))] = (_idstr(ge.get("player_id")), ge.get("player_name"))
                if ge.get("team_id") is not None:
                    team_names[side] = (_idstr(ge.get("team_id")), ge.get("team_name"))

            r += 1
            if r == _CHUNK:
                xs_chunks.append(cx)
                ys_chunks.append(cy)
                meta_chunks.append(cur_meta)
                cx = np.full((_CHUNK, _MAX_SLOTS), np.nan, dtype=np.float32)
                cy = np.full((_CHUNK, _MAX_SLOTS), np.nan, dtype=np.float32)
                cur_meta = np.full((_CHUNK, 7), np.nan, dtype=np.float64)
                r = 0

    if r:
        xs_chunks.append(cx[:r])
        ys_chunks.append(cy[:r])
        meta_chunks.append(cur_meta[:r])
    if not meta_chunks:
        raise ValueError(f"No readable frames in {path}")

    n_slots = len(slot_of)
    X = np.concatenate(xs_chunks)[:, :n_slots]
    Y = np.concatenate(ys_chunks)[:, :n_slots]
    meta = np.concatenate(meta_chunks)

    # Order the slots: home first then away, then by shirt number.
    order = sorted(slot_of.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 999))
    perm = np.array([slot for _, slot in order], dtype=int)
    X, Y = X[:, perm], Y[:, perm]
    keys = [k for k, _ in order]

    return {
        "X": X, "Y": Y, "meta": meta, "keys": keys,
        "actor_map": actor_map, "team_names": team_names, "malformed": malformed,
    }


# ---------------------------------------------------------------------------
# Gap filling (cubic spline) and filtering (Butterworth)
# ---------------------------------------------------------------------------
def _fill_short_gaps(series: np.ndarray, max_gap: int) -> tuple[np.ndarray, int]:
    """Fill NaN holes that sit BETWEEN valid samples and are <= max_gap long.
    Returns (filled copy, number of filled samples)."""
    nan = np.isnan(series)
    n = series.size
    if not nan.any() or nan.all():
        return series, 0
    valid_idx = np.flatnonzero(~nan)
    if valid_idx.size < 4:
        return series, 0
    edges = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    ok = (starts > 0) & (ends < n) & ((ends - starts) <= max_gap)
    if not ok.any():
        return series, 0
    targets = np.concatenate([np.arange(s, e) for s, e in zip(starts[ok], ends[ok])])
    cs = CubicSpline(valid_idx, series[valid_idx].astype(np.float64))
    out = series.copy()
    out[targets] = cs(targets)
    return out, int(targets.size)


def _smooth_motion(x: np.ndarray, y: np.ndarray, fps: float, b: np.ndarray, a: np.ndarray, padlen: int):
    """Position (T,) -> filtered velocity and acceleration vectors, run by run."""
    T = x.size
    vx = np.full(T, np.nan)
    vy = np.full(T, np.nan)
    ax = np.full(T, np.nan)
    ay = np.full(T, np.nan)
    valid = ~(np.isnan(x) | np.isnan(y))
    if not valid.any():
        return vx, vy, ax, ay
    dt = 1.0 / fps
    edges = np.diff(np.concatenate(([0], valid.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        n = e - s
        if n < 3:
            continue
        xs = x[s:e].astype(np.float64)
        ys = y[s:e].astype(np.float64)
        vxs, vys = np.gradient(xs, dt), np.gradient(ys, dt)
        if n > padlen:
            vxs, vys = filtfilt(b, a, vxs), filtfilt(b, a, vys)
        axs, ays = np.gradient(vxs, dt), np.gradient(vys, dt)
        if n > padlen:
            axs, ays = filtfilt(b, a, axs), filtfilt(b, a, ays)
        vx[s:e], vy[s:e], ax[s:e], ay[s:e] = vxs, vys, axs, ays
    return vx, vy, ax, ay


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
_EVENT_COLUMNS = {
    "gameEventId": "event_id",
    "possessionEventId": "possession_event_id",
    "eventTime": "event_time_s",
    "startTime": "start_time_s",
    "endTime": "end_time_s",
    "sequence": "sequence",
    "gameEvents.gameEventType": "game_event_type",
    "gameEvents.period": "period",
    "gameEvents.startGameClock": "game_clock_s",
    "gameEvents.teamId": "team_id",
    "gameEvents.teamName": "team_name",
    "gameEvents.homeTeam": "home_team",
    "gameEvents.playerId": "player_id",
    "gameEvents.playerName": "player_name",
    "gameEvents.setpieceType": "setpiece_type",
    "possessionEvents.possessionEventType": "poss_type",
    "possessionEvents.passerPlayerId": "passer_id",
    "possessionEvents.passOutcomeType": "pass_outcome",
    "possessionEvents.passType": "pass_type",
    "possessionEvents.targetPlayerId": "target_id",
    "possessionEvents.receiverPlayerId": "receiver_id",
    "possessionEvents.crosserPlayerId": "crosser_id",
    "possessionEvents.crossOutcomeType": "cross_outcome",
    "possessionEvents.shooterPlayerId": "shooter_id",
    "possessionEvents.shotOutcomeType": "shot_outcome",
    "possessionEvents.shotType": "shot_type",
    "possessionEvents.carrierPlayerId": "carrier_id",
    "possessionEvents.dribblerPlayerId": "dribbler_id",
    "possessionEvents.clearerPlayerId": "clearer_id",
    "possessionEvents.challengeType": "challenge_type",
    "initialTouch.initialPressureType": "pressure_type",
    "initialTouch.initialPressurePlayerId": "pressure_player_id",
}
_ID_COLUMNS = ["player_id", "passer_id", "target_id", "receiver_id", "crosser_id", "shooter_id",
               "carrier_id", "dribbler_id", "clearer_id", "pressure_player_id", "team_id"]


def _load_events(path: Path) -> pd.DataFrame:
    data = _read_json(path)
    if isinstance(data, dict):
        for k in ("events", "data", "items"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            data = list(data.values())
    df = pd.json_normalize(data, sep=".")
    df = df.rename(columns=_EVENT_COLUMNS)
    for col in _EVENT_COLUMNS.values():  # make sure every column exists
        if col not in df.columns:
            df[col] = None
    for col in _ID_COLUMNS:
        df[col] = df[col].map(_idstr)
    df["event_time_s"] = pd.to_numeric(df["event_time_s"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df = df.sort_values(["event_time_s", "sequence"], kind="stable").reset_index(drop=True)
    return df


def _load_roster(path: Path) -> pd.DataFrame:
    data = _read_json(path)
    rows = []
    for r in data if isinstance(data, list) else data.get("rosters", []):
        p, t = r.get("player") or {}, r.get("team") or {}
        rows.append({
            "player_id": _idstr(p.get("id")),
            "name": p.get("nickname") or p.get("name"),
            "jersey": None if r.get("shirtNumber") is None else str(r.get("shirtNumber")),
            "team_id": _idstr(t.get("id")),
            "team_name": t.get("name"),
            "position": r.get("positionGroupType"),
            "started": r.get("started"),
        })
    return pd.DataFrame(rows)


def _read_metadata(path: Optional[Path]) -> dict:
    out = {}
    if not path:
        return out
    try:
        m = _read_json(path)
        if isinstance(m, list):
            m = m[0] if m else {}
        pitch = None
        try:
            pitch = m["stadium"]["pitches"][0]
        except Exception:
            pass
        if pitch and pitch.get("length") and pitch.get("width"):
            out["pitch"] = (float(pitch["length"]), float(pitch["width"]))
        elif m.get("pitchLength") and m.get("pitchWidth"):
            out["pitch"] = (float(m["pitchLength"]), float(m["pitchWidth"]))
        if m.get("fps"):
            out["fps"] = float(m["fps"])
        for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
            t = m.get(key)
            if isinstance(t, dict):
                out[side] = {"id": _idstr(t.get("id")), "name": t.get("name")}
    except Exception as exc:  # metadata is optional - never fail because of it
        warnings.warn(f"Could not read metadata file {path}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Cache reading
# ---------------------------------------------------------------------------
class _CacheUnpickler(pickle.Unpickler):
    """Opens cache files no matter which script created them.

    When ``python data_loader.py`` is run directly, Python records our classes as ``__main__.MatchData``.
    Any other script (player_analytics.py, main.py ...) cannot find that name, so we redirect the
    lookup of our own classes to THIS module, whatever module name the file says."""

    _OWN = ("MatchData", "PitchGrid", "LoaderParams")

    def find_class(self, module, name):
        if name in self._OWN:
            return globals()[name]
        return super().find_class(module, name)


def _read_cache(path: Path):
    """Return the cached MatchData, or None if the file is unreadable (it is then rebuilt)."""
    try:
        with open(path, "rb") as f:
            obj = _CacheUnpickler(f).load()
        return obj if isinstance(obj, MatchData) else None
    except Exception as exc:
        warnings.warn(f"Ignoring unreadable cache {path.name} ({exc!r}); rebuilding.")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _cache_key(files: dict, params: LoaderParams) -> str:
    h = hashlib.sha1()
    h.update(LOADER_VERSION.encode())
    h.update(params.fingerprint().encode())
    for k in ("tracking", "events", "metadata", "roster"):
        p = files.get(k)
        if p:
            st = Path(p).stat()
            h.update(f"{k}:{Path(p).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def load_match(
    match_id,
    data_dir: Optional[Path] = None,
    params: Optional[LoaderParams] = None,
    use_cache: bool = True,
    tracking_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    roster_path: Optional[Path] = None,
    verbose: bool = False,
) -> MatchData:
    """Load one match from disk (or from the cache) as a ready-to-use MatchData."""
    params = params or LoaderParams()
    mid = str(match_id)
    files = find_match_files(mid, data_dir)
    if tracking_path:
        files["tracking"] = Path(tracking_path)
    if events_path:
        files["events"] = Path(events_path)
    if metadata_path:
        files["metadata"] = Path(metadata_path)
    if roster_path:
        files["roster"] = Path(roster_path)
    if not files["tracking"] or not Path(files["tracking"]).exists():
        raise FileNotFoundError(
            f"No tracking file for match {mid} in {files['data_dir']}. "
            f"Expected e.g. tracking_{mid}.jsonl (or .jsonl.bz2)."
        )

    cache_path = Path(files["data_dir"]) / "interim" / f"match_{mid}_{_cache_key(files, params)}.pkl"
    if use_cache and cache_path.exists():
        md = _read_cache(cache_path)
        if md is not None:
            if verbose:
                print(f"[loader] loaded from cache {cache_path.name}")
            return md

    t0 = time.time()
    log = (lambda m: print(f"[loader] {m}")) if verbose else (lambda m: None)
    meta = _read_metadata(files["metadata"])

    # 1) stream the tracking file --------------------------------------------------
    log(f"reading {Path(files['tracking']).name} ...")
    parsed = _parse_tracking(Path(files["tracking"]), params.position_source)
    m = parsed["meta"]
    fn_raw = m[:, 0].astype(np.int64)
    per_raw = np.nan_to_num(m[:, 1]).astype(np.int64)
    keep = per_raw > 0  # drop frames that are outside a match period
    quality = {"frames_in_file": int(len(fn_raw)), "malformed_lines": int(parsed["malformed"]),
               "frames_outside_periods_dropped": int((~keep).sum())}
    X0, Y0, m = parsed["X"][keep], parsed["Y"][keep], m[keep]
    fn_raw, per_raw = fn_raw[keep], per_raw[keep]

    # sort + de-duplicate by frameNum
    quality["duplicate_frame_lines_dropped"] = int(len(fn_raw) - len(np.unique(fn_raw)))
    fn_raw, uniq = np.unique(fn_raw, return_index=True)
    X0, Y0, m, per_raw = X0[uniq], Y0[uniq], m[uniq], per_raw[uniq]
    vt_raw, el_raw = m[:, 2], m[:, 3]

    # frame rate: from the data (29.97), unless the metadata says otherwise
    d = np.diff(vt_raw)[np.diff(fn_raw) == 1]
    d = d[np.isfinite(d) & (d > 0)]
    fps = float(1000.0 / np.median(d)) if d.size else 29.97
    if not (15 <= fps <= 60):
        fps = 29.97
    if meta.get("fps") and abs(meta["fps"] - fps) < 1.0:
        fps = meta["fps"]
    log(f"fps = {fps:.3f}")

    # 2) put every period on a continuous frame ruler (missing frames become NaN rows)
    full_fn, full_per = [], []
    for p in np.unique(per_raw):
        sel = per_raw == p
        rng = np.arange(fn_raw[sel].min(), fn_raw[sel].max() + 1)
        full_fn.append(rng)
        full_per.append(np.full(rng.size, p, dtype=np.int8))
    full_fn = np.concatenate(full_fn)
    full_per = np.concatenate(full_per)
    order = np.argsort(full_fn, kind="stable")
    full_fn, full_per = full_fn[order], full_per[order]
    T, P = full_fn.size, X0.shape[1]
    pos = np.searchsorted(full_fn, fn_raw)
    observed = np.zeros(T, dtype=bool)
    observed[pos] = True
    quality["frames_total"] = int(T)
    quality["frames_missing_rebuilt"] = int(T - observed.sum())

    X = np.full((T, P), np.nan, dtype=np.float32)
    Y = np.full((T, P), np.nan, dtype=np.float32)
    X[pos], Y[pos] = X0, Y0
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    ball[pos] = m[:, 4:7]
    video_time_ms = np.interp(full_fn, fn_raw, vt_raw)
    if np.isfinite(el_raw).any():
        elapsed = np.interp(full_fn, fn_raw[np.isfinite(el_raw)], el_raw[np.isfinite(el_raw)])
    else:
        elapsed = np.zeros(T)
        for p in np.unique(full_per):
            sel = full_per == p
            elapsed[sel] = (full_fn[sel] - full_fn[sel][0]) / fps

    # 3) standardise the pitch to 105 x 68 m
    src_len, src_wid = meta.get("pitch", (STD_LENGTH, STD_WIDTH))
    sx, sy = STD_LENGTH / src_len, STD_WIDTH / src_wid
    if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
        X *= np.float32(sx)
        Y *= np.float32(sy)
        ball[:, 0] *= np.float32(sx)
        ball[:, 1] *= np.float32(sy)

    # 4) cubic-spline filling of short holes, per period
    max_gap = max(1, int(round(params.max_gap_s * fps)))
    bounds = np.flatnonzero(np.diff(full_per.astype(int))) + 1
    slices = list(zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [T]))))
    filled = 0
    for a_, b_ in slices:
        for j in range(P):
            for arr in (X, Y):
                out, n = _fill_short_gaps(arr[a_:b_, j], max_gap)
                arr[a_:b_, j] = out
                filled += n
        for c in (0, 1):
            out, _ = _fill_short_gaps(ball[a_:b_, c], max_gap)
            ball[a_:b_, c] = out
    quality["samples_spline_filled"] = int(filled // 2)  # x and y counted together

    # 5) Butterworth low-pass -> velocity + acceleration
    wn = params.cutoff_hz / (fps / 2.0)
    b, a = butter(params.filter_order, min(wn, 0.99), btype="low")
    padlen = 3 * max(len(a), len(b))
    VX = np.full((T, P), np.nan, dtype=np.float32)
    VY = np.full_like(VX, np.nan)
    AX = np.full_like(VX, np.nan)
    AY = np.full_like(VX, np.nan)
    for a_, b_ in slices:
        for j in range(P):
            vx, vy, ax, ay = _smooth_motion(X[a_:b_, j], Y[a_:b_, j], fps, b, a, padlen)
            VX[a_:b_, j], VY[a_:b_, j], AX[a_:b_, j], AY[a_:b_, j] = vx, vy, ax, ay
    SPEED = np.hypot(VX, VY)
    bad = SPEED > params.max_valid_speed
    quality["samples_impossible_speed"] = int(bad.sum())
    for arr in (VX, VY, AX, AY, SPEED):
        arr[bad] = np.nan
    log("filtering done")

    # 6) time axes
    minute_base = np.array([_PERIOD_MINUTE_BASE.get(int(p), 0.0) for p in full_per])
    match_minute = (minute_base + elapsed / 60.0).astype(np.float32)
    within = np.floor(elapsed / 60.0 / 15.0).astype(int)
    bin_id = np.empty(T, dtype=np.int8)
    for p in np.unique(full_per):
        sel = full_per == p
        if p == 1:
            bin_id[sel] = np.minimum(within[sel], 2)
        elif p == 2:
            bin_id[sel] = 3 + np.minimum(within[sel], 2)
        elif p == 3:
            bin_id[sel] = 6
        else:
            bin_id[sel] = 7

    # 7) players table (columns of X / Y / ...)
    keys = parsed["keys"]
    side_of_slot = np.array([k[0] for k in keys])
    events = _load_events(Path(files["events"])) if files["events"] else pd.DataFrame(columns=list(_EVENT_COLUMNS.values()))

    teams = {"home": {"id": None, "name": None}, "away": {"id": None, "name": None}}
    for sd, name in ((0, "home"), (1, "away")):
        if sd in parsed["team_names"]:
            teams[name] = {"id": parsed["team_names"][sd][0], "name": parsed["team_names"][sd][1]}
    if len(events):
        for name, flag in (("home", True), ("away", False)):
            ev = events[events["home_team"] == flag]
            ev = ev[ev["team_id"].notna()]
            if len(ev):
                teams[name] = {"id": ev["team_id"].iloc[0], "name": ev["team_name"].iloc[0]}
    for name in ("home", "away"):
        if meta.get(name):
            teams[name] = {**teams[name], **{k: v for k, v in meta[name].items() if v}}
        if not teams[name]["name"]:
            teams[name]["name"] = name.title()

    roster = _load_roster(Path(files["roster"])) if files["roster"] else None
    rows = []
    for slot, (sd, jersey) in enumerate(keys):
        side = "home" if sd == 0 else "away"
        pid, name = parsed["actor_map"].get((sd, jersey), (None, None))
        position, started = None, None
        if roster is not None and len(roster):
            rr = roster[(roster["team_id"] == teams[side]["id"]) & (roster["jersey"] == jersey)]
            if len(rr):
                pid = pid or rr["player_id"].iloc[0]
                name = name or rr["name"].iloc[0]
                position, started = rr["position"].iloc[0], rr["started"].iloc[0]
        present = ~np.isnan(X[:, slot])
        idx_present = np.flatnonzero(present)
        rows.append({
            "slot": slot, "side": side, "team_name": teams[side]["name"], "jersey": jersey,
            "player_id": pid, "name": name or f"{side.title()} #{jersey}",
            "position": position, "started": started,
            "first_idx": int(idx_present[0]) if idx_present.size else -1,
            "last_idx": int(idx_present[-1]) if idx_present.size else -1,
            "frames_present": int(present.sum()),
        })
    players = pd.DataFrame(rows)
    # names for ids we know from events but whose shirt we could not map
    if len(events):
        id_to_name = dict(zip(events["player_id"], events["player_name"]))
        missing = players["name"].str.contains(r"#\d+$", regex=True) & players["player_id"].notna()
        players.loc[missing, "name"] = players.loc[missing, "player_id"].map(id_to_name).fillna(players.loc[missing, "name"])

    # 8) who attacks which way in each period
    attack_dir = {}
    for p in np.unique(full_per):
        idx = np.flatnonzero(full_per == p)[: params.attack_window_frames]
        hx = X[idx][:, side_of_slot == 0]
        ax_ = X[idx][:, side_of_slot == 1]
        if np.isfinite(hx).any() and np.isfinite(ax_).any():
            home_dir = 1 if np.nanmean(hx) < np.nanmean(ax_) else -1
        else:
            home_dir = 1 if p in (1, 3) else -1
        attack_dir[int(p)] = {"home": home_dir, "away": -home_dir}

    # 9) events -> frames, positions, slots
    quality["events_total"] = int(len(events))
    last_event_idx = np.full(T, -1, dtype=np.int32)
    if len(events):
        ev_ms = events["event_time_s"].to_numpy(dtype=float) * 1000.0
        fidx = np.searchsorted(video_time_ms, np.nan_to_num(ev_ms, nan=-1.0))
        fidx = np.clip(fidx, 0, T - 1)
        near = np.where((fidx > 0) & (np.abs(video_time_ms[np.maximum(fidx - 1, 0)] - ev_ms) < np.abs(video_time_ms[fidx] - ev_ms)), fidx - 1, fidx)
        bad_ev = ~np.isfinite(ev_ms) | (np.abs(video_time_ms[near] - ev_ms) > 2000.0)
        events["frame_idx"] = np.where(bad_ev, -1, near).astype(int)
        quality["events_without_frame"] = int(bad_ev.sum())
        ok = events["frame_idx"] >= 0
        events["frame_num"] = np.where(ok, full_fn[np.maximum(events["frame_idx"], 0)], -1)
        fi = events["frame_idx"].to_numpy()
        events["ball_x"] = np.where(ok, ball[np.maximum(fi, 0), 0], np.nan)
        events["ball_y"] = np.where(ok, ball[np.maximum(fi, 0), 1], np.nan)
        events["is_goal"] = False
        # next event frame = when the ball is received / the action ends
        nxt = pd.Series(fi).shift(-1).fillna(-1).astype(int).to_numpy()
        events["next_frame_idx"] = np.where(ok & (nxt >= 0), nxt, np.where(ok, fi, -1))
        nf = events["next_frame_idx"].to_numpy()
        events["end_ball_x"] = np.where(nf >= 0, ball[np.maximum(nf, 0), 0], np.nan)
        events["end_ball_y"] = np.where(nf >= 0, ball[np.maximum(nf, 0), 1], np.nan)
        # side and slots
        id_team_side = {}
        for sd in ("home", "away"):
            if teams[sd]["id"]:
                id_team_side[teams[sd]["id"]] = sd
        events["side"] = [
            ("home" if h in (True,) else "away" if h in (False,) else id_team_side.get(t))
            for h, t in zip(events["home_team"], events["team_id"])
        ]
        pid_to_slot = {pid: int(s) for pid, s in zip(players["player_id"], players["slot"]) if pid}
        for col, out in (("player_id", "actor_slot"), ("passer_id", "passer_slot"), ("receiver_id", "receiver_slot"),
                         ("target_id", "target_slot"), ("shooter_id", "shooter_slot")):
            events[out] = events[col].map(lambda v: pid_to_slot.get(v, -1) if v else -1).astype(int)
        quality["event_players_unmapped"] = int(((events["player_id"].notna()) & (events["actor_slot"] < 0)).sum())
        # latest event at or before each frame
        marker = np.full(T, -1, dtype=np.int32)
        for i, f_ in enumerate(fi):
            if f_ >= 0:
                marker[f_] = max(marker[f_], i)
        last_event_idx = np.maximum.accumulate(marker)
        # "G" = ball in the net, but goals later disallowed (offside/VAR) are also marked G.
        # A goal is REAL only if the next restart is a kick-off (setpiece 'K'); anything else = disallowed.
        sp = events["setpiece_type"].to_numpy(dtype=object)
        is_goal = np.zeros(len(events), dtype=bool)
        shots_g = np.flatnonzero(((events["poss_type"] == "SH") & (events["shot_outcome"] == "G")).to_numpy())
        for i in shots_g:
            real = True  # nothing follows (end of data) -> assume real
            for j in range(i + 1, min(i + 40, len(events))):
                v = sp[j]
                if pd.isna(v) or v in ("O", ""):
                    continue
                real = (v == "K")
                break
            is_goal[i] = real
        events["is_goal"] = is_goal
        g_all = events.iloc[shots_g]
        quality["goals_in_net_G"] = g_all["side"].value_counts().to_dict()
        quality["goals_real_(kickoff_follows)"] = events[events["is_goal"]]["side"].value_counts().to_dict()
        quality["goals_suspected_disallowed"] = g_all[~g_all["is_goal"].astype(bool)]["side"].value_counts().to_dict() if len(g_all) else {}
    quality["fps"] = round(fps, 3)
    quality["load_seconds"] = round(time.time() - t0, 1)

    md = MatchData(
        match_id=mid, fps=fps, grid=PitchGrid(), source_pitch=(src_len, src_wid),
        frame_num=full_fn.astype(np.int32), period=full_per, time_s=elapsed.astype(np.float32),
        match_minute=match_minute, bin_id=bin_id, video_time_ms=video_time_ms, observed=observed,
        X=X, Y=Y, VX=VX, VY=VY, AX=AX, AY=AY, SPEED=SPEED,
        ball_x=ball[:, 0], ball_y=ball[:, 1], ball_z=ball[:, 2],
        players=players, teams=teams, attack_dir=attack_dir, events=events,
        last_event_idx=last_event_idx, quality=quality,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(md, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"cache written: {cache_path.name}")
    return md


# ---------------------------------------------------------------------------
# Self-test:  python data_loader.py --match 3816
# ---------------------------------------------------------------------------
def _summary(md: MatchData) -> None:
    print("=" * 70)
    print(f"Match {md.match_id}: {md.teams['home']['name']} (home) vs {md.teams['away']['name']} (away)")
    print(f"fps={md.fps:.3f}  frames={md.n_frames:,}  players tracked={md.n_players}  pitch(source)={md.source_pitch}")
    print(f"periods found: {sorted(set(md.period.tolist()))}   attack directions: {md.attack_dir}")
    print("-" * 70)
    print(md.players[["slot", "side", "jersey", "player_id", "name", "frames_present"]].to_string(index=False))
    print("-" * 70)
    print("data quality:")
    for k, v in md.quality.items():
        print(f"  {k}: {v}")
    if len(md.events):
        print("-" * 70)
        print("event types:", md.events["poss_type"].fillna(md.events["game_event_type"]).value_counts().head(12).to_dict())
    i = md.n_frames // 3
    pr = md.present_slots(i)
    print("-" * 70)
    print(f"sample frame row {i} (frameNum {md.frame_num[i]}, period {md.period[i]}, t={md.time_s[i]:.1f}s): {len(pr)} players visible")
    top = pr[np.argsort(-np.nan_to_num(md.SPEED[i, pr]))[:3]]
    for s in top:
        r = md.players.iloc[s]
        print(f"  {r['name']:<28} x={md.X[i, s]:7.2f} y={md.Y[i, s]:7.2f} speed={md.SPEED[i, s]:5.2f} m/s")
    print(f"  ball x={md.ball_x[i]:.2f} y={md.ball_y[i]:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load one match and print a summary.")
    ap.add_argument("--match", required=True, help="match id, e.g. 3816")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--source", default="smoothed", choices=["smoothed", "raw"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    t = time.time()
    data = load_match(args.match, data_dir=args.data_dir, params=LoaderParams(position_source=args.source),
                      use_cache=not args.no_cache, verbose=True)
    _summary(data)
    print(f"total time: {time.time() - t:.1f}s")

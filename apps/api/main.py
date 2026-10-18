"""
main.py  -  Step 4 of the analytics engine: the FastAPI web service
===================================================================

Run it (from the apps/api folder, with the virtual environment active):

    uvicorn main:app --port 8000

Then open http://localhost:8000/docs to try every endpoint in the browser.

Every endpoint accepts an optional ``?match_id=`` so ANY match in ``data/`` can be analysed.
Nothing is loaded or computed when the server starts: a match is read the first time it is asked for, and the
heavy player analytics only when someone opens the player tab. New matches can be sent with
``POST /api/matches/upload`` (tracking + events files), which saves them into ``data/``.
(Calls without ``?match_id=`` still fall back to env DEFAULT_MATCH_ID / 3816 / the first match, for convenience in /docs.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import orjson
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from data_loader import BIN_LABELS, default_data_dir, find_match_files, list_available_matches, load_match
from metrics_engine import MetricsEngine
from player_analytics import PlayerAnalytics, _native
from player_live import LivePlayers
from reference_metrics import das_source


# ---------------------------------------------------------------------------
# Fast JSON (handles numpy and NaN -> null)
# ---------------------------------------------------------------------------
class FastJSON(Response):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(_native(content) if not isinstance(content, (list, dict)) else content,
                            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
                            default=lambda o: None)


# ---------------------------------------------------------------------------
# Match registry: loads a match once, keeps engine + analytics in memory
# ---------------------------------------------------------------------------
class Bundle:
    def __init__(self, match_id: str):
        self.match = load_match(match_id)
        self.engine = MetricsEngine(self.match)
        self.analytics = PlayerAnalytics(self.engine)
        self.live = LivePlayers(self.engine)  # cheap to create: the tables are built the first time they are asked for
        self.warm_state = "idle"  # idle | running | done | error
        self.warm_error: Optional[str] = None
        self.lock = threading.Lock()


_bundles: dict[str, Bundle] = {}
_build_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
MAX_LOADED_MATCHES = int(os.environ.get("MAX_LOADED_MATCHES", "2"))  # each match keeps ~100s of MB in memory
_ID_RE = re.compile(r"^\d{1,12}$")


def clean_match_id(match_id) -> str:
    """Only digits are accepted: the id becomes part of a file name, so nothing else may get through."""
    mid = str(match_id).strip()
    if not _ID_RE.match(mid):
        raise HTTPException(400, "match_id must be a number (digits only)")
    return mid


def default_match_id() -> str:
    env = os.environ.get("DEFAULT_MATCH_ID")
    if env:
        return clean_match_id(env)
    ids = list_available_matches()
    if "3816" in ids:
        return "3816"
    if ids:
        return ids[0]
    raise HTTPException(404, f"No tracking files found in {default_data_dir()}")


def get_bundle(match_id: Optional[str]) -> Bundle:
    """Return the loaded match, reading it from disk the first time (only that match is blocked while it loads)."""
    mid = clean_match_id(match_id) if match_id else default_match_id()
    with _registry_lock:
        b = _bundles.get(mid)
        build_lock = _build_locks.setdefault(mid, threading.Lock())
    if b is not None:
        return b
    with build_lock:
        with _registry_lock:
            b = _bundles.get(mid)
        if b is not None:
            return b
        try:
            b = Bundle(mid)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        with _registry_lock:
            _bundles[mid] = b
            while len(_bundles) > MAX_LOADED_MATCHES:  # forget the oldest idle match, never one that is computing
                old = next((k for k, v in _bundles.items() if k != mid and v.warm_state != "running"), None)
                if old is None:
                    break
                _bundles.pop(old)
        return b


def start_warmup(b: Bundle) -> None:
    """Compute the heavy player analytics in a background thread (once per match)."""
    with b.lock:
        if b.warm_state in ("running", "done"):
            return
        b.warm_state = "running"

    def work():
        try:
            b.analytics.warm(progress=lambda s: print(f"[warmup {b.match.match_id}] {s}"))
            b.warm_state = "done"
        except Exception as exc:  # pragma: no cover
            b.warm_state, b.warm_error = "error", repr(exc)
            print("[warmup] failed:", exc)

    threading.Thread(target=work, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing heavy at start-up. (Set PRELOAD_DEFAULT_MATCH=1 only if you want the old behaviour for demos.)
    if os.environ.get("PRELOAD_DEFAULT_MATCH") == "1":
        def boot():
            try:
                b = get_bundle(None)
                print(f"[startup] match {b.match.match_id} loaded ({b.match.n_frames:,} frames)")
                start_warmup(b)
            except Exception as exc:
                print("[startup] could not preload a match:", exc)

        threading.Thread(target=boot, daemon=True).start()
    yield


app = FastAPI(title="2awasheet analytics API", version="1.0", default_response_class=FastJSON, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # any page served from this computer (localhost / 127.0.0.1, any port: Vite moves to 5174... when 5173 is busy)
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
try:  # file uploads need the small "python-multipart" package
    import python_multipart  # noqa: F401
    UPLOAD_ENABLED = True
except ImportError:
    try:
        import multipart  # noqa: F401
        UPLOAD_ENABLED = True
    except ImportError:
        UPLOAD_ENABLED = False


@app.get("/api/health")
def health():
    return {"status": "ok", "data_dir": str(default_data_dir()), "loaded_matches": list(_bundles.keys()), "upload_enabled": UPLOAD_ENABLED,
            "das_source": das_source(), "obso_source": "C-OBSO formula (Fujii)"}


@app.get("/api/matches")
def matches():
    """Every match that has a tracking file in data/ (cheap: nothing is loaded). No match is marked as 'the' default:
    the web app decides which one to show from the files the user opened."""
    out = []
    for mid in list_available_matches():
        f = find_match_files(mid)
        b = _bundles.get(mid)
        out.append({
            "match_id": mid,
            "tracking_bytes": f["tracking"].stat().st_size if f["tracking"] else None,
            "has_events": bool(f["events"]),
            "loaded": b is not None,
            "analytics_state": b.warm_state if b else "idle",
        })
    return {"matches": [m["match_id"] for m in out], "details": out, "upload_enabled": UPLOAD_ENABLED}


def _game_id_from_tracking_header(raw_line: bytes) -> Optional[str]:
    try:
        d = json.loads(raw_line)
    except Exception:
        return None
    if not isinstance(d, dict) or "homePlayers" not in d:
        raise HTTPException(400, "This does not look like a PFF tracking file (first line has no 'homePlayers').")
    v = d.get("gameRefId")
    return str(int(float(v))) if v not in (None, "") and re.match(r"^\d+(\.0)?$", str(v)) else None


def _game_id_from_events(raw: bytes) -> Optional[str]:
    try:
        d = json.loads(raw)
    except Exception:
        return None
    first = d[0] if isinstance(d, list) and d else d
    if isinstance(first, dict):
        v = first.get("gameId") or first.get("game_id") or (first.get("gameEvents") or {}).get("gameId")
        if v not in (None, "") and re.match(r"^\d+(\.0)?$", str(v)):
            return str(int(float(v)))
    return None


if UPLOAD_ENABLED:
    from fastapi import File, Form, UploadFile

    @app.post("/api/matches/upload")
    def upload_match(tracking: Optional[UploadFile] = File(None), events: Optional[UploadFile] = File(None),
                     metadata: Optional[UploadFile] = File(None), match_id: Optional[str] = Form(None)):
        """Save a match sent from the web app into data/ so that every metric can be computed for it.
        `tracking` may be left out to replace only the events / metadata of a match that is already on the server.
        The match id is taken from (in this order): the form field, the tracking file's gameRefId, the events file's
        gameId, the digits in the tracking file name."""
        data_dir = Path(default_data_dir())
        data_dir.mkdir(parents=True, exist_ok=True)
        if tracking is None and events is None and metadata is None:
            raise HTTPException(422, "Send at least a tracking file.")
        track_id = None
        if tracking is not None:
            head = tracking.file.readline(8 * 1024 * 1024)
            tracking.file.seek(0)
            track_id = _game_id_from_tracking_header(head)
        events_raw = None
        if events is not None:
            events_raw = events.file.read()
            events.file.seek(0)
        event_id = _game_id_from_events(events_raw) if events_raw else None
        if track_id and event_id and track_id != event_id:
            raise HTTPException(422, f"The events file belongs to match {event_id} but the tracking file belongs to match {track_id}.")
        mid = (clean_match_id(match_id) if match_id else None) or track_id or event_id
        if mid is None and tracking is not None:
            m_ = re.search(r"(\d{3,})", tracking.filename or "")
            mid = m_.group(1) if m_ else None
        if mid is None:
            raise HTTPException(422, "Could not find the match id in the files. Send it as the 'match_id' form field.")
        mid = clean_match_id(mid)
        if (track_id and track_id != mid) or (event_id and event_id != mid):
            raise HTTPException(422, f"match_id {mid} does not agree with the id found inside the files ({track_id or event_id}).")
        if tracking is None and not find_match_files(mid)["tracking"]:
            raise HTTPException(422, f"Match {mid} has no tracking file on the server yet; send the tracking file too.")
        targets, sources = {}, {}
        if tracking is not None:
            compressed = (tracking.filename or "").lower().endswith(".bz2")
            targets["tracking"] = data_dir / (f"tracking_{mid}.jsonl.bz2" if compressed else f"tracking_{mid}.jsonl")
            sources["tracking"] = tracking
        if events is not None:
            targets["events"], sources["events"] = data_dir / f"events_{mid}.json", events
        if metadata is not None:
            targets["metadata"], sources["metadata"] = data_dir / f"metadata_{mid}.json", metadata
        with _registry_lock:  # a re-upload replaces the match: forget the old in-memory copy first
            if _bundles.get(mid) and _bundles[mid].warm_state == "running":
                raise HTTPException(409, f"match {mid} is being computed right now; try again in a minute")
            _bundles.pop(mid, None)
        for kind, dest in targets.items():
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(sources[kind].file, out, 1024 * 1024)
            os.replace(tmp, dest)  # the file appears complete or not at all
        return {"match_id": mid, "saved": {k: v.name for k, v in targets.items()}}


@app.get("/api/match")
def match_info(match_id: Optional[str] = None):
    """Match overview: teams, fps, periods with their frame ranges (used by the time slider)."""
    b = get_bundle(match_id)
    m = b.match
    periods = []
    for p in sorted(set(m.period.tolist())):
        idx = np.flatnonzero(m.period == p)
        periods.append({"period": int(p), "first_frame": int(m.frame_num[idx[0]]), "last_frame": int(m.frame_num[idx[-1]]),
                        "duration_s": float(m.time_s[idx[-1]]), "attack_dir": m.attack_dir[int(p)]})
    return {
        "match_id": m.match_id, "fps": m.fps, "n_frames": m.n_frames, "teams": m.teams, "periods": periods,
        "bins": BIN_LABELS, "pitch": {"length": m.grid.length, "width": m.grid.width, "nx": m.grid.nx, "ny": m.grid.ny},
        "quality": m.quality, "warmup": b.warm_state, "analytics_ready": b.analytics.ready,
    }


@app.get("/api/metrics/live-segments")
def live_segments(match_id: Optional[str] = None, min_len_s: float = Query(3.0, ge=0.5, le=60.0),
                  merge_gap_s: float = Query(1.5, ge=0.0, le=10.0)):
    """Stretches of each period where the ball is tracked ('live play'). Everything outside them is dead ball
    (out of play, set-ups, stoppages) where DAS / OBSO / EPV cannot be computed because they need the ball.
    Times are seconds from the start of the period, the same clock the time slider uses."""
    b = get_bundle(match_id)
    m = b.match
    out = []
    for p in sorted(set(m.period.tolist())):
        idx = np.flatnonzero(m.period == p)
        t = (m.frame_num[idx] - m.frame_num[idx[0]]) / m.fps
        ok = np.isfinite(m.ball_x[idx]) & np.isfinite(m.ball_y[idx])
        d = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        segs: list[list[float]] = []
        for s0, e0 in zip(starts, ends):
            a, z = float(t[s0]), float(t[e0 - 1])
            if segs and a - segs[-1][1] < merge_gap_s:
                segs[-1][1] = z
            else:
                segs.append([a, z])
        segs = [[round(a, 2), round(z, 2)] for a, z in segs if z - a >= min_len_s]
        out.append({"period": int(p), "duration_s": round(float(t[-1]), 2), "segments": segs,
                    "live_s": round(sum(z - a for a, z in segs), 1)})
    return {"match_id": m.match_id, "periods": out}


@app.get("/api/players")
def players(match_id: Optional[str] = None):
    """Players of both teams."""
    b = get_bundle(match_id)
    lst = b.analytics.players_list()
    return {"match_id": b.match.match_id, "teams": b.match.teams,
            "home": [p for p in lst if p["side"] == "home"], "away": [p for p in lst if p["side"] == "away"]}


@app.get("/api/metrics/frame/{frame_id}")
def metrics_frame(frame_id: int, match_id: Optional[str] = None):
    """Player coordinates, pitch-control heat map, DAS surface, OBSO per attacker and EPV for one frame.
    `frame_id` is the PFF frameNum."""
    b = get_bundle(match_id)
    try:
        idx = b.match.frame_to_idx(frame_id)
    except KeyError:
        raise HTTPException(404, f"frame {frame_id} not found in match {b.match.match_id}")
    t = time.time()
    payload = b.engine.frame_payload(idx)
    payload["compute_ms"] = round((time.time() - t) * 1000, 1)
    return payload


@app.get("/api/metrics/epv-timeline")
def epv_timeline(match_id: Optional[str] = None, step_s: float = Query(2.0, ge=0.5, le=30.0)):
    """EPV of the ball location over the match, signed from the home team's point of view."""
    b = get_bundle(match_id)
    df = b.engine.epv_timeline(step_s)
    return {"match_id": b.match.match_id, "step_s": step_s, "points": df.round(5).to_dict("records")}


@app.get("/api/metrics/actions")
def actions(match_id: Optional[str] = None, from_frame: Optional[int] = None, to_frame: Optional[int] = None,
            period: Optional[int] = None, action: Optional[str] = None, limit: int = Query(2000, le=10000)):
    """On-ball actions with their VAEP and OBV values (markers / ratings for the pitch view)."""
    b = get_bundle(match_id)
    av = b.engine.action_values()
    if not len(av):
        return {"actions": []}
    if from_frame is not None:
        av = av[av["frame_num"] >= from_frame]
    if to_frame is not None:
        av = av[av["frame_num"] <= to_frame]
    if period is not None:
        av = av[av["period"] == period]
    if action:
        av = av[av["action"] == action]
    pl = b.match.players
    av = av.head(limit).copy()
    av["player"] = av["slot"].map(lambda s: pl.iloc[int(s)]["name"] if s >= 0 else None)
    av["jersey"] = av["slot"].map(lambda s: pl.iloc[int(s)]["jersey"] if s >= 0 else None)
    cols = ["frame_num", "period", "match_minute", "side", "player", "jersey", "action", "success", "x", "y", "end_x", "end_y",
            "epv_before", "epv_after", "delta_epv", "vaep", "vaep_oa", "obv"]
    return {"match_id": b.match.match_id, "n": int(len(av)), "actions": av[cols].round(5).to_dict("records")}


@app.get("/api/analytics/player/{player_id}")
def analytics_player(player_id: str, match_id: Optional[str] = None):
    """Decision efficiency, fatigue decay, chemistry matrix, decoy runs and press resistance of one player.
    The first call after start-up can take a minute or two while the heavy analytics are computed."""
    b = get_bundle(match_id)
    start_warmup(b)
    try:
        b.analytics.player_row(player_id)
    except KeyError:
        raise HTTPException(404, f"player {player_id} not found in match {b.match.match_id}")
    if b.warm_state == "error":
        raise HTTPException(500, f"analytics warm-up failed: {b.warm_error}")
    return b.analytics.player_report(player_id)  # blocks until the warm-up finished


@app.get("/api/analytics/status")
def analytics_status(match_id: Optional[str] = None):
    b = get_bundle(match_id)
    return {"match_id": b.match.match_id, "state": b.warm_state, "ready": b.analytics.ready, "error": b.warm_error}


@app.post("/api/analytics/warmup")
def analytics_warmup(match_id: Optional[str] = None):
    b = get_bundle(match_id)
    start_warmup(b)
    return {"match_id": b.match.match_id, "state": b.warm_state}


# ---------------------------------------------------------------------------
# Live player statistics: what had each player done by this moment of the match?
# ---------------------------------------------------------------------------
_SUM_KEYS = ["passes", "passes_ok", "shots", "shots_on_target", "goals", "xg", "tackles_won", "interceptions", "blocks", "clearances",
             "duels", "duels_won", "key_passes", "progressive_passes", "distance_m", "sprints"]


def _live_idx(b: Bundle, frame_id: Optional[int]) -> int:
    m = b.match
    if frame_id is None:
        return m.n_frames - 1
    try:
        return m.frame_to_idx(frame_id)
    except KeyError:
        raise HTTPException(404, f"frame {frame_id} not found in match {m.match_id}")


def _live_slot(b: Bundle, player_id: str) -> int:
    pl = b.match.players
    hit = pl.index[pl["player_id"].astype(str) == str(player_id)]
    if not len(hit):
        raise HTTPException(404, f"player {player_id} not found in match {b.match.match_id}")
    return int(pl.loc[hit[0], "slot"])


@app.get("/api/live/players")
def live_players(frame_id: Optional[int] = None, match_id: Optional[str] = None):
    """Every player's numbers as of `frame_id` (default: full time) + the totals of each team."""
    b = get_bundle(match_id)
    idx = _live_idx(b, frame_id)
    t = time.time()
    rows = b.live.table(idx)
    teams = {}
    for side in ("home", "away"):
        r = [x for x in rows if x["side"] == side]
        tot = {k: float(sum(x[k] for x in r)) for k in _SUM_KEYS}
        tot["pass_acc"] = tot["passes_ok"] / tot["passes"] if tot["passes"] else None
        tot["name"] = b.match.teams[side]["name"]
        teams[side] = tot
    return {"match_id": b.match.match_id, "frame_num": int(b.match.frame_num[idx]), "period": int(b.match.period[idx]),
            "minute": round(float(b.match.match_minute[idx]), 1), "players": rows, "teams": teams,
            "ball": ({"x": float(b.match.ball_x[idx]), "y": float(b.match.ball_y[idx])} if np.isfinite(b.match.ball_x[idx]) and np.isfinite(b.match.ball_y[idx]) else None),
            "has_events": bool(len(b.live.acts)), "compute_ms": round((time.time() - t) * 1000, 1)}


@app.get("/api/live/player/{player_id}")
def live_player(player_id: str, frame_id: Optional[int] = None, match_id: Optional[str] = None):
    """One player in detail as of `frame_id`: numbers, plain-English summary, heat map, pass / shot / defending maps."""
    b = get_bundle(match_id)
    idx = _live_idx(b, frame_id)
    slot = _live_slot(b, player_id)
    t = time.time()
    out = b.live.detail(slot, idx)
    return {**out, "compute_ms": round((time.time() - t) * 1000, 1)}


@app.get("/api/live/player/{player_id}/timeline")
def live_player_timeline(player_id: str, match_id: Optional[str] = None):
    """The whole match of one player minute by minute (cumulative distance, passes, value) + key moments."""
    b = get_bundle(match_id)
    slot = _live_slot(b, player_id)
    return b.live.timeline(slot)

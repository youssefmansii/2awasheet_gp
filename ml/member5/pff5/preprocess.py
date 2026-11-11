"""
Step 1 - convert raw PFF FC files into a compact, fast-to-load format.

Raw PFF tracking is ~860 MB of JSON per match. This script streams it once and
writes a ~10 MB ``<game_id>.npz`` holding int16 centimetre coordinates, plus a
slimmed ``<game_id>_events.json.gz`` (events without the embedded per-event
tracking snapshot).

Only needs the standard library + numpy, so it runs anywhere (Colab, laptop).

Usage
-----
    python -m pff5.preprocess --tracking "Tracking Data" --events "Event Data" \
        --metadata Metadata --rosters Rosters --out data/compact

Each directory may contain ``<id>.jsonl.bz2`` / ``<id>.jsonl`` / ``<id>.json``.
Inputs may also be ``.zip`` archives (members are streamed, nothing is unpacked).
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

try:  # ~3x faster parsing when available
    import orjson

    _loads = orjson.loads
except Exception:  # pragma: no cover
    _loads = json.loads

NA = -32768  # int16 sentinel for "no position"
VIS = {"VISIBLE": 1, "ESTIMATED": 2}
CONF = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
FORMAT_VERSION = 4


# --------------------------------------------------------------------------- #
# generic input helpers (plain files or zip members)
# --------------------------------------------------------------------------- #
class Source:
    """A file on disk or a member of a zip archive."""

    def __init__(self, path: str, member: str | None = None):
        self.path, self.member = path, member
        self.name = os.path.basename(member or path)

    def open(self):
        if self.member is None:
            return open(self.path, "rb")
        return zipfile.ZipFile(self.path).open(self.member)

    def open_text_lines(self):
        """Yield raw byte lines, transparently un-bz2/un-gzip-ing."""
        raw = self.open()
        if self.name.endswith(".bz2"):
            raw = bz2.open(raw, "rb")
        elif self.name.endswith(".gz"):
            raw = gzip.open(raw, "rb")
        return raw

    def read_json(self):
        with self.open_text_lines() as f:
            return _loads(f.read())


def discover(root: str | None, pattern: str) -> dict[str, Source]:
    """Map game id -> Source for every file under ``root`` (dir, zip or list of
    zips separated by ',') whose basename matches ``pattern`` (one group = id)."""
    out: dict[str, Source] = {}
    if not root:
        return out
    rx = re.compile(pattern)
    for r in root.split(","):
        r = r.strip()
        if not r:
            continue
        if os.path.isdir(r):
            for dp, _, fns in os.walk(r):
                for fn in fns:
                    m = rx.fullmatch(fn)
                    if m:
                        out.setdefault(m.group(1), Source(os.path.join(dp, fn)))
                    elif fn.lower().endswith(".zip"):
                        out_zip = discover(os.path.join(dp, fn), pattern)
                        for k, v in out_zip.items():
                            out.setdefault(k, v)
        elif zipfile.is_zipfile(r):
            with zipfile.ZipFile(r) as z:
                for info in z.infolist():
                    m = rx.fullmatch(os.path.basename(info.filename))
                    if m and not info.is_dir():
                        out.setdefault(m.group(1), Source(r, info.filename))
        elif os.path.isfile(r):
            m = rx.fullmatch(os.path.basename(r))
            if m:
                out.setdefault(m.group(1), Source(r))
    return out


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def _cm(v):
    if v is None:
        return NA
    v = int(round(v * 100.0))
    return v if -32767 <= v <= 32767 else NA


class _Team:
    def __init__(self, cap):
        self.slot: dict[int, int] = {}
        self.xy = np.full((cap, 32, 2), NA, np.int16)
        self.vis = np.zeros((cap, 32), np.uint8)   # low 2 bits visibility, next 2 bits confidence

    def put(self, i, players, overwrite=True):
        if not players:
            return
        slot, xy, vis = self.slot, self.xy, self.vis
        for p in players:
            try:
                j = int(p["jerseyNum"])
            except (KeyError, TypeError, ValueError):
                continue
            s = slot.get(j)
            if s is None:
                if len(slot) >= 32:
                    continue
                s = slot[j] = len(slot)
            x, y = p.get("x"), p.get("y")
            if x is None or y is None:
                continue
            if not overwrite and vis[i, s]:
                continue
            xy[i, s, 0] = _cm(x)
            xy[i, s, 1] = _cm(y)
            vis[i, s] = VIS.get(p.get("visibility"), 2) | (CONF.get(p.get("confidence"), 0) << 2)

    def grow(self, cap):
        n = self.xy.shape[0]
        self.xy = np.concatenate([self.xy, np.full((cap - n, 32, 2), NA, np.int16)])
        self.vis = np.concatenate([self.vis, np.zeros((cap - n, 32), np.uint8)])

    def finish(self, n):
        k = max(len(self.slot), 1)
        jerseys = np.zeros(k, np.int16)
        for j, s in self.slot.items():
            jerseys[s] = j
        return jerseys, self.xy[:n, :k].copy(), self.vis[:n, :k].copy()


def convert_tracking(src: Source, cap: int = 240_000) -> dict:
    """Stream one tracking file into dense numpy arrays."""
    home, away = _Team(cap), _Team(cap)          # PFF-smoothed
    home_r, away_r = _Team(cap), _Team(cap)      # raw broadcast tracking
    frame = np.zeros(cap, np.int32)
    period = np.zeros(cap, np.int8)
    t_video = np.zeros(cap, np.float64)
    t_period = np.zeros(cap, np.float32)
    ball = np.full((cap, 3), NA, np.int16)  # PFF-smoothed (event-informed, see README)
    ball_raw = np.full((cap, 3), NA, np.int16)
    ball_vis = np.zeros(cap, np.uint8)  # smoothed ball: 0 none, 1 visible, 2 estimated
    ge = np.zeros(cap, np.int64)
    pe = np.zeros(cap, np.int64)
    index: dict[int, int] = {}
    n = 0
    with src.open_text_lines() as f:
        for line in f:
            if not line or line.isspace():
                continue
            d = _loads(line)
            fn = d.get("frameNum")
            if fn is None:
                continue
            i = index.get(fn)
            if i is None:
                if n >= cap:
                    cap = int(cap * 1.5)
                    for tm in (home, away, home_r, away_r):
                        tm.grow(cap)
                    frame = np.resize(frame, cap)
                    period = np.resize(period, cap)
                    t_video = np.resize(t_video, cap)
                    t_period = np.resize(t_period, cap)
                    grow = cap - ball.shape[0]
                    ball = np.concatenate([ball, np.full((grow, 3), NA, np.int16)])
                    ball_raw = np.concatenate([ball_raw, np.full((grow, 3), NA, np.int16)])
                    ball_vis = np.concatenate([ball_vis, np.zeros(grow, np.uint8)])
                    ge = np.concatenate([ge, np.zeros(grow, np.int64)])
                    pe = np.concatenate([pe, np.zeros(grow, np.int64)])
                i = index[fn] = n
                n += 1
                frame[i] = fn
            period[i] = d.get("period") or 0
            t_video[i] = (d.get("videoTimeMs") or 0.0) / 1000.0
            t_period[i] = d.get("periodElapsedTime") or 0.0
            home.put(i, d.get("homePlayersSmoothed"))
            away.put(i, d.get("awayPlayersSmoothed"))
            home_r.put(i, d.get("homePlayers"))
            away_r.put(i, d.get("awayPlayers"))
            # ball
            b = d.get("balls")
            if isinstance(b, dict):
                b = [b]
            if b:
                b0 = b[0]
                if b0.get("x") is not None and b0.get("y") is not None:
                    ball_raw[i] = (_cm(b0["x"]), _cm(b0["y"]), _cm(b0.get("z") or 0.0))
            bs = d.get("ballsSmoothed")
            if isinstance(bs, list):
                bs = bs[0] if bs else None
            if bs and bs.get("x") is not None and bs.get("y") is not None:
                z = bs.get("z")
                if z is None:
                    z = (ball_raw[i, 2] / 100.0) if ball_raw[i, 2] != NA else 0.0
                ball[i] = (_cm(bs["x"]), _cm(bs["y"]), _cm(z))
                ball_vis[i] = VIS.get(bs.get("visibility"), 2)
            g = d.get("game_event_id")
            if g is not None:
                ge[i] = int(g)
            p = d.get("possession_event_id")
            if p is not None:
                pe[i] = int(p)
    order = np.argsort(frame[:n], kind="stable")
    # raw and smoothed must share one jersey -> slot table per team
    for sm, rw in ((home, home_r), (away, away_r)):
        jerseys = list(dict.fromkeys(list(sm.slot) + list(rw.slot)))[:32]
        for tm in (sm, rw):
            perm = [tm.slot.get(j, -1) for j in jerseys]
            xy = np.full((n, len(jerseys), 2), NA, np.int16)
            vis = np.zeros((n, len(jerseys)), np.uint8)
            for k, src_slot in enumerate(perm):
                if src_slot >= 0:
                    xy[:, k] = tm.xy[:n, src_slot]
                    vis[:, k] = tm.vis[:n, src_slot]
            tm.xy, tm.vis, tm.slot = xy, vis, {j: k for k, j in enumerate(jerseys)}
    hj, hxy, hvis = home.finish(n)
    aj, axy, avis = away.finish(n)
    _, hrxy, hrvis = home_r.finish(n)
    _, arxy, arvis = away_r.finish(n)
    return dict(
        home_raw_xy=hrxy[order], home_raw_vis=hrvis[order], away_raw_xy=arxy[order], away_raw_vis=arvis[order],
        frame=frame[:n][order], period=period[:n][order], t_video=t_video[:n][order],
        t_period=t_period[:n][order], home_jerseys=hj, home_xy=hxy[order], home_vis=hvis[order],
        away_jerseys=aj, away_xy=axy[order], away_vis=avis[order], ball=ball[:n][order],
        ball_raw=ball_raw[:n][order], ball_vis=ball_vis[:n][order],
        ge_id=ge[:n][order], pe_id=pe[:n][order],
    )


_DELTA_KEYS = ("home_xy", "away_xy", "home_raw_xy", "away_raw_xy", "ball", "ball_raw", "frame", "ge_id", "pe_id")


def delta_encode(arrays: dict) -> dict:
    """Lossless: store time-differences (wrap-around integer maths) - compresses ~3x better."""
    out = dict(arrays)
    for k in _DELTA_KEYS:
        a = arrays[k]
        d = a.copy()
        d[1:] = a[1:] - a[:-1]
        out[k] = d
    out["delta"] = np.array(1)
    return out


def delta_decode(z) -> dict:
    """Inverse of :func:`delta_encode`; accepts an ``np.load`` handle or a dict."""
    keys = z.files if hasattr(z, "files") else list(z)
    out = {k: z[k] for k in keys}
    if int(out.pop("delta", 0)):
        for k in _DELTA_KEYS:
            out[k] = np.cumsum(out[k], axis=0, dtype=out[k].dtype)
    return out


# --------------------------------------------------------------------------- #
# events / roster / metadata
# --------------------------------------------------------------------------- #
_DROP_EVENT_KEYS = ("homePlayers", "awayPlayers", "ball")


def slim_events(events: list[dict]) -> tuple[list[dict], dict]:
    """Drop the per-event tracking snapshot; derive a roster + pitch info from it."""
    roster: dict[str, dict] = {}
    info: dict = {}
    out = []
    for e in events:
        ge = e.get("gameEvents") or {}
        for side, is_home in (("homePlayers", True), ("awayPlayers", False)):
            for p in e.get(side) or []:
                pid = p.get("playerId")
                if pid is None:
                    continue
                r = roster.setdefault(str(int(pid)), {"playerId": int(pid), "home": is_home})
                r["jerseyNum"] = int(p["jerseyNum"])
                if p.get("positionGroupType"):
                    r["positionGroupType"] = p["positionGroupType"]
        if ge.get("teamId") is not None and ge.get("homeTeam") is not None:
            key = "homeTeamId" if ge["homeTeam"] else "awayTeamId"
            info.setdefault(key, int(ge["teamId"]))
        sm = e.get("stadiumMetadata") or {}
        for k in ("pitchLength", "pitchWidth", "homeTeamStartLeft"):
            if sm.get(k) is not None:
                info.setdefault(k, sm[k])
        out.append({k: v for k, v in e.items() if k not in _DROP_EVENT_KEYS})
    return out, {"roster": list(roster.values()), **info}


def normalise_roster(raw_roster, meta) -> list[dict]:
    """Official PFF roster file -> list of {playerId, jerseyNum, home, positionGroupType, name}."""
    out = []
    home_id = str(((meta or {}).get("homeTeam") or {}).get("id"))
    for r in raw_roster or []:
        try:
            out.append(dict(
                playerId=int(r["player"]["id"]), name=r["player"].get("nickname"),
                jerseyNum=int(r["shirtNumber"]), positionGroupType=r.get("positionGroupType"),
                home=str(r["team"]["id"]) == home_id, started=bool(r.get("started")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_meta(gid, meta_src, roster_src, derived, tracking) -> dict:
    meta = None
    if meta_src is not None:
        meta = meta_src.read_json()
        if isinstance(meta, list):
            meta = meta[0] if meta else None
    out = {"gameId": int(gid), "formatVersion": FORMAT_VERSION}
    if meta:
        pitch = ((meta.get("stadium") or {}).get("pitches") or [{}])[0]
        out.update(
            fps=meta.get("fps"), homeTeamStartLeft=meta.get("homeTeamStartLeft"),
            homeTeamStartLeftExtraTime=meta.get("homeTeamStartLeftExtraTime"),
            homeTeamId=int(meta["homeTeam"]["id"]), awayTeamId=int(meta["awayTeam"]["id"]),
            homeTeamName=meta["homeTeam"].get("name"), awayTeamName=meta["awayTeam"].get("name"),
            pitchLength=pitch.get("length"), pitchWidth=pitch.get("width"),
            startPeriod1=meta.get("startPeriod1"), startPeriod2=meta.get("startPeriod2"),
            date=meta.get("date"),
        )
    for k in ("homeTeamId", "awayTeamId", "pitchLength", "pitchWidth", "homeTeamStartLeft"):
        if out.get(k) is None and derived.get(k) is not None:
            out[k] = derived[k]
    if not out.get("fps"):
        # estimate from the tracking clock
        f, t = tracking["frame"], tracking["t_video"]
        ok = t > 0
        out["fps"] = float(np.round((f[ok][-1] - f[ok][0]) / (t[ok][-1] - t[ok][0]), 3)) if ok.sum() > 10 else 29.97
    out.setdefault("pitchLength", 105.0)
    out.setdefault("pitchWidth", 68.0)
    roster = None
    if roster_src is not None and meta:
        roster = normalise_roster(roster_src.read_json(), meta)
    if not roster:
        roster = derived.get("roster", [])
    else:  # add anything only seen in events (defensive)
        known = {r["playerId"] for r in roster}
        roster += [r for r in derived.get("roster", []) if r["playerId"] not in known]
    out["roster"] = roster
    return out


# --------------------------------------------------------------------------- #
def convert_game(gid, trk, evs, meta_src, roster_src, out_dir, overwrite=False, verbose=True):
    npz_path = os.path.join(out_dir, f"{gid}.npz")
    ev_path = os.path.join(out_dir, f"{gid}_events.json.gz")
    if not overwrite and os.path.exists(npz_path) and os.path.exists(ev_path):
        return "skipped"
    t0 = time.time()
    events, derived = slim_events(evs.read_json())
    arrays = convert_tracking(trk)
    meta = build_meta(gid, meta_src, roster_src, derived, arrays)
    tmp = npz_path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **delta_encode(arrays))
    os.replace(tmp, npz_path)
    tmp = ev_path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(events, f)
    os.replace(tmp, ev_path)
    if verbose:
        print(f"[{gid}] {len(arrays['frame'])} frames, {len(events)} events, "
              f"{os.path.getsize(npz_path) / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)
    return "done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", required=True, help="dir / zip(s) with <id>.jsonl.bz2")
    ap.add_argument("--events", required=True, help="dir / zip(s) with <id>.json event files")
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--rosters", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", default=None, help="comma separated ids (default: all found)")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop starting new games after this long")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    trk = discover(a.tracking, r"(\d+)\.jsonl(?:\.bz2|\.gz)?")
    evs = discover(a.events, r"(\d+)(?:_events)?\.json(?:\.gz)?")
    mds = discover(a.metadata, r"(\d+)\.json")
    ros = discover(a.rosters, r"(\d+)\.json")
    ids = sorted(set(trk) & set(evs), key=int)
    if a.games:
        ids = [g for g in ids if g in set(a.games.split(","))]
    print(f"{len(trk)} tracking, {len(evs)} event files -> {len(ids)} games to convert", flush=True)
    t0, done = time.time(), 0
    for gid in ids:
        if a.max_seconds and time.time() - t0 > a.max_seconds:
            break
        try:
            if convert_game(gid, trk[gid], evs[gid], mds.get(gid), ros.get(gid), a.out, a.overwrite) == "done":
                done += 1
        except Exception as ex:  # keep going, report at the end
            print(f"[{gid}] FAILED: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
    left = [g for g in ids if not os.path.exists(os.path.join(a.out, f"{g}.npz"))]
    print(f"converted {done}; remaining {len(left)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

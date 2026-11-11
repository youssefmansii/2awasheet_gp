"""
answer_key.py - PFF's hand-made shots (SH) = the ANSWER KEY
===========================================================

Used ONLY to (1) train the models and (2) score them.  Never imported by feature code.
Each SH event becomes one row with the tracking frame index it belongs to (via eventTime ->
tracking video time) and every Member-4 field PFF filled.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from .tracking_reader import Tracking, find_files

# PFF field -> our column name (Member 4 targets)
SHOT_FIELDS = {
    "shooterPlayerId": "shooter_id",
    "shotOutcomeType": "outcome",
    "keeperPlayerId": "keeper_id",
    "blockerPlayerId": "blocker_id",
    "failedBlockerPlayerId": "failed_blocker_id",
    "failedBlocker2PlayerId": "failed_blocker2_id",
    "deflectorPlayerId": "deflector_id",
    "glClearerPlayerId": "gl_clearer_id",
    "bodyMovementType": "body_movement",
    "ballMoving": "ball_moving",
    "shotInitialHeightType": "shot_height",
    "saveHeightType": "save_height",
    "saveReboundType": "save_rebound",
    "shotNatureType": "shot_nature",
    "shotType": "shot_type",          # group C (out of scope), kept for analysis
    "bodyType": "body_type",          # group C
}
ID_COLS = ["shooter_id", "keeper_id", "blocker_id", "failed_blocker_id", "failed_blocker2_id",
           "deflector_id", "gl_clearer_id"]


def _id(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def load_events(match_id) -> list:
    path = find_files(match_id)["events"]
    if not path:
        raise FileNotFoundError(f"no events file for match {match_id}")
    return json.load(open(path, encoding="utf-8"))


def load_shots(match_id, tr: Optional[Tracking] = None) -> pd.DataFrame:
    """One row per PFF SH event (+ inadvertent shots flagged with is_sh=False)."""
    rows = []
    for e in load_events(match_id):
        pe = e.get("possessionEvents") or {}
        ge = e.get("gameEvents") or {}
        is_sh = pe.get("possessionEventType") == "SH"
        if not is_sh and not pe.get("shotOutcomeType"):
            continue
        r = {"match_id": str(match_id), "event_id": _id(e.get("gameEventId")),
             "possession_event_id": _id(e.get("possessionEventId")), "is_sh": is_sh,
             "event_type": pe.get("possessionEventType"),
             "event_time_s": e.get("eventTime"), "period": ge.get("period"),
             "game_clock": pe.get("formattedGameClock"), "team_id": _id(ge.get("teamId")),
             "home_team": ge.get("homeTeam"), "setpiece_type": ge.get("setpieceType"),
             "video_url": pe.get("eventVideoUrl")}
        for k, c in SHOT_FIELDS.items():
            r[c] = pe.get(k)
        for c in ID_COLS:
            r[c] = _id(r[c])
        b = e.get("ball") or []
        b = b[0] if isinstance(b, list) and b else (b if isinstance(b, dict) else {})
        r["pff_ball_x"], r["pff_ball_y"], r["pff_ball_z"] = b.get("x"), b.get("y"), b.get("z")
        rows.append(r)
    df = pd.DataFrame(rows)
    if tr is not None and len(df):
        df["idx"] = [nearest_idx(tr, t, p) for t, p in zip(df.event_time_s, df.period)]
        df["frame"] = tr.frame[df["idx"].values]
        # shots outside the tracked video (e.g. penalty shoot-outs after tracking ends) cannot be seen
        df["in_tracking"] = np.abs(tr.t[df["idx"].values] - df.event_time_s.values) <= 1.0
    return df.sort_values("event_time_s").reset_index(drop=True)


def nearest_idx(tr: Tracking, t_s: float, period=None) -> int:
    """Index of the tracking row whose video time is closest to t_s (same period if given)."""
    if period is not None and not pd.isna(period):
        cand = np.flatnonzero(tr.period == int(period))
        if cand.size:
            i = cand[np.argmin(np.abs(tr.t[cand] - t_s))]
            return int(i)
    return int(np.argmin(np.abs(tr.t - t_s)))

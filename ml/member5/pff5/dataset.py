"""
Step 2 - turn every compact match into model-ready tables (cached as parquet).

    python -m pff5.dataset --compact data/compact --out data/tables --ball raw

Tables written to ``<out>/<ball>/<table>/<gid>.parquet``:

    press_ev     one row per pressure label (initial touch + event), event-level features
    press_cand   candidate pressers (nearest opponents) for those rows
    ch_rows      every PFF challenge: frame, participants, labels
    ch_part      every player at every challenge frame (participant ranking)
    ch_attr      challenge attribute features using the TRUE participants
    foul_rows    every possession event with its foul labels
    foul_ev      features for foul / offside detection on those events
    foul_cand    players around each infringement (culprit / victim ranking)
    frames       per-frame features at ~10 Hz for challenge spotting (+ distance to nearest challenge)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import features as F
from .data import game_path, list_games, load_match

TABLES = ["press_ev", "press_cand", "ch_rows", "ch_part", "ch_attr", "foul_rows", "foul_ev", "foul_cand", "frames"]
FRAME_STEP = 3


def _f32(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def spotting_labels(m, frames: pd.DataFrame) -> pd.DataFrame:
    """Signed time (s) from each sampled frame to the nearest challenge, + index of that challenge."""
    ev = m.events
    ch = ev[(ev["pe_possessionEventType"] == "CH") & ev["has_trk"]]
    ci = np.sort(ch["i_event"].to_numpy())
    j = frames["j"].to_numpy()
    if len(ci) == 0:
        frames["dt_ch"] = np.float32(99.0)
        return frames
    k = np.clip(np.searchsorted(ci, j), 1, len(ci) - 1) if len(ci) > 1 else np.zeros(len(j), int)
    if len(ci) > 1:
        left = (j - ci[k - 1]) < (ci[k] - j)
        k = k - left
    dt = (j - ci[k]) / m.fps
    dt[m.period[j] != m.period[ci[k]]] = 99.0
    frames["dt_ch"] = dt.astype(np.float32)
    # in-play flag from PFF's game events (used only to describe the data, never as a feature)
    return frames


def build_match_tables(compact_dir: str, gid: int, ball: str) -> dict[str, pd.DataFrame]:
    m = load_match(game_path(compact_dir, gid), ball_source=ball)
    out = {}
    out["press_ev"], out["press_cand"] = F.pressure_tables(m)
    ch = F.challenge_rows(m)
    out["ch_rows"] = ch
    out["ch_part"] = F.challenge_participant_table(m, ch["idx"].values, ch["home"].values, ch["away"].values,
                                                   ch["poss"].values)
    attr = F.challenge_attr_table(m, ch["idx"].values, ch["home"].values, ch["away"].values, ch["poss"].values)
    out["ch_attr"] = attr
    fr = F.foul_rows(m)
    out["foul_rows"] = fr
    out["foul_ev"] = F.event_foul_table(m, fr["idx"].values, fr["actor"].values, fr["pe_type"].values)
    fl = fr[fr["foul_type"].notna()]
    out["foul_cand"] = F.foul_candidate_table(m, fl["idx"].values, fl["actor"].values, fl["culprit"].values,
                                              fl["victim"].values)
    out["foul_cand"]["ev_row"] = out["foul_cand"]["key"].map(
        dict(zip((m.gid * 100000 + np.arange(len(fl))), fl["ev_row"].values))).values
    out["frames"] = spotting_labels(m, F.frame_table(m, step=FRAME_STEP))
    return {k: _f32(v) for k, v in out.items()}


def build_all(compact_dir: str, out_dir: str, ball: str, games=None, overwrite=False, max_seconds=None):
    games = games or list_games(compact_dir)
    t0 = time.time()
    for gid in games:
        done = all(os.path.exists(os.path.join(out_dir, ball, t, f"{gid}.parquet")) for t in TABLES)
        if done and not overwrite:
            continue
        if max_seconds and time.time() - t0 > max_seconds:
            print("time budget reached - run again to continue", flush=True)
            return False
        t1 = time.time()
        tabs = build_match_tables(compact_dir, gid, ball)
        for name, df in tabs.items():
            d = os.path.join(out_dir, ball, name)
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, f"{gid}.parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, os.path.join(d, f"{gid}.parquet"))
        print(f"[{ball}] {gid}: " + ", ".join(f"{k}={len(v)}" for k, v in tabs.items()) + f"  ({time.time() - t1:.1f}s)",
              flush=True)
    return True


def load_table(out_dir: str, ball: str, name: str, games=None, columns=None, frame_filter=None) -> pd.DataFrame:
    d = os.path.join(out_dir, ball, name)
    files = sorted((f for f in os.listdir(d) if f.endswith(".parquet")), key=lambda f: int(f[:-8]))  # numeric match order
    if games is not None:
        keep = {f"{g}.parquet" for g in games}
        files = [f for f in files if f in keep]
    parts = []
    for f in files:
        df = pd.read_parquet(os.path.join(d, f), columns=columns)
        if frame_filter is not None:
            df = frame_filter(df)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def table_games(out_dir: str, ball: str, name: str = "ch_rows") -> list[int]:
    d = os.path.join(out_dir, ball, name)
    return sorted(int(f[:-8]) for f in os.listdir(d) if f.endswith(".parquet")) if os.path.isdir(d) else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ball", default="raw", choices=["raw", "smoothed", "both"])
    ap.add_argument("--games", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args(argv)
    games = [int(g) for g in a.games.split(",")] if a.games else None
    ok = True
    for ball in (["raw", "smoothed"] if a.ball == "both" else [a.ball]):
        ok &= bool(build_all(a.compact, a.out, ball, games, a.overwrite, a.max_seconds))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())

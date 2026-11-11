"""
check_data.py - Step 1 data check: can tracking "see" PFF's shots?
==================================================================

For every training match given, and every PFF SH event, it measures:
  * is the ball tracked around the shot (+-0.5 s)?  raw ball vs PFF-smoothed ball
  * is the ball height z there and plausible?
  * is the shooter the player nearest the ball, and how far is he?
  * where (in frames) is the real kick compared with PFF's event time?
  * does the attacking direction from the metadata agree with the goalkeepers' positions?
Answer-key use is allowed here: this is a check, not a feature.

Run (from apps/api):   python -m event_detection.check_data --match 3816 3828 3829 3830 3831 10514
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .answer_key import load_shots
from .tracking_reader import cache_dir, load_tracking

UNSEEN = set(json.load(open(cache_dir().parents[1] / "split_member4.json"))["unseen"])


def ball_speed(xyz: np.ndarray, fps: float) -> np.ndarray:
    """2-D ball speed (m/s) from consecutive frames; NaN where a neighbour is missing."""
    v = np.full(len(xyz), np.nan, np.float32)
    d = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) * fps
    v[1:] = d
    return v


def check_match(mid: str) -> tuple[dict, pd.DataFrame]:
    if int(mid) in UNSEEN:
        raise SystemExit(f"match {mid} is in the locked UNSEEN set - not allowed in step 1")
    tr = load_tracking(mid)
    sh = load_shots(mid, tr)
    shots = sh[sh.is_sh].copy()
    fps = tr.fps
    braw = tr.ball_raw
    raw_ok = np.isfinite(braw[:, 0])
    z = braw[raw_ok, 2]
    spd = ball_speed(braw, fps)
    slot_of = {pid: s for s, pid in zip(tr.players.slot, tr.players.player_id)}
    side_of = dict(zip(tr.players.slot, tr.players.side))
    rows = []
    for _, r in shots.iterrows():
        i = int(r.idx); w = int(round(0.5 * fps)); W = int(round(1.5 * fps))
        lo, hi = max(0, i - w), min(tr.n_frames, i + w + 1)
        s_slot = slot_of.get(r.shooter_id)
        d_all = np.hypot(tr.px[i] - braw[i, 0], tr.py[i] - braw[i, 1])
        nearest = int(np.nanargmin(d_all)) if np.isfinite(d_all).any() else -1
        # shooter-ball distance in +-1.5 s and the kick (biggest ball speed jump with shooter < 3 m)
        L, H = max(0, i - W), min(tr.n_frames, i + W + 1)
        ds = np.hypot(tr.px[L:H, s_slot] - braw[L:H, 0], tr.py[L:H, s_slot] - braw[L:H, 1]) if s_slot is not None else np.full(H - L, np.nan)
        jump = np.full(H - L, np.nan)
        sp = spd[L:H]
        for k in range(H - L):
            a, b = sp[max(0, k - 3):k], sp[k + 1:k + 5]
            if np.isfinite(a).any() and np.isfinite(b).any() and np.isfinite(ds[k]) and ds[k] < 3.0:
                jump[k] = np.nanmedian(b) - np.nanmedian(a)
        kick_off = (int(np.nanargmax(jump)) + L - i) if np.isfinite(jump).any() else None
        after = spd[i:min(tr.n_frames, i + int(fps))]
        rows.append({
            "match_id": mid, "clock": r.game_clock, "period": r.period, "outcome": r.outcome,
            "setpiece": r.setpiece_type,
            "raw_ball_share_pm05s": float(raw_ok[lo:hi].mean()),
            "ball_z_at_event": float(braw[i, 2]) if raw_ok[i] else np.nan,
            "pff_ball_z": r.pff_ball_z,
            "shooter_dist_at_event_m": float(d_all[s_slot]) if s_slot is not None and np.isfinite(d_all[s_slot]) else np.nan,
            "shooter_min_dist_pm15s_m": float(np.nanmin(ds)) if np.isfinite(ds).any() else np.nan,
            "nearest_is_shooter": nearest == s_slot,
            "kick_offset_frames": kick_off,
            "max_ball_speed_1s_after": float(np.nanmax(after)) if np.isfinite(after).any() else np.nan,
            "shooter_side": side_of.get(s_slot),
        })
    per = pd.DataFrame(rows)
    # attack-direction sanity: home GK should stand on the side opposite to where home attacks
    dir_ok = []
    for p in np.unique(tr.period):
        sel = tr.period == p
        for side in ("home", "away"):
            for g in tr.gk_slots(side):
                x = tr.px[sel, g]
                if np.isfinite(x).sum() > 1000:
                    dir_ok.append(np.sign(np.nanmedian(x)) == -tr.attack_sign(side, int(p)))
    summ = {
        "match_id": mid, "teams": f"{tr.meta['homeTeam'].get('name')} v {tr.meta['awayTeam'].get('name')}",
        "frames": tr.n_frames, "missing_frames": int((~tr.observed).sum()),
        "raw_ball_present": float(raw_ok.mean()),
        "raw_ball_visible_flag": float((tr.ball_raw_vis == 1).mean()),
        "z_share_above_0.5m": float((z > 0.5).mean()), "z_negative_share": float((z < 0).mean()),
        "z_max": float(np.nanmax(z)), "attack_direction_agrees_with_gk": bool(all(dir_ok)) if dir_ok else None,
        "sh_events": int(len(shots)), "inadvertent_shots": int((~sh.is_sh).sum()),
    }
    return summ, per


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    a = ap.parse_args()
    S, P = [], []
    for m in a.match:
        s, p = check_match(m)
        S.append(s); P.append(p)
    S, P = pd.DataFrame(S), pd.concat(P, ignore_index=True)
    out = cache_dir()
    S.to_csv(out / "step1_matches.csv", index=False); P.to_csv(out / "step1_shots.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(S.round(3).to_string(index=False))
    print("\nALL SHOTS:", len(P))
    print(" raw ball tracked >=50% of +-0.5 s:", int((P.raw_ball_share_pm05s >= 0.5).sum()))
    print(" shooter is nearest player at PFF frame:", int(P.nearest_is_shooter.sum()))
    print(" shooter distance at PFF frame (m) median / 90%:", round(P.shooter_dist_at_event_m.median(), 2),
          round(P.shooter_dist_at_event_m.quantile(0.9), 2))
    print(" shooter min distance in +-1.5 s (m) median / 90%:", round(P.shooter_min_dist_pm15s_m.median(), 2),
          round(P.shooter_min_dist_pm15s_m.quantile(0.9), 2))
    ko = P.kick_offset_frames.dropna()
    print(" kick found:", len(ko), "| kick offset vs PFF frame (frames) median", ko.median(),
          "| within +-5 frames:", int((ko.abs() <= 5).sum()), "| within +-15:", int((ko.abs() <= 15).sum()))
    print(" ball z present at event:", int(P.ball_z_at_event.notna().sum()),
          "| z>0.5 m:", int((P.ball_z_at_event > 0.5).sum()))
    print(" max ball speed in 1 s after (m/s) median:", round(P.max_ball_speed_1s_after.median(), 1))
    print(" outcomes:", P.outcome.value_counts().to_dict(), "| set pieces:", P.setpiece.value_counts().to_dict())

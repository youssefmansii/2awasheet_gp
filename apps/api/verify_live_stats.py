"""
verify_live_stats.py  -  checks that the live player statistics are LOGICALLY right
===================================================================================

    python verify_live_stats.py --match 3816

1. RECOUNT   every counted statistic is counted a second time straight from the raw events file (no code from
             data_loader / player_live is used for that) and compared player by player.
2. RULES     things that must always be true (completed <= attempted, shots = sum of their outcomes, ...).
3. TIME      the numbers can only grow as the match moves forward; the last minute of the timeline equals the
             full-time table; before the first event everything is zero.
4. RANGES    minutes, distance, speeds are physically possible.

Prints PASS / FAIL for each check and exits with a non-zero code if anything failed.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from data_loader import find_match_files, load_match
from metrics_engine import MetricsEngine
from player_live import LivePlayers

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((bool(ok), name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def _id(v):
    try:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(float(v)))
    except (TypeError, ValueError):
        return None


def raw_table(path) -> pd.DataFrame:
    raw = json.load(open(path, encoding="utf-8"))
    raw = raw if isinstance(raw, list) else raw.get("events", raw)
    rows = []
    for e in raw:
        pe = e.get("possessionEvents") or {}
        rows.append({k: pe.get(v) for k, v in dict(t="possessionEventType", passer="passerPlayerId", po="passOutcomeType", shooter="shooterPlayerId",
                                                     so="shotOutcomeType", crosser="crosserPlayerId", co="crossOutcomeType", clearer="clearerPlayerId",
                                                     win="challengeWinnerPlayerId", chal="challengerPlayerId", ct="challengeType",
                                                     defender="defenderPlayerId", blocker="blockerPlayerId", keeper="keeperPlayerId",
                                                     disc="disciplinePlayerId").items()})
    R = pd.DataFrame(rows)
    for c in ["passer", "shooter", "crosser", "clearer", "win", "chal", "defender", "blocker", "keeper", "disc"]:
        R[c] = R[c].map(_id)
    return R


def recount(T: pd.DataFrame, R: pd.DataFrame) -> None:
    P, S, C = R[R.t == "PA"], R[R.t == "SH"], R[R.t == "CR"]

    def cmp(name: str, exp: pd.Series, col: str) -> None:
        bad = []
        for pid in set(exp.index) | set(T.index):
            want = int(exp.get(pid, 0))
            have = int(T.loc[pid, col]) if pid in T.index else None
            if have is None:
                if want:
                    bad.append(f"unknown player {pid} ({want})")
            elif have != want:
                bad.append(f"{T.loc[pid, 'name']}: raw {want} vs live {have}")
        check(f"recount {name} ({int(exp.sum())} in total)", not bad, "; ".join(bad[:4]))

    cmp("passes", P.groupby("passer").size(), "passes")
    cmp("completed passes", P[P.po == "C"].groupby("passer").size(), "passes_ok")
    cmp("shots", S.groupby("shooter").size(), "shots")
    cmp("shots saved", S[S.so == "S"].groupby("shooter").size(), "shots_saved")
    cmp("shots off target", S[S.so == "O"].groupby("shooter").size(), "shots_off")
    cmp("shots blocked", S[S.so == "B"].groupby("shooter").size(), "shots_blocked")
    cmp("crosses", C.groupby("crosser").size(), "crosses")
    cmp("completed crosses", C[C.co == "C"].groupby("crosser").size(), "crosses_ok")
    cmp("clearances", R[R.t == "CL"].groupby("clearer").size(), "clearances")
    cmp("interceptions (passes + crosses defended)", pd.concat([P[P.po == "D"].defender, C[C.co == "D"].defender]).dropna().value_counts(), "interceptions")
    cmp("blocks (passes, crosses, shots)", pd.concat([P[P.po == "B"].blocker, C[C.co == "B"].blocker, S[S.so == "B"].blocker]).dropna().value_counts(), "blocks")
    cmp("goalkeeper saves", S[S.so == "S"].groupby("keeper").size(), "saves")
    cmp("tackles attempted", R[(R.t == "CH") & (R.ct == "T") & R.chal.notna()].groupby("chal").size(), "tackles")
    cmp("tackles won", R[(R.t == "CH") & (R.ct == "T") & (R.chal == R.win) & R.chal.notna()].groupby("chal").size(), "tackles_won")
    cmp("cautions", R[R.disc.notna()].groupby("disc").size(), "cautions")


def rules(T: pd.DataFrame) -> None:
    check("rule: completed passes <= passes", (T.passes_ok <= T.passes).all())
    check("rule: completed crosses <= crosses", (T.crosses_ok <= T.crosses).all())
    check("rule: shots = goals + saved + off + blocked + other + ruled out", (T.shots == T.goals + T.shots_saved + T.shots_off + T.shots_blocked + T.shots_other + T.goals_ruled_out).all())
    check("rule: shots on target = goals + saved", (T.shots_on_target == T.goals + T.shots_saved).all())
    check("rule: duels won <= duels, tackles won <= tackles", ((T.duels_won <= T.duels) & (T.tackles_won <= T.tackles)).all())
    check("rule: key passes <= completed passes, assists <= key passes", ((T.key_passes <= T.passes_ok + T.crosses_ok) & (T.assists <= T.key_passes)).all())
    check("rule: percentages are between 0 and 1", T[["pass_acc", "duel_win_pct", "save_pct"]].astype(float).stack().dropna().between(0, 1).all())
    check("rule: only goalkeepers have saves", (T[T.position != "GK"].saves == 0).all())
    check("rule: xG is never negative and never above shots", ((T.xg >= 0) & (T.xg <= T.shots + 1e-9)).all())


def ranges(T: pd.DataFrame, fps: float, n_frames: int) -> None:
    mins_total = n_frames / fps / 60
    check("range: minutes between 0 and the length of the data", T.minutes.between(0, mins_total + 0.1).all())
    check("range: top speed <= 36 km/h", (T.top_speed_kmh <= 36.01).all(), f"max {T.top_speed_kmh.max():.1f}")
    check("range: distance <= 16 km, and an outfield player who played 80+ min ran at least 5 km", T.distance_m.between(0, 16000).all() and (T[(T.minutes >= 80) & (T.position != "GK")].distance_m >= 5000).all())
    check("range: high-speed + sprint metres <= distance", (T.hsr_m + T.sprint_m <= T.distance_m).all())
    check("range: sprints are rare enough to be real (<= 40 per player)", (T.sprints <= 40).all(), f"max {T.sprints.max()}")


def time_checks(lp: LivePlayers, md) -> None:
    n = md.n_frames
    idxs = np.unique(np.linspace(0, n - 1, 15).astype(int))
    prev = None
    mono = True
    why = ""
    keys = ["passes", "passes_ok", "shots", "goals", "interceptions", "clearances", "duels", "distance_m", "hsr_m", "sprints", "minutes", "xg"]
    for i in idxs:
        cur = pd.DataFrame(lp.table(int(i))).set_index("slot")[keys]
        if prev is not None:
            d = (cur - prev).min()
            bad = d[d < -1e-6]
            if len(bad):
                mono = False
                why = f"frame {i}: {list(bad.index)} went down"
                break
        prev = cur
    check("time: every cumulative number only grows as the match moves on", mono, why)
    t0 = pd.DataFrame(lp.table(0))
    check("time: at the first frame only the kick-off can have happened", (t0.passes.sum() <= 1) and (t0.shots.sum() == 0) and (t0.distance_m.sum() < 50))
    full = pd.DataFrame(lp.table(n - 1)).set_index("slot")
    slot = int(full.passes.idxmax())
    tl = lp.timeline(slot)["minutes"][-1]
    check("time: timeline's last minute equals the full-time table (passes, shots)", tl["passes"] == full.loc[slot, "passes"] and tl["shots"] == full.loc[slot, "shots"], f"{full.loc[slot, 'name']}")
    check("time: timeline distance equals the full-time distance (within 30 m)", abs(tl["distance_km"] * 1000 - full.loc[slot, "distance_m"]) < 30)
    half = pd.DataFrame(lp.table(n // 2)).set_index("slot")
    check("time: halfway numbers are smaller than full-time numbers", half.passes.sum() < full.passes.sum() and half.distance_m.sum() < full.distance_m.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    md = load_match(a.match, data_dir=a.data_dir, verbose=False)
    files = find_match_files(a.match, a.data_dir)
    lp = LivePlayers(MetricsEngine(md))
    T = pd.DataFrame(lp.table(md.n_frames - 1))
    if files.get("events"):
        print("--- 1. recount from the raw events file ---")
        recount(T.set_index("player_id"), raw_table(files["events"]))
    else:
        print("(no events file: recount skipped)")
    print("--- 2. rules ---")
    rules(T)
    print("--- 3. time ---")
    time_checks(lp, md)
    print("--- 4. ranges ---")
    ranges(T, md.fps, md.n_frames)
    print("--- totals ---")
    for side in ("home", "away"):
        t = T[T.side == side]
        print(f"{side}: passes {int(t.passes.sum())} ({t.passes_ok.sum() / max(t.passes.sum(), 1):.0%}), shots {int(t.shots.sum())} (on target {int(t.shots_on_target.sum())}, goals {int(t.goals.sum())}, ruled out {int(t.goals_ruled_out.sum())}), xG {t.xg.sum():.2f}, distance {t.distance_m.sum() / 1000:.1f} km")
    fails = [r for r in RESULTS if not r[0]]
    print(f"\nSUMMARY: {len(RESULTS) - len(fails)} PASS, {len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

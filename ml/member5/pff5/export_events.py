"""
Write the Member 5 predictions for one match as a PFF-style event file that the
team's replay app (apps/web) can load next to the tracking file.

    python -m pff5.export_events --run runs/raw --compact data/compact --game 3816 \
        --events-dir data/compact --out 3816_member5_events.json [--fold auto --tables data/tables]

Event types written (all predicted by the models, none copied from PFF):
    CH  challenges found from tracking alone: who has the ball -> challenger, type, winner
    FO  fouls flagged on events (culprit -> victim) and offside flags
    PR  pressure on the player on the ball (player <- presser), type and confidence
Pressure and fouls are predicted for the moments where an event happens; PFF's own
event moments stand in for the moments the other team members detect.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np

from . import features as F
from . import models as M
from .data import game_path, load_match
from .predict import Member5

TYPE_NAMES = {"A": "aerial duel", "FIFTY": "50/50", "D": "dribble", "B": "tackle from behind", "T": "standing tackle",
              "L": "slide tackle", "S": "shoulder to shoulder", "H": "shielding"}
P_NAMES = {"A": "attempted pressure", "L": "passing-lane pressure", "P": "pressured"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--events-dir", default=None, help="folder with <id>_events.json.gz (names and teams)")
    ap.add_argument("--fold", default=None, help="'auto' = the fold model that never saw this match (needs --tables)")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    fold = None
    if a.fold == "auto":
        from .dataset import table_games
        held = set(json.load(open(os.path.join(a.run, "results", "pressure.json"))).get("held_out_matches") or [])
        games = [g for g in table_games(a.tables, os.path.basename(os.path.normpath(a.run))) if g not in held]
        if a.game in games:
            fold = int(M.fold_of(np.array(games))[games.index(a.game)])
    elif a.fold is not None:
        fold = int(a.fold)
    m5 = Member5(a.run, fold)
    m = load_match(game_path(a.compact, a.game), with_events=True, ball_source=m5.ball)

    # names, teams
    names, teams = {}, {}
    ev_path = os.path.join(a.events_dir or a.compact, f"{a.game}_events.json.gz")
    for e in json.load(gzip.open(ev_path, "rt")):
        g = e.get("gameEvents") or {}
        if g.get("teamId") is not None and g.get("teamName"):
            teams[bool(g.get("homeTeam"))] = (int(g["teamId"]), g["teamName"])
        for part in ("gameEvents", "possessionEvents", "initialTouch", "fouls"):
            d = e.get(part) or {}
            for k, v in d.items():
                if k.endswith("PlayerId") and v is not None and d.get(k[:-2] + "Name"):
                    names[int(v)] = d[k[:-2] + "Name"]
    lo = m._period_bounds()[0]

    def clock(i):
        t = (i - lo[m.period[i]]) / m.fps
        mm = {1: 0, 2: 45, 3: 90, 4: 105}.get(int(m.period[i]), 0) + int(t // 60)
        return f"{mm:02d}:{int(t % 60):02d}"

    def pid(x):
        return int(x) if x is not None and x == x and int(x) > 0 else None

    def nm(p):
        return names.get(p, f"Player {p}") if p else None

    def side(p):
        if p is None:
            return None
        s = np.flatnonzero(m.player_id == p)
        return bool(m.team[s[0]] == 0) if len(s) else None

    def base(i, kind, player, team_home):
        t = float(m.t[i])
        tid, tname = teams.get(team_home, (None, None)) if team_home is not None else (None, None)
        return {"gameId": a.game, "gameEventId": None, "possessionEventId": None, "startTime": t, "endTime": t,
                "eventTime": t, "duration": 0.0, "source": "member5-model",
                "gameEvents": {"gameEventType": "OTB", "period": int(m.period[i]), "startFormattedGameClock": clock(i),
                               "teamId": tid, "teamName": tname, "homeTeam": team_home,
                               "playerId": player, "playerName": nm(player)},
                "possessionEvents": {"possessionEventType": kind, "formattedGameClock": clock(i)}}

    out = []
    # ---------------- challenges from tracking only
    ch = m5.challenges_from_tracking(m)
    for r in ch.to_dict("records"):
        i = int(r["frame_idx"])
        duel = r["challengeType"] in F.DUEL_TYPES
        a1 = pid(r["homeDuelPlayerId"]) if duel else pid(r["ballCarrierPlayerId"])
        a2 = pid(r["awayDuelPlayerId"]) if duel else pid(r["challengerPlayerId"])
        win = pid(r["challengeWinnerPlayerId"])
        e = base(i, "CH", a1, side(a1))
        pe = e["possessionEvents"]
        pe.update(challengeType=r["challengeType"], challengerPlayerId=a2, challengerPlayerName=nm(a2),
                  challengeWinnerPlayerId=win, challengeWinnerPlayerName=nm(win),
                  challengeOutcomeType=f"{TYPE_NAMES.get(r['challengeType'], r['challengeType'])}, won by {nm(win)}"
                                       f" (found {r['detection_score']:.0%})",
                  targetPlayerId=a2, targetPlayerName=nm(a2))
        if r.get("foulType") == "I":
            pe["challengeOutcomeType"] += ", foul"
        out.append(e)
    # ---------------- fouls and offside on event moments
    fres = json.load(open(os.path.join(a.run, "results", "fouls.json")))
    thr_f = fres["foul_given_any_event"]["all_events"]["best_f1_threshold"]
    thr_o = fres["offside"]["non_challenge_events"]["best_f1_threshold"]
    fr = F.foul_rows(m)
    fo = m5.fouls(m, fr["idx"].values, m.player_id[np.clip(fr["actor"].values, 0, None)], fr["pe_type"].values)
    for k, r in enumerate(fo.to_dict("records")):
        i = int(r["frame_idx"])
        if r["p_foul_given"] >= thr_f:
            cul, vic = pid(r.get("culpritPlayerId_if_infringement")), pid(r.get("victimPlayerId_if_infringement"))
            e = base(i, "FO", cul, side(cul))
            e["possessionEvents"].update(targetPlayerId=vic, targetPlayerName=f"{nm(vic)} (foul {r['p_foul_given']:.0%})",
                                         foulModelProbability=round(float(r["p_foul_given"]), 3))
            out.append(e)
        if r["p_offside"] >= thr_o:
            pas = pid(m.player_id[fr["actor"].values[k]]) if fr["actor"].values[k] >= 0 else None
            e = base(i, "FO", pas, side(pas))
            e["possessionEvents"].update(targetPlayerName=f"OFFSIDE pass ({r['p_offside']:.0%})",
                                         offsideModelProbability=round(float(r["p_offside"]), 3))
            out.append(e)
    # ---------------- pressure on event moments
    rows = F.pressure_rows(m)
    pr = m5.pressure(m, rows["idx"].values, m.player_id[rows["actor"].values], rows["pe_type"].values,
                     rows["kind"].values == 0, rows["hold"].values)
    for k, r in enumerate(pr.to_dict("records")):
        if r["pressureType"] == "N":
            continue
        i = int(r["frame_idx"])
        on_ball = pid(r["playerId"])
        pres = pid(r["pressurePlayerId"])
        conf = r[f"p_{r['pressureType']}"]
        e = base(i, "PR", on_ball, side(on_ball))
        e["possessionEvents"].update(pressureType=r["pressureType"], pressurePlayerId=pres, pressurePlayerName=nm(pres),
                                     targetPlayerId=pres,
                                     targetPlayerName=f"{nm(pres)} ({P_NAMES[r['pressureType']]} {conf:.0%})")
        out.append(e)
    out.sort(key=lambda e: e["eventTime"])
    # roster so the app can find the players on the pitch (team + jersey)
    roster = m.meta.get("roster", [])
    hp = [{"playerId": r["playerId"], "jerseyNum": r["jerseyNum"], "positionGroupType": r.get("positionGroupType")}
          for r in roster if r.get("home")]
    apl = [{"playerId": r["playerId"], "jerseyNum": r["jerseyNum"], "positionGroupType": r.get("positionGroupType")}
           for r in roster if not r.get("home")]
    for e in out:
        e["homePlayers"], e["awayPlayers"] = hp, apl
    with open(a.out, "w") as f:
        json.dump(out, f)
    cnt = {t: sum(e["possessionEvents"]["possessionEventType"] == t for e in out) for t in ("CH", "FO", "PR")}
    print(f"wrote {a.out}: {cnt}  (models: {a.run})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

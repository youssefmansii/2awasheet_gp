"""
Final test: score the saved models on a match that was held out of ALL training.

    python -m pff5.holdout --run runs/raw --compact data/compact --game 10517 --out runs/raw/results/holdout_10517.json

Uses the same public API a teammate would use (pff5.predict.Member5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import features as F
from .data import game_path, load_match
from .models import binary_report, clf_report
from .predict import Member5
from .train import CH_TYPES, PRESSURE_LABELS, codes, match_detections


def evaluate(run_dir: str, compact: str, gid: int) -> dict:
    m5 = Member5(run_dir)
    m = load_match(game_path(compact, gid), ball_source=m5.ball)
    res = dict(gameId=gid, ball_source=m5.ball,
               teams=f"{m.meta.get('homeTeamName', m.meta.get('homeTeamId'))} v {m.meta.get('awayTeamName', m.meta.get('awayTeamId'))}")

    # ---- pressure ----
    rows = F.pressure_rows(m)
    pr = m5.pressure(m, rows["idx"].values, m.player_id[rows["actor"].values], rows["pe_type"].values,
                     rows["kind"].values == 0, rows["hold"].values)
    y = codes(rows["y_type"].values, PRESSURE_LABELS)
    proba = pr[["p_N", "p_A", "p_L", "p_P"]].to_numpy()
    rep = clf_report(y, proba, PRESSURE_LABELS, mask=y >= 0)
    truth = np.where(rows["presser"].values >= 0, m.player_id[np.clip(rows["presser"].values, 0, None)], -1)
    has = (truth >= 0) & (y > 0)
    pred_any = pr["pressurePlayerId"].values
    res["pressure"] = dict(
        n=rep["n"], type_accuracy=rep["accuracy"], type_macro_f1=rep["macro_f1"],
        majority_baseline_accuracy=rep["majority_baseline_accuracy"],
        per_class_f1={k: v["f1"] for k, v in rep["per_class"].items()},
        pressured_events=int(has.sum()),
        presser_correct_when_flagged=float((pred_any[has & (pred_any >= 0)] == truth[has & (pred_any >= 0)]).mean()),
        type_and_presser_both_correct=float(((proba.argmax(1) == y) & (pred_any == truth))[has].mean()))

    # ---- challenges ----
    ch = F.challenge_rows(m)
    gt = np.sort(ch["idx"].values)
    det = m5.spot_challenges(m)
    tp, which = match_detections(det["frame_idx"].values, det["score"].values, gt, 1.0 * m.fps)
    prec, rec = tp.mean() if len(tp) else 0.0, tp.sum() / max(len(gt), 1)
    res["challenge_spotting_tracking_only"] = dict(
        real_challenges=int(len(gt)), detections=int(len(det)), precision=float(prec), recall=float(rec),
        f1=float(2 * prec * rec / max(prec + rec, 1e-9)), threshold=m5.spot_threshold)

    def ids(rec_df):
        a = np.where(rec_df["ballCarrierPlayerId"].values >= 0, rec_df["ballCarrierPlayerId"].values, rec_df["homeDuelPlayerId"].values)
        b = np.where(rec_df["challengerPlayerId"].values >= 0, rec_df["challengerPlayerId"].values, rec_df["awayDuelPlayerId"].values)
        return a, b

    def score(rec_df, g):
        hid = np.where(g["home"].values >= 0, m.player_id[np.clip(g["home"].values, 0, None)], -2)
        aid = np.where(g["away"].values >= 0, m.player_id[np.clip(g["away"].values, 0, None)], -3)
        a, b = ids(rec_df)
        hit_h = (a == hid) | (b == hid)
        hit_a = (a == aid) | (b == aid)
        win_slot = np.where(g["home_wins"].values == 1, hid, aid)
        wv = np.isfinite(g["home_wins"].values.astype(float))
        return dict(n=int(len(g)), both_players_correct=float((hit_h & hit_a).mean()),
                    at_least_one_player_correct=float((hit_h | hit_a).mean()),
                    type_accuracy=float((rec_df["challengeType"].values == g["ctype"].values).mean()),
                    winner_player_correct=float((rec_df["challengeWinnerPlayerId"].values[wv] == win_slot[wv]).mean()),
                    foul_flag_accuracy=float(((rec_df["foulType"].values == "I") == (g["foul_type"].values == "I")).mean())
                    if "foulType" in rec_df else None)

    res["challenge_details_on_true_moments"] = score(m5.challenges(m, ch["idx"].values), ch)
    poss_id = np.where(ch["poss"].values >= 0, m.player_id[np.clip(ch["poss"].values, 0, None)], None)
    res["challenge_details_on_true_moments_possessor_given"] = score(m5.challenges(m, ch["idx"].values, list(poss_id)), ch)
    if tp.any():
        lut = {int(v): i for i, v in enumerate(ch["idx"].values)}
        gi = np.array([lut[int(gt[w])] for w in which[tp]])
        rec_df = m5.challenges(m, det["frame_idx"].values[tp])
        e2e = score(rec_df, ch.iloc[gi])
        full = ((rec_df["challengeType"].values == ch.iloc[gi]["ctype"].values)).sum()
        e2e["note"] = "computed on the detections that matched a real challenge within 1 s"
        res["tracking_only_end_to_end"] = e2e

    # ---- fouls ----
    fr = F.foul_rows(m)
    fo = m5.fouls(m, fr["idx"].values, m.player_id[np.clip(fr["actor"].values, 0, None)], fr["pe_type"].values)
    given = np.array([v == "I" for v in fr["foul_type"].values]).astype(int)
    off = np.array([v == "O" for v in fr["offense"].values]).astype(int)
    not_ch = (fr["pe_type"] != "CH").values
    res["fouls"] = dict(events=int(len(fr)), fouls_given=int(given.sum()),
                        foul_given=binary_report(given, fo["p_foul_given"].values),
                        offside=binary_report(off[not_ch], fo["p_offside"].values[not_ch]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = evaluate(a.run, a.compact, a.game)
    out = a.out or os.path.join(a.run, "results", f"holdout_{a.game}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

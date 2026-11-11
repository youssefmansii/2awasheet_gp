"""
Collect the cross-validated metrics of both ball modes into one markdown file.

    python -m pff5.report --runs runs --out RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _load(runs, ball, task):
    p = os.path.join(runs, ball, "results", f"{task}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _f(v, pct=True):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(runs: str) -> str:
    R = {b: {t: _load(runs, b, t) for t in ("pressure", "participants", "attributes", "fouls", "spotting", "e2e")}
         for b in ("raw", "smoothed")}
    have = [b for b in R if R[b]["pressure"] or R[b]["spotting"]]
    n = max((_g(R[b][t], "n_matches", default=0) or 0) for b in R for t in R[b])
    held = next((_g(R[b][t], "held_out_matches") for b in R for t in R[b] if _g(R[b][t], "held_out_matches")), [])
    L = [f"# Member 5 results - {n} matches, 5-fold cross-validation grouped by match", "",
         "Every number is measured on matches the model did not see during training."
         + (f" Match {', '.join(str(h) for h in held)} was kept out of everything and is scored separately at the end."
            if held else ""),
         "`raw` = honest raw ball. `smoothed` = PFF's event-informed ball (leaks label information, see README).",
         "\"Baseline\" is the simplest sensible rule (most common class, or nearest player).", ""]
    hdr = ["Metric"] + [f"{b} ball" for b in have] + ["Baseline"]

    def row(name, fn, base=None, pct=True):
        return [name] + [_f(fn(R[b]), pct) for b in have] + [base if isinstance(base, str) else _f(base, pct)]

    # ---------------- pressure ----------------
    p0 = R[have[0]]["pressure"] if have else None
    if p0:
        L += ["## 1. Pressure", "", f"{_g(p0, 'n_events')} labelled moments (initial touches + events).", ""]
        rows = [
            row("Pressure type accuracy (N / A / L / P)", lambda r: _g(r["pressure"], "pressure_type", "accuracy"),
                _g(p0, "pressure_type", "majority_baseline_accuracy")),
            row("Pressure type macro-F1", lambda r: _g(r["pressure"], "pressure_type", "macro_f1"), "-"),
            row("- on initial touch: accuracy", lambda r: _g(r["pressure"], "pressure_type_initial_touch", "accuracy"),
                _g(p0, "pressure_type_initial_touch", "majority_baseline_accuracy")),
            row("- on the event itself: accuracy", lambda r: _g(r["pressure"], "pressure_type_on_event", "accuracy"),
                _g(p0, "pressure_type_on_event", "majority_baseline_accuracy")),
            row("Pressured vs not: best F1", lambda r: _g(r["pressure"], "pressured_vs_not", "best_f1"), "-"),
            row("Pressured vs not: AUC", lambda r: _g(r["pressure"], "pressured_vs_not", "auc"), "-"),
            row("Pressing player correct (top-1)", lambda r: _g(r["pressure"], "presser_pick", "model", "top1_accuracy"),
                _g(p0, "presser_pick", "baseline_nearest_opponent", "top1_accuracy")),
            row("Pressing player in top-2", lambda r: _g(r["pressure"], "presser_pick", "model", "top2_accuracy"),
                _g(p0, "presser_pick", "baseline_nearest_opponent", "top2_accuracy")),
            row("Type AND player both correct (pressured events)",
                lambda r: _g(r["pressure"], "end_to_end", "type_and_presser_both_correct"), "-"),
        ]
        L += [table(hdr, rows), ""]
        pc = _g(p0, "pressure_type", "per_class") or {}
        L += ["Per-class F1 (" + have[0] + " ball): " + ", ".join(f"{k} {v['f1']:.2f} (n={v['support']})" for k, v in pc.items()), ""]

    # ---------------- spotting ----------------
    s0 = R[have[0]]["spotting"] if have else None
    if s0:
        L += ["## 2. Finding challenges from tracking alone", "",
              f"{_g(s0, 'n_challenges')} challenges. A detection counts if it is within the tolerance of a real challenge.", ""]
        rows = []
        for tol in ("1s", "2s"):
            k = f"tolerance_{tol}"
            rows += [row(f"F1 within {tol}", lambda r, k=k: _g(r["spotting"], k, "best_f1"),
                         _g(s0, "baseline_proximity_rule", k, "best_f1")),
                     row(f"- precision / recall within {tol}",
                         lambda r, k=k: None, "-"),
                     row(f"Average precision within {tol}", lambda r, k=k: _g(r["spotting"], k, "average_precision"),
                         _g(s0, "baseline_proximity_rule", k, "average_precision"))]
            rows[-2] = [f"- precision / recall within {tol}"] + [
                f"{_f(_g(R[b]['spotting'], k, 'precision'))} / {_f(_g(R[b]['spotting'], k, 'recall'))}" for b in have] + ["-"]
        L += [table(hdr, rows), ""]

    # ---------------- participants + attributes ----------------
    for mode, title in (("possessor_unknown", "tracking only"), ("possessor_given", "player in possession known (from Member 1)")):
        a0 = R[have[0]]["attributes"] if have else None
        pt0 = R[have[0]]["participants"] if have else None
        if not (a0 and pt0):
            continue
        L += [f"## 3. Challenge details - {title}", "",
              "Scored on real challenge moments. \"End to end\" rows use the model's own choice of the two players.", ""]
        heads0 = _g(a0, mode, "with_true_participants", "heads") or {}

        def hd(r, name, key="accuracy", which="with_true_participants"):
            h = _g(r["attributes"], mode, which, "heads", name) or {}
            return h.get(key, h.get("accuracy_at_0.5") if key == "accuracy" else None)

        def base(name):
            return (heads0.get(name) or {}).get("majority_baseline_accuracy")

        rows = [
            row("Involved player correct (per team, top-1)", lambda r: _g(r["participants"], mode, "per_player", "top1_accuracy"),
                _g(pt0, mode, "baseline_closest_to_ball_and_opponent", "top1_accuracy")),
            row("Both involved players correct", lambda r: _g(r["participants"], mode, "pair_both_correct"), "-"),
            row("Challenge type (8 classes)", lambda r: hd(r, "type"), base("type")),
            row("Challenge type macro-F1", lambda r: hd(r, "type", "macro_f1"), "-"),
            row("Tackle/dribble vs duel structure", lambda r: _g(r["attributes"], mode, "with_true_participants",
                                                                  "player_id_fields", "duel_vs_tackle_structure_accuracy"), "-"),
            row("Which side has the ball", lambda r: hd(r, "home_has_ball"), base("home_has_ball")),
            row("Which side wins", lambda r: hd(r, "home_wins"), base("home_wins")),
            row("Outcome (13 classes)", lambda r: hd(r, "outcome"), base("outcome")),
            row("Foul given (accuracy)", lambda r: hd(r, "foul_given"), base("foul_given")),
            row("Foul given (best F1)", lambda r: hd(r, "foul_given", "best_f1"), "-"),
            row("Culprit side, given a foul", lambda r: hd(r, "culprit_is_home"), base("culprit_is_home")),
            row("Card, given a foul (AUC)", lambda r: hd(r, "card_given_foul", "auc"), "50.0%"),
            row("Dribble type (dribbles only)", lambda r: hd(r, "dribble_type"), base("dribble_type")),
            row("Tackle attempt type (dribbles only)", lambda r: hd(r, "tackle_attempt"), base("tackle_attempt")),
            row("Trick type", lambda r: hd(r, "trick_type"), base("trick_type")),
            row("End to end: carrier/dribbler player id", lambda r: _g(r["attributes"], mode, "with_predicted_participants",
                                                                      "player_id_fields", "carrier_or_dribbler_id_accuracy"), "-"),
            row("End to end: challenger player id", lambda r: _g(r["attributes"], mode, "with_predicted_participants",
                                                                  "player_id_fields", "challenger_id_accuracy"), "-"),
            row("End to end: duel player ids (home / away)", lambda r: None, "-"),
            row("End to end: winner player id", lambda r: _g(r["attributes"], mode, "with_predicted_participants",
                                                              "player_id_fields", "winner_id_accuracy"), "-"),
        ]
        rows[-2] = ["End to end: duel player ids (home / away)"] + [
            f"{_f(_g(R[b]['attributes'], mode, 'with_predicted_participants', 'player_id_fields', 'home_duel_player_id_accuracy'))} / "
            f"{_f(_g(R[b]['attributes'], mode, 'with_predicted_participants', 'player_id_fields', 'away_duel_player_id_accuracy'))}"
            for b in have] + ["-"]
        L += [table(hdr, rows), ""]
        pc = _g(a0, mode, "with_true_participants", "heads", "type", "per_class") or {}
        L += ["Challenge type per-class F1 (" + have[0] + " ball): " +
              ", ".join(f"{k} {v['f1']:.2f} (n={v['support']})" for k, v in pc.items()), ""]

    # ---------------- end to end ----------------
    e0 = R[have[0]]["e2e"] if have else None
    if e0:
        L += ["## 4. Tracking only -> complete challenge events", "",
              "Spot the challenge, then name both players and the type, all with models that never saw the match.", ""]
        rows = [
            row("Challenges found within 1 s (recall)", lambda r: _g(r["e2e"], "detection_recall")),
            row("Detections that are real challenges (precision)", lambda r: _g(r["e2e"], "detection_precision")),
            row("On found challenges: both players correct", lambda r: _g(r["e2e"], "on_true_positives", "both_participants_correct")),
            row("On found challenges: at least one player correct",
                lambda r: _g(r["e2e"], "on_true_positives", "at_least_one_participant_correct")),
            row("On found challenges: type correct", lambda r: _g(r["e2e"], "on_true_positives", "challenge_type_correct")),
            row("On found challenges: winning side correct", lambda r: _g(r["e2e"], "on_true_positives", "winning_team_correct")),
            row("Share of all real challenges fully right (time + both players + type)",
                lambda r: _g(r["e2e"], "fully_correct_challenge_events", "recall")),
        ]
        L += [table(["Metric"] + [f"{b} ball" for b in have], [r[:-1] for r in rows]), ""]

    # ---------------- fouls ----------------
    f0 = R[have[0]]["fouls"] if have else None
    if f0:
        L += ["## 5. Fouls and offside on events", "",
              f"{_g(f0, 'n_events')} possession events, {_g(f0, 'n_infringements')} with an infringement.", ""]
        fg = lambda r, k, m: _g(r["fouls"], "foul_given_any_event", k, m)
        rows = [
            row("Foul given, all events: best F1", lambda r: fg(r, "all_events", "best_f1"), "-"),
            row("Foul given, all events: AUC", lambda r: fg(r, "all_events", "auc"), "50.0%"),
            row("Foul given, challenges only: best F1", lambda r: fg(r, "challenges_only", "best_f1"), "-"),
            row("Foul given, other events: best F1", lambda r: fg(r, "non_challenge_events", "best_f1"), "-"),
            row("Offside: best F1", lambda r: _g(r["fouls"], "offside", "non_challenge_events", "best_f1"), "-"),
            row("Offside: AUC", lambda r: _g(r["fouls"], "offside", "non_challenge_events", "auc"), "50.0%"),
            row("Given / advantage / missed (accuracy)", lambda r: _g(r["fouls"], "foul_type_I_A_M", "accuracy"),
                _g(f0, "foul_type_I_A_M", "majority_baseline_accuracy")),
            row("Offense type (accuracy)", lambda r: _g(r["fouls"], "offense_type", "accuracy"),
                _g(f0, "offense_type", "majority_baseline_accuracy")),
            row("Card / outcome (accuracy)", lambda r: _g(r["fouls"], "foul_outcome_cards", "accuracy"),
                _g(f0, "foul_outcome_cards", "majority_baseline_accuracy")),
            row("Culprit player (top-1)", lambda r: _g(r["fouls"], "culprit_pick", "top1_accuracy"),
                _g(f0, "culprit_pick", "baseline_nearest_player", "top1_accuracy")),
            row("Victim player (top-1)", lambda r: _g(r["fouls"], "victim_pick", "top1_accuracy"),
                _g(f0, "victim_pick", "baseline_nearest_player", "top1_accuracy")),
            row("VAR used (AUC)", lambda r: _g(r["fouls"], "var_used", "auc"), "50.0%"),
        ]
        L += [table(hdr, rows), ""]

    # ---------------- untouched test match ----------------
    import glob
    H = {b: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(runs, b, "results", "holdout_*.json")))] for b in have}
    if any(H.values()):
        h0 = next(v[0] for v in H.values() if v)
        L += [f"## 6. Untouched test match: {h0.get('gameId')} ({h0.get('teams')})", "",
              "This match was never used for training, validation or choosing thresholds. "
              "It is one match, so these numbers are noisier than the cross-validated ones above.", ""]
        hh = lambda b, *p: _g(H[b][0], *p) if H[b] else None
        rows = [
            ["Pressure type accuracy"] + [_f(hh(b, "pressure", "type_accuracy")) for b in have] + [_f(_g(h0, "pressure", "majority_baseline_accuracy"))],
            ["Pressure type macro-F1"] + [_f(hh(b, "pressure", "type_macro_f1")) for b in have] + ["-"],
            ["Pressure: type AND player both correct"] + [_f(hh(b, "pressure", "type_and_presser_both_correct")) for b in have] + ["-"],
            ["Challenge spotting F1 within 1 s (tracking only)"] + [_f(hh(b, "challenge_spotting_tracking_only", "f1")) for b in have] + ["-"],
            ["- precision / recall"] + [f"{_f(hh(b, 'challenge_spotting_tracking_only', 'precision'))} / {_f(hh(b, 'challenge_spotting_tracking_only', 'recall'))}" for b in have] + ["-"],
            ["Challenge type (true moments)"] + [_f(hh(b, "challenge_details_on_true_moments", "type_accuracy")) for b in have] + ["-"],
            ["Both players correct (true moments, tracking only)"] + [_f(hh(b, "challenge_details_on_true_moments", "both_players_correct")) for b in have] + ["-"],
            ["Both players correct (possessor given)"] + [_f(hh(b, "challenge_details_on_true_moments_possessor_given", "both_players_correct")) for b in have] + ["-"],
            ["Winner player correct (possessor given)"] + [_f(hh(b, "challenge_details_on_true_moments_possessor_given", "winner_player_correct")) for b in have] + ["-"],
            ["Foul given: AUC"] + [_f(hh(b, "fouls", "foul_given", "auc")) for b in have] + ["50.0%"],
            ["Foul given: best F1"] + [_f(hh(b, "fouls", "foul_given", "best_f1")) for b in have] + ["-"],
        ]
        L += [table(hdr, rows), ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args(argv)
    md = build(a.runs)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

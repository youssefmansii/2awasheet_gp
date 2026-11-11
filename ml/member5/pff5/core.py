"""
The CORE deliverable: the part of Member 5's scope that tracking data supports well.

Instead of every PFF sub-label, the core answers a short list of questions and
attaches a confidence to each answer, so the team can keep only confident ones:

    pressure    is the player on the ball pressured?  by whom?
    challenge   what kind (aerial duel / ground duel / dribble / tackle)?
                which side had the ball?  which side won?  was a foul given?

    python -m pff5.core --tables data/tables --compact data/compact --out runs --ball raw --holdout 10517

Writes runs/<ball>/results/core.json and core.md.  Every number is out-of-fold
(the model never saw the match it is scored on).  "Coverage" = the share of
cases the model answers when it only answers above a confidence level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from . import models as M
from .train import ATTR_POSS, Run, cv, feats

GROUP_OF = {"A": "aerial_duel", "FIFTY": "ground_duel", "D": "dribble", "B": "tackle", "T": "tackle", "L": "tackle",
            "S": "tackle", "H": "tackle"}
GROUPS = ["aerial_duel", "ground_duel", "dribble", "tackle"]
LEVELS = (1.0, 0.75, 0.5, 0.25)


def coverage(correct, conf) -> list:
    """Accuracy when answering only the most confident 100% / 75% / 50% / 25% of cases."""
    correct = np.asarray(correct, float)
    conf = np.asarray(conf, float)
    order = np.argsort(-conf, kind="stable")
    out = []
    for c in LEVELS:
        k = max(1, int(round(c * len(order))))
        out.append(dict(coverage=c, n=int(k), accuracy=float(correct[order[:k]].mean()),
                        min_confidence=float(conf[order[k - 1]])))
    return out


def run_core(run: Run) -> dict:
    res = {}
    # ------------------------------------------------------------------ pressure (from saved out-of-fold predictions)
    o = run.load_oof("pressure")
    o = o[o["y"] >= 0]
    p = 1.0 - o["p_N"].to_numpy()
    y = (o["y"].to_numpy() > 0)
    pred = p > 0.5
    res["pressured_yes_no"] = dict(
        n=int(len(o)), baseline_accuracy=float(max(y.mean(), 1 - y.mean())),
        by_coverage=coverage(pred == y, np.maximum(p, 1 - p)),
        precision_when_saying_pressured=float(y[pred].mean()), recall=float(pred[y].mean()),
        on_event=dict(by_coverage=coverage((pred == y)[o["kind"].values == 1], np.maximum(p, 1 - p)[o["kind"].values == 1])),
        on_initial_touch=dict(by_coverage=coverage((pred == y)[o["kind"].values == 0], np.maximum(p, 1 - p)[o["kind"].values == 0])))
    has = y & (o["presser"].to_numpy() >= 0)
    hit = (o["pred_presser"].to_numpy() == o["presser"].to_numpy())
    res["pressing_player"] = dict(n=int(has.sum()), by_coverage=coverage(hit[has], p[has]))
    # ------------------------------------------------------------------ challenges
    rows = run.table("ch_rows")
    gold = run.table("ch_attr")
    gid = rows["gid"].values
    grp = np.array([GROUPS.index(GROUP_OF[t]) if t in GROUP_OF else -1 for t in rows["ctype"].values])
    foul = (rows["foul_type"].values == "I").astype(float)
    for use_poss in (0, 1):
        X = gold.drop(columns=[] if use_poss else ATTR_POSS)
        fl = feats(X)
        tag = "possessor_given" if use_poss else "tracking_only"
        r = {}
        ok = grp >= 0
        c = cv(X[fl], np.clip(grp, 0, None), gid, "multiclass", len(GROUPS), train_mask=ok)
        run.save_model(f"core_group_poss{use_poss}", c, dict(labels=GROUPS))
        pr = c["oof"]
        rep = M.clf_report(grp, pr, GROUPS, mask=ok)
        r["challenge_group"] = dict(n=rep["n"], baseline_accuracy=rep["majority_baseline_accuracy"], macro_f1=rep["macro_f1"],
                                    per_class={k: dict(f1=v["f1"], precision=v["precision"], recall=v["recall"],
                                                       support=v["support"]) for k, v in rep["per_class"].items()},
                                    by_coverage=coverage((pr.argmax(1) == grp)[ok], pr.max(1)[ok]))
        for name, col in (("side_with_ball", "home_has_ball"), ("winning_side", "home_wins"), ("foul_given", None)):
            v = foul if col is None else rows[col].values.astype(float)
            okb = np.isfinite(v)
            yb = np.where(okb, v, 0).astype(int)
            cb = cv(X[fl], yb, gid, train_mask=okb)
            pb = cb["oof"]
            entry = dict(n=int(okb.sum()), baseline_accuracy=float(max(yb[okb].mean(), 1 - yb[okb].mean())),
                         auc=M.binary_report(yb[okb], pb[okb]).get("auc"),
                         by_coverage=coverage(((pb > 0.5) == (yb == 1))[okb], np.maximum(pb, 1 - pb)[okb]))
            if col is None:  # for fouls the useful question is "when the model flags a foul, is it one?"
                order = np.argsort(-pb[okb])
                yy = yb[okb][order]
                entry["precision_of_most_confident_foul_flags"] = [
                    dict(share_of_all_fouls_found=float(yy[:k].sum() / max(yy.sum(), 1)), precision=float(yy[:k].mean()),
                         flags=int(k)) for k in (int(0.25 * yy.sum()), int(0.5 * yy.sum()), int(yy.sum()))]
            r[name] = entry
        res[f"challenge_{tag}"] = r
    return res


def _pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def to_markdown(res: dict, n_matches: int) -> str:
    L = [f"## Core deliverable - high-confidence answers ({n_matches} matches, scored on unseen matches)", "",
         "Each row shows accuracy when the model answers **all** cases, and when it answers only its most "
         "confident 75% / 50% / 25%. Baseline = always giving the most common answer.", ""]
    hdr = "| Question | Baseline | All cases | Top 75% | Top 50% | Top 25% |\n|---|---|---|---|---|---|"

    def row(name, e):
        c = {x["coverage"]: x["accuracy"] for x in e["by_coverage"]}
        return f"| {name} | {_pct(e.get('baseline_accuracy'))} | {_pct(c[1.0])} | {_pct(c[0.75])} | {_pct(c[0.5])} | {_pct(c[0.25])} |"

    L += ["**Pressure** (frame and player on the ball supplied by the event, as PFF defines it)", "", hdr,
          row("Is the player pressured? (yes / no)", res["pressured_yes_no"]),
          row("- on the event itself (pass, shot, ...)", res["pressured_yes_no"]["on_event"]),
          row("- on the initial touch", res["pressured_yes_no"]["on_initial_touch"]),
          row("Who is pressing? (exact player)", res["pressing_player"]), ""]
    for tag, title in (("challenge_possessor_given", "**Challenges - with the player in possession known (from Member 1)**"),
                       ("challenge_tracking_only", "**Challenges - tracking only**")):
        r = res[tag]
        L += [title, "", hdr,
              row("Kind: aerial duel / ground duel / dribble / tackle", r["challenge_group"]),
              row("Which side had the ball", r["side_with_ball"]),
              row("Which side won", r["winning_side"]),
              row("Foul given? (yes / no)", r["foul_given"]), ""]
        pc = r["challenge_group"]["per_class"]
        L += ["Kind, per class F1: " + ", ".join(f"{k} {v['f1']:.2f}" for k, v in pc.items()), ""]
        ff = r["foul_given"]["precision_of_most_confident_foul_flags"]
        L += ["Foul flags: " + "; ".join(f"finding {_pct(x['share_of_all_fouls_found'])} of fouls at {_pct(x['precision'])} precision"
                                          for x in ff) + f" (AUC {_pct(r['foul_given']['auc'])}).", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ball", default="raw", choices=["raw", "smoothed"])
    ap.add_argument("--holdout", default="")
    a = ap.parse_args(argv)
    run = Run(a.tables, a.compact, a.out, a.ball, [g for g in a.holdout.split(",") if g])
    res = run_core(run)
    with open(os.path.join(run.root, "results", "core.json"), "w") as f:
        json.dump(dict(n_matches=len(run.games), held_out_matches=run.holdout, **res), f, indent=1, default=float)
    md = to_markdown(res, len(run.games))
    with open(os.path.join(run.root, "results", "core.md"), "w") as f:
        f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

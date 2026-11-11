"""
Step 6 - blend LightGBM and transformer predictions.

Both model families are validated on the same match folds, so their
out-of-fold predictions line up row by row.  For each head we search one
blending weight w in [0, 1]  (p = w * p_deep + (1 - w) * p_gbm) and report
LightGBM alone, transformer alone and the blend.

    python -m pff5.ensemble --gbm runs/raw --deep runs/deep/raw

Run it after ``pff5.deep finetune`` has been run for every fold (0..4).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from .models import clf_report
from .train import CH_TYPES, PRESSURE_LABELS


def _best_blend(y, p_gbm, p_deep, labels):
    rows = []
    for w in np.linspace(0, 1, 11):
        p = w * p_deep + (1 - w) * p_gbm
        r = clf_report(y, p, labels)
        rows.append((float(w), r["accuracy"], r["macro_f1"]))
    best = max(rows, key=lambda t: t[2])
    return dict(gbm_only=dict(accuracy=rows[0][1], macro_f1=rows[0][2]),
                deep_only=dict(accuracy=rows[-1][1], macro_f1=rows[-1][2]),
                blend=dict(weight_deep=best[0], accuracy=best[1], macro_f1=best[2]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gbm", required=True)
    ap.add_argument("--deep", required=True)
    a = ap.parse_args(argv)
    out = {}

    def deep(name):
        files = sorted(glob.glob(os.path.join(a.deep, f"pred_{name}_fold*.parquet")))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None

    # pressure type
    d = deep("pressure")
    g = pd.read_parquet(os.path.join(a.gbm, "oof", "pressure.parquet"))
    if d is not None:
        mg = g.merge(d, on=["gid", "key"], suffixes=("_g", "_d"))
        mg = mg[mg["y"] >= 0]
        pg = mg[[f"p_{c}_g" for c in PRESSURE_LABELS]].to_numpy()
        pdp = mg[[f"p_{c}_d" for c in PRESSURE_LABELS]].to_numpy()
        out["pressure_type"] = dict(n=int(len(mg)), **_best_blend(mg["y"].values, pg, pdp, PRESSURE_LABELS))
    # the deep model's own validation reports, averaged over folds
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.deep, "results_fold*.json")))]
    if reps:
        def avg(path):
            vals = []
            for r in reps:
                v = r
                for k in path:
                    v = v.get(k, {}) if isinstance(v, dict) else {}
                if isinstance(v, (int, float)):
                    vals.append(v)
            return float(np.mean(vals)) if vals else None
        out["deep_alone_mean_over_folds"] = dict(
            folds=len(reps), challenge_type_accuracy=avg(["ch_type", "accuracy"]),
            challenge_outcome_accuracy=avg(["ch_outcome", "accuracy"]),
            participants_home_top1=avg(["participants_top1", "home"]),
            participants_away_top1=avg(["participants_top1", "away"]),
            pressure_type_macro_f1=avg(["pressure_type", "macro_f1"]), presser_top1=avg(["presser_top1", "accuracy"]),
            spotting_ap_1s=avg(["spotting_1s", "average_precision"]), spotting_f1_1s=avg(["spotting_1s", "best_f1"]))
    gb = {}
    for t in ("pressure", "participants", "attributes", "spotting"):
        p = os.path.join(a.gbm, "results", f"{t}.json")
        if os.path.exists(p):
            gb[t] = json.load(open(p))
    if gb:
        out["gbm_reference"] = dict(
            pressure_type_macro_f1=gb.get("pressure", {}).get("pressure_type", {}).get("macro_f1"),
            presser_top1=gb.get("pressure", {}).get("presser_pick", {}).get("model", {}).get("top1_accuracy"),
            spotting_ap_1s=gb.get("spotting", {}).get("tolerance_1s", {}).get("average_precision"),
            spotting_f1_1s=gb.get("spotting", {}).get("tolerance_1s", {}).get("best_f1"))
    path = os.path.join(a.deep, "ensemble.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
attacks_expert_check.py  -  compare the app's attack danger with expert (coach) ratings
=========================================================================================

The research standard for a danger measure is agreement with experts (Link, Lang & Seidenschwarz 2016 had three coaches rate
100 scenes blind on a 1-5 scale). This script prepares such a test and scores it.

1) Make a blind rating sheet (no scores shown), 30 attacks spread over the whole danger range, in random order:

       python attacks_expert_check.py --match 3816 --make-sheet expert_sheet.csv

   Each rater watches every attack in the "Top attacks" tab (use the minute / frame numbers) and writes a rating
   from 1 (harmless) to 5 (very dangerous) in a new column named after them, e.g. "rater_ahmed".
   Raters must not see the app's scores while rating.

2) Score the filled sheet:

       python attacks_expert_check.py --match 3816 --score expert_sheet.csv

   It reports: agreement between raters (Kendall's W), the rank correlation of the app's danger with the mean expert
   rating (Spearman and Kendall, with 95 % bootstrap intervals), how many of the experts' top 10 are also in the app's
   top 10, and the same numbers for simple baselines (shot xG only, best spot only, "reached the box"), so you can show
   the combined score does at least as well as its parts.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from attacks import AttackFinder
from data_loader import load_match
from metrics_engine import MetricsEngine


def make_sheet(at: list[dict], out: str, n: int = 30, seed: int = 7) -> None:
    df = pd.DataFrame(at).sort_values("danger_p").reset_index(drop=True)
    # spread over the whole range: n equal-count strata, one random attack from each
    rng = np.random.default_rng(seed)
    strata = np.array_split(np.arange(len(df)), min(n, len(df)))
    pick = [int(rng.choice(s)) for s in strata if len(s)]
    sheet = df.loc[pick, ["id", "team", "period", "minute", "start_frame", "end_frame", "replay_start_frame", "replay_end_frame"]].copy()
    sheet["minute"] = sheet["minute"].round(1)
    sheet = sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # random order, so the order gives nothing away
    sheet.insert(0, "order", range(1, len(sheet) + 1))
    sheet.to_csv(out, index=False)
    print(f"saved {out}: {len(sheet)} attacks. Add one column per rater (rating 1-5) and run --score.")


def kendalls_w(r: np.ndarray) -> float:
    """r: items x raters ratings -> Kendall's coefficient of concordance (ties ignored)."""
    m, k = r.shape[1], r.shape[0]
    ranks = np.column_stack([rankdata(r[:, j]) for j in range(m)])
    s = ((ranks.sum(axis=1) - m * (k + 1) / 2) ** 2).sum()
    return float(12 * s / (m ** 2 * (k ** 3 - k)))


def boot(x: np.ndarray, y: np.ndarray, f, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(x), len(x))
        v = f(x[i], y[i])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(at: list[dict], sheet_path: str) -> None:
    sheet = pd.read_csv(sheet_path)
    raters = [c for c in sheet.columns if c.startswith("rater")]
    if not raters:
        raise SystemExit("No rater columns found (name them rater_<name>).")
    by_id = {a["id"]: a for a in at}
    sheet = sheet[sheet["id"].isin(by_id)].dropna(subset=raters)
    r = sheet[raters].to_numpy(float)
    expert = r.mean(axis=1)
    print(f"{len(sheet)} attacks rated by {len(raters)} rater(s)")
    if len(raters) >= 2:
        print(f"  agreement between raters (Kendall's W, 0..1): {kendalls_w(r):.2f}   "
              "(the app cannot be expected to agree with experts more than they agree with each other)")
    cand = {
        "danger score (app)": lambda a: a["danger_p"],
        "shot xG only": lambda a: a["xg"],
        "best spot only (peak xT)": lambda a: a["peak_xt"],
        "reached the box (yes/no)": lambda a: float(a["entered_box"]),
    }
    exp_top = set(sheet.iloc[np.argsort(-expert)[:10]]["id"])
    for name, f in cand.items():
        x = np.array([f(by_id[i]) for i in sheet["id"]], float)
        rho = spearmanr(x, expert).statistic
        tau = kendalltau(x, expert).statistic
        lo, hi = boot(x, expert, lambda a, b: spearmanr(a, b).statistic)
        top = set(sheet.iloc[np.argsort(-x)[:10]]["id"])
        print(f"  {name:28s} Spearman {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]   Kendall {tau:+.2f}   top-10 overlap {len(top & exp_top)}/10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--make-sheet")
    ap.add_argument("--score")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    md = load_match(args.match, data_dir=args.data_dir)
    at = AttackFinder(MetricsEngine(md)).attacks()
    if args.make_sheet:
        make_sheet(at, args.make_sheet, args.n)
    if args.score:
        score(at, args.score)


if __name__ == "__main__":
    main()

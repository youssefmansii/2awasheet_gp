"""
eval_fields.py - Steps 4-5: score every Member-4 field on PFF's real shot moments ("oracle" detection)
Every learned part is tested leave-one-match-out.  Also prints the majority-class baseline per field.
Run (from apps/api):  python -m event_detection.eval_fields --match 3816 3828 3829 3830 3831 10514
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .field_models import GROUP, FieldModels, height_rule
from .field_table import TRUTH_COLS, shot_rows
from .train_eval import match_table
from .tracking_reader import cache_dir

FIELDS = ["shooter_id", "outcome", "keeper_id", "blocker_id", "failed_blocker_id", "gl_clearer_id", "body_movement",
          "ball_moving", "shot_height", "save_height", "save_rebound", "shot_nature"]


def oracle_table(matches, use_cache=True) -> pd.DataFrame:
    parts = []
    for m in matches:
        p = cache_dir() / f"step4_oracle_{m}.pkl"
        if use_cache and p.exists():
            parts.append(pd.read_pickle(p)); continue
        F, S = match_table(m)
        kick = []
        for r, s in S.iterrows():
            hit = F.index[(F.shot_row == r)]
            kick.append(int(F.idx[hit[0]]) if len(hit) else int(s.idx))
        T = shot_rows(m, F, kick)
        for c in TRUTH_COLS:
            T["true_" + c] = S[c].values
        T.to_pickle(p); parts.append(T)
    return pd.concat(parts, ignore_index=True)


def lomo_predict(T: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in sorted(T.match_id.unique()):
        k = T.match_id == m
        out.append(FieldModels().fit(T[~k]).predict(T[k]))
    return pd.concat(out).loc[T.index]


def _norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


def field_scores(T: pd.DataFrame, P: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two numbers per field:
       accuracy_where_pff_has_value  - identity/class right, on the shots where PFF filled the field
       agreement_all_shots           - same value incl. 'empty' (did we fill it when PFF did and vice-versa)"""
    rows = []
    for f in FIELDS:
        t = np.array([_norm(v) for v in T["true_" + f]], object)
        p = np.array([_norm(v) for v in P[f]], object)
        has = t != None  # noqa: E711
        r = {"field": f, "pff_filled": int(has.sum()),
             "accuracy_where_pff_has_value": float((p[has] == t[has]).mean()) if has.any() else np.nan,
             "agreement_all_shots": float((p == t).mean())}
        if base is not None and f in base:
            b = np.array([_norm(v) for v in base[f]], object)
            r["majority_baseline"] = float((b[has] == t[has]).mean()) if has.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def majority_baseline(T: pd.DataFrame) -> pd.DataFrame:
    """For every held-out match: predict the most common class of the other matches."""
    B = pd.DataFrame(index=T.index)
    for f in ["outcome", "body_movement", "ball_moving", "shot_height", "save_height", "save_rebound", "shot_nature"]:
        col = []
        for m in T.match_id:
            v = T.loc[T.match_id != m, "true_" + f].dropna()
            col.append(v.mode().iloc[0] if len(v) else None)
        B[f] = col
    return B


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", nargs="+", required=True)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    T = oracle_table(a.match, use_cache=not a.no_cache)
    P = lomo_predict(T)
    S = field_scores(T, P, majority_baseline(T))
    pd.set_option("display.width", 200)
    print(f"{len(T)} PFF shots, fields scored leave-one-match-out at PFF's shot moments")
    print(S.round(3).to_string(index=False))
    print("\noutcome confusion (rows PFF, columns ours):")
    print(pd.crosstab(T.true_outcome, P.outcome))
    print("outcome in 4 groups (goal / off target / saved / blocked):",
          f"{(T.true_outcome.map(GROUP) == P.outcome.map(GROUP)).mean():.1%}")
    T.join(P.add_prefix("pred_f_")).to_pickle(cache_dir() / "step45_fields_oof.pkl")
    S.to_csv(cache_dir() / "step45_field_scores.csv", index=False)

"""
Step 3 - train and cross-validate every Member-5 model.

    python -m pff5.train --tables data/tables --compact data/compact --out runs --ball raw

For each task this writes
    runs/<ball>/results/<task>.json      cross-validated metrics (match-grouped folds)
    runs/<ball>/models/<name>.txt        final LightGBM model trained on all matches
    runs/<ball>/models/<name>.fold<k>.txt  fold models (used for honest end-to-end evaluation)
    runs/<ball>/oof/<task>.parquet       out-of-fold predictions

Tasks: pressure, participants, attributes, fouls, spotting, e2e
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import features as F
from . import models as M
from .data import game_path, load_match
from .dataset import load_table, table_games

PRESSURE_LABELS = ["N", "A", "L", "P"]
CH_TYPES = ["A", "B", "D", "FIFTY", "H", "L", "S", "T"]
CH_OUTCOMES = ["D", "R", "P", "F", "O", "K", "C", "M", "N", "S", "L", "B", "I"]
DRIBBLE_TYPES = ["I", "O", "T", "K", "B"]
TACKLE_ATTEMPTS = ["G", "T", "F", "D"]
TRICK_TYPES = ["N", "F", "S", "C", "R", "D", "O", "E"]
FOUL_TYPES = ["none", "I", "A", "M"]
OFFENSES = ["N", "O", "H", "F", "T", "D", "V"]
FOUL_OUTCOMES = ["N", "Y", "F", "W", "S", "R"]
META = {"key", "gid", "ev_row", "y", "y_type", "presser", "presser_in_cands", "cand_slot", "slot", "ev", "has_gold",
        "idx", "actor", "j", "dt_ch", "_pair_h", "_pair_a", "y_culprit", "y_victim", "h_slot", "a_slot"}
PART_POSS = ["poss_known", "is_poss", "poss_same_team", "d_poss", "d_poss_min", "rank_d_poss"]
ATTR_POSS = ["poss_known", "poss_is_h", "poss_is_a", "poss_team"]


def feats(df, drop=()):
    return [c for c in df.columns if c not in META and c not in set(drop) and not c.startswith("y_")]


def codes(values, labels):
    lut = {l: i for i, l in enumerate(labels)}
    return np.array([lut.get(v, -1) if isinstance(v, str) else -1 for v in values], np.int64)


class Run:
    def __init__(self, tables, compact, out, ball, holdout=()):
        self.tables, self.compact, self.ball = tables, compact, ball
        self.holdout = [int(g) for g in holdout]
        self.root = os.path.join(out, ball)
        for d in ("results", "models", "oof"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        # held-out matches are never loaded here: not for training, validation or thresholds
        self.games = [g for g in table_games(tables, ball) if g not in self.holdout]

    def table(self, name, **kw):
        kw.setdefault("games", self.games)
        return load_table(self.tables, self.ball, name, **kw)

    def done(self, task):
        return os.path.exists(os.path.join(self.root, "results", f"{task}.json"))

    def save_result(self, task, res):
        res = dict(task=task, ball_source=self.ball, n_matches=len(self.games), held_out_matches=self.holdout, **res)
        with open(os.path.join(self.root, "results", f"{task}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        return res

    def save_model(self, name, cv, extra=None):
        path = os.path.join(self.root, "models", f"{name}.txt")
        meta = dict(features=cv["features"], best_iter=cv["best_iter"], **(extra or {}))
        if cv.get("final") is not None:
            M.save_booster(cv["final"], path, meta)
        for k, b in (cv.get("fold_models") or {}).items():
            b.save_model(os.path.join(self.root, "models", f"{name}.fold{k}.txt"))

    def load_model(self, name, fold=None):
        import lightgbm as lgb
        path = os.path.join(self.root, "models", f"{name}.txt")
        with open(path + ".json") as f:
            meta = json.load(f)
        if fold is not None:
            fp = os.path.join(self.root, "models", f"{name}.fold{fold}.txt")
            if os.path.exists(fp):
                path = fp
        return lgb.Booster(model_file=path), meta

    def save_oof(self, task, df):
        df.to_parquet(os.path.join(self.root, "oof", f"{task}.parquet"), index=False)

    def load_oof(self, task):
        return pd.read_parquet(os.path.join(self.root, "oof", f"{task}.parquet"))


def cv(X, y, groups, objective="binary", num_class=None, weight=None, params=None, train_mask=None, rounds=3000):
    """lgb_cv with optional ``train_mask`` (rows used for fitting; OOF is produced for all rows)."""
    import lightgbm as lgb
    y = np.asarray(y)
    groups = np.asarray(groups)
    folds = M.fold_of(groups)
    single = len(np.unique(groups)) == 1
    if single:
        folds = np.random.RandomState(0).randint(0, M.N_FOLDS, len(y))
    tm = np.ones(len(y), bool) if train_mask is None else np.asarray(train_mask, bool)
    p = dict(M.BASE_PARAMS if params is None else params, objective=objective)
    if objective == "multiclass":
        p.update(num_class=num_class, metric="multi_logloss")
        oof = np.full((len(y), num_class), 1.0 / num_class, np.float32)
    else:
        p["metric"] = "binary_logloss"
        oof = np.full(len(y), float(np.mean(y[tm])) if tm.any() else 0.0, np.float32)
    fl = list(X.columns)
    Xv = X.to_numpy(np.float32)
    iters, fold_models = [], {}
    for k in np.unique(folds):
        tr, va = (folds != k) & tm, folds == k
        vfit = va & tm
        if tr.sum() < 20 or vfit.sum() < 5 or len(np.unique(y[tr])) < 2:
            continue
        dtr = lgb.Dataset(Xv[tr], y[tr], weight=None if weight is None else weight[tr], feature_name=fl)
        dva = lgb.Dataset(Xv[vfit], y[vfit], weight=None if weight is None else weight[vfit], reference=dtr)
        bst = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = bst.predict(Xv[va], num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
        fold_models[int(k)] = bst
    best = int(np.median(iters)) if iters else 100
    final = None
    if tm.sum() >= 20 and len(np.unique(y[tm])) >= 2:
        final = lgb.train(p, lgb.Dataset(Xv[tm], y[tm], weight=None if weight is None else weight[tm], feature_name=fl),
                          max(int(best * 1.1), 20))
    return dict(oof=oof, final=final, best_iter=best, folds=folds, features=fl, fold_models=fold_models,
                single_match=single)


def importance(cvres, top=15):
    b = cvres.get("final")
    if b is None:
        return {}
    g = b.feature_importance("gain")
    order = np.argsort(-g)[:top]
    tot = g.sum() or 1.0
    return {cvres["features"][i]: round(float(g[i] / tot), 4) for i in order}


# =========================================================================== #
# PRESSURE
# =========================================================================== #
def task_pressure(run: Run):
    ev = run.table("press_ev")
    cand = run.table("press_cand")
    # ---- stage B: who is pressing (candidate ranking) ----
    ev_has = dict(zip(ev["key"].values, (ev["presser"].values >= 0)))
    has = cand["key"].map(ev_has).to_numpy(bool)
    fB = feats(cand)
    cvB = cv(cand[fB], cand["y"].astype(int).values, cand["gid"].values, train_mask=has)
    cand["score"] = cvB["oof"]
    run.save_model("pressure_presser", cvB)
    pickB = M.pick_report(cand["key"].values[has], cand["score"].values[has], cand["y"].values[has])
    base = M.pick_report(cand["key"].values[has], -cand["d_0"].fillna(99).values[has], cand["y"].values[has])
    n_has = int((ev["presser"] >= 0).sum())
    cover = float(ev.loc[ev["presser"] >= 0, "presser_in_cands"].mean())
    # ---- stage A: pressure type, stacked on the presser scores ----
    g = cand.sort_values(["key", "score"], ascending=[True, False]).groupby("key")["score"]
    top = pd.DataFrame(dict(ps1=g.nth(0), ps2=g.nth(1))) if False else pd.DataFrame(
        dict(ps1=g.max(), ps2=g.apply(lambda s: s.iloc[1] if len(s) > 1 else np.nan)))
    ev = ev.join(top, on="key")
    y = codes(ev["y_type"].values, PRESSURE_LABELS)
    ok = y >= 0
    fA = feats(ev)
    w = M.class_weights(np.clip(y, 0, None), 4, 0.5)
    cvA = cv(ev[fA], np.clip(y, 0, None), ev["gid"].values, "multiclass", 4, weight=w, train_mask=ok)
    run.save_model("pressure_type", cvA, dict(labels=PRESSURE_LABELS))
    proba = cvA["oof"]
    rep = M.clf_report(y, proba, PRESSURE_LABELS, mask=ok)
    rep_init = M.clf_report(y, proba, PRESSURE_LABELS, mask=ok & (ev["kind"].values == 0))
    rep_event = M.clf_report(y, proba, PRESSURE_LABELS, mask=ok & (ev["kind"].values == 1))
    pressured = M.binary_report((y > 0)[ok], 1 - proba[ok, 0])
    # ---- end to end: type + player both right ----
    pred_type = proba.argmax(1)
    best = cand.sort_values("score", ascending=False).drop_duplicates("key").set_index("key")["cand_slot"]
    pred_presser = ev["key"].map(best).fillna(-1).astype(int).values
    truth_p = ev["presser"].values
    is_p = ok & (y > 0) & (truth_p >= 0)
    e2e = dict(
        pressured_events=int(is_p.sum()),
        detected_as_pressured=float((pred_type[is_p] > 0).mean()),
        presser_correct_given_true_pressure=float((pred_presser[is_p] == truth_p[is_p]).mean()),
        type_and_presser_both_correct=float(((pred_type[is_p] == y[is_p]) & (pred_presser[is_p] == truth_p[is_p])).mean()),
    )
    run.save_oof("pressure", pd.DataFrame(dict(
        key=ev["key"].values, gid=ev["gid"].values, ev_row=ev["ev_row"].values, kind=ev["kind"].values, y=y,
        pred=pred_type, p_N=proba[:, 0], p_A=proba[:, 1], p_L=proba[:, 2], p_P=proba[:, 3],
        presser=truth_p, pred_presser=pred_presser)))
    return run.save_result("pressure", dict(
        n_events=int(ok.sum()), pressure_type=rep, pressure_type_initial_touch=rep_init,
        pressure_type_on_event=rep_event, pressured_vs_not=pressured,
        presser_pick=dict(model=pickB, baseline_nearest_opponent=base, events_with_presser=n_has,
                          presser_among_candidates=cover),
        end_to_end=e2e, top_features_type=importance(cvA), top_features_presser=importance(cvB),
        single_match_debug=cvA["single_match"]))


# =========================================================================== #
# CHALLENGE PARTICIPANTS
# =========================================================================== #
def _pairs_from_scores(part: pd.DataFrame, score):
    d = part[["gid", "ev", "team", "slot"]].copy()
    d["s"] = score
    best = d.sort_values("s", ascending=False).drop_duplicates(["gid", "ev", "team"])
    piv = best.pivot_table(index=["gid", "ev"], columns="team", values="slot", aggfunc="first")
    piv = piv.rename(columns={0.0: "pred_home", 1.0: "pred_away"}).reset_index()
    return piv


def task_participants(run: Run):
    part = run.table("ch_part")
    rows = run.table("ch_rows")
    rows["ev"] = rows.groupby("gid").cumcount()
    res = {}
    oof = rows[["gid", "ev", "ev_row", "home", "away"]].copy()
    for use_poss in (0, 1):
        drop = [] if use_poss else PART_POSS
        fl = feats(part, drop)
        c = cv(part[fl], part["y"].astype(int).values, part["gid"].values, train_mask=part["has_gold"].values)
        run.save_model(f"participants_poss{use_poss}", c)
        hg = part["has_gold"].values
        rep = M.pick_report(part["key"].values[hg], c["oof"][hg], part["y"].values[hg])
        base = M.pick_report(part["key"].values[hg], -part["rank_sum"].values[hg], part["y"].values[hg])
        pairs = _pairs_from_scores(part, c["oof"])
        mrg = rows[["gid", "ev", "home", "away"]].merge(pairs, on=["gid", "ev"], how="left")
        both = (mrg["home"] >= 0) & (mrg["away"] >= 0)
        res[f"possessor_{'given' if use_poss else 'unknown'}"] = dict(
            per_player=rep, baseline_closest_to_ball_and_opponent=base,
            pair_both_correct=float(((mrg["pred_home"] == mrg["home"]) & (mrg["pred_away"] == mrg["away"]))[both].mean()),
            pair_at_least_one_correct=float(((mrg["pred_home"] == mrg["home"]) | (mrg["pred_away"] == mrg["away"]))[both].mean()),
            top_features=importance(c))
        oof[f"pred_home_poss{use_poss}"] = mrg["pred_home"].fillna(-1).astype(int).values
        oof[f"pred_away_poss{use_poss}"] = mrg["pred_away"].fillna(-1).astype(int).values
    run.save_oof("participants", oof)
    return run.save_result("participants", dict(n_challenges=int(len(rows)), **res))


# =========================================================================== #
# CHALLENGE ATTRIBUTES
# =========================================================================== #
def build_attr_pred(run: Run, use_poss: int) -> pd.DataFrame:
    """Second pass: attribute features computed with the out-of-fold PREDICTED participants."""
    path = os.path.join(run.root, "oof", f"ch_attr_pred_poss{use_poss}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    oof = run.load_oof("participants")
    rows = run.table("ch_rows")
    parts = []
    for gid in pd.unique(rows["gid"]):          # same match order as ``rows`` - the result is row-aligned with it
        m = load_match(game_path(run.compact, gid), with_events=False, ball_source=run.ball)
        r = rows[rows["gid"] == gid]
        o = r[["ev_row"]].merge(oof[oof["gid"] == gid], on="ev_row", how="left")
        parts.append(F.challenge_attr_table(m, r["idx"].values, o[f"pred_home_poss{use_poss}"].values,
                                            o[f"pred_away_poss{use_poss}"].values,
                                            r["poss"].values if use_poss else None))
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df.to_parquet(path, index=False)
    return df


def _attr_models(run, X, rows, tag, save):
    """Train all attribute heads on feature table X (aligned with rows). Returns reports + oof dict."""
    out, oof = {}, {}
    gid = rows["gid"].values
    fl = feats(X)

    def multi(name, values, labels, mask=None, params=None, power=0.5):
        y = codes(values, labels)
        ok = (y >= 0) if mask is None else ((y >= 0) & mask)
        if ok.sum() < 40:
            out[name] = dict(n=int(ok.sum()), note="too few examples to train")
            return None
        yc = np.clip(y, 0, None)
        c = cv(X[fl], yc, gid, "multiclass", len(labels), weight=M.class_weights(yc, len(labels), power),
               train_mask=ok, params=params)
        if save:
            run.save_model(f"ch_{name}_{tag}", c, dict(labels=labels))
        out[name] = M.clf_report(y, c["oof"], labels, mask=ok)
        out[name]["top_features"] = importance(c, 10)
        oof[name] = c["oof"]
        return c

    def binary(name, values, mask=None, params=None):
        v = np.asarray(values, float)
        ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
        if ok.sum() < 40:
            out[name] = dict(n=int(ok.sum()), note="too few examples to train")
            return None
        y = np.where(ok, v, 0).astype(int)
        c = cv(X[fl], y, gid, train_mask=ok, params=params)
        if save:
            run.save_model(f"ch_{name}_{tag}", c)
        r = M.binary_report(y[ok], c["oof"][ok])
        r["accuracy_at_0.5"] = float(((c["oof"][ok] > 0.5) == (y[ok] == 1)).mean())
        r["majority_baseline_accuracy"] = float(max(y[ok].mean(), 1 - y[ok].mean()))
        r["top_features"] = importance(c, 10)
        out[name] = r
        oof[name] = c["oof"]
        return c

    multi("type", rows["ctype"].values, CH_TYPES)
    binary("home_has_ball", rows["home_has_ball"].values)
    binary("home_wins", rows["home_wins"].values)
    multi("outcome", rows["outcome"].values, CH_OUTCOMES)
    is_d = (rows["ctype"] == "D").values
    # low-signal sub-labels: no class re-weighting, so the model falls back to the common class when unsure
    multi("dribble_type", rows["dribble_type"].values, DRIBBLE_TYPES, mask=is_d, params=M.SMALL_PARAMS, power=0.0)
    multi("tackle_attempt", rows["tackle_attempt"].values, TACKLE_ATTEMPTS, mask=is_d, params=M.SMALL_PARAMS, power=0.0)
    multi("trick_type", rows["trick_type"].values, TRICK_TYPES, mask=is_d, params=M.SMALL_PARAMS, power=0.0)
    ft = np.where(pd.isna(rows["foul_type"].values), "none", rows["foul_type"].values)
    multi("foul_type", ft, FOUL_TYPES, power=0.25)
    binary("foul_given", (ft == "I").astype(float))
    is_foul = ft == "I"
    card = np.where(is_foul, np.isin(rows["foul_outcome"].values, ["Y", "S", "R"]).astype(float), np.nan)
    binary("card_given_foul", card, params=M.SMALL_PARAMS)
    # culprit side: is the HOME participant the one who fouled?
    cul = rows["culprit"].values
    cul_home = np.where(is_foul & (cul >= 0), (cul == rows["home"].values).astype(float), np.nan)
    cul_in_pair = float(((cul == rows["home"].values) | (cul == rows["away"].values))[is_foul & (cul >= 0)].mean())
    binary("culprit_is_home", cul_home, params=M.SMALL_PARAMS)
    if isinstance(out.get("culprit_is_home"), dict):
        out["culprit_is_home"]["culprit_is_one_of_the_two_participants"] = cul_in_pair
    return out, oof


def _derived_ids(rows, oof, pred_home, pred_away):
    """Turn head outputs into PFF player-id fields and score them against the labels."""
    n = len(rows)
    ph, pa = np.asarray(pred_home), np.asarray(pred_away)
    typ = np.array(CH_TYPES)[oof["type"].argmax(1)] if "type" in oof else np.array(["T"] * n)
    duel_pred = np.isin(typ, F.DUEL_TYPES)
    duel_true = np.isin(rows["ctype"].values, F.DUEL_TYPES)
    res = {}
    hb = oof.get("home_has_ball")
    if hb is not None:
        truth = rows["home_has_ball"].values
        ok = np.isfinite(truth) & (rows["home"].values >= 0) & (rows["away"].values >= 0)
        carrier_pred = np.where(hb > 0.5, ph, pa)
        chall_pred = np.where(hb > 0.5, pa, ph)
        carrier_true = np.where(truth == 1, rows["home"].values, rows["away"].values)
        chall_true = np.where(truth == 1, rows["away"].values, rows["home"].values)
        res["carrier_or_dribbler_id_accuracy"] = float((carrier_pred == carrier_true)[ok].mean())
        res["challenger_id_accuracy"] = float((chall_pred == chall_true)[ok].mean())
    ok = duel_true & (rows["home"].values >= 0) & (rows["away"].values >= 0)
    if ok.any():
        res["home_duel_player_id_accuracy"] = float((ph == rows["home"].values)[ok].mean())
        res["away_duel_player_id_accuracy"] = float((pa == rows["away"].values)[ok].mean())
    hw = oof.get("home_wins")
    if hw is not None:
        truth = rows["home_wins"].values
        ok = np.isfinite(truth) & (rows["home"].values >= 0) & (rows["away"].values >= 0)
        win_pred = np.where(hw > 0.5, ph, pa)
        win_true = np.where(truth == 1, rows["home"].values, rows["away"].values)
        res["winner_id_accuracy"] = float((win_pred == win_true)[ok].mean())
    res["duel_vs_tackle_structure_accuracy"] = float((duel_pred == duel_true).mean())
    return res


def task_attributes(run: Run):
    rows = run.table("ch_rows")
    gold = run.table("ch_attr")
    oofp = rows[["gid", "ev_row"]].merge(run.load_oof("participants"), on=["gid", "ev_row"], how="left")
    assert len(oofp) == len(rows)
    res = {}
    for use_poss in (0, 1):
        tagp = f"poss{use_poss}"
        Xg = gold.drop(columns=[] if use_poss else ATTR_POSS)
        r_gold, o_gold = _attr_models(run, Xg, rows, f"{tagp}", save=True)
        ids_gold = _derived_ids(rows, o_gold, rows["home"].values, rows["away"].values)
        # end-to-end: features from predicted participants, scored with models trained on gold participants
        Xp = build_attr_pred(run, use_poss)
        assert (Xp["gid"].values == rows["gid"].values).all(), "feature rows are not aligned with the challenge rows"
        Xp = Xp.drop(columns=[] if use_poss else ATTR_POSS)
        r_pred, o_pred = _attr_models(run, Xp, rows, f"{tagp}_predpair", save=False)
        ids_pred = _derived_ids(rows, o_pred, oofp[f"pred_home_{tagp}"].values, oofp[f"pred_away_{tagp}"].values)
        res[f"possessor_{'given' if use_poss else 'unknown'}"] = dict(
            with_true_participants=dict(heads=r_gold, player_id_fields=ids_gold),
            with_predicted_participants=dict(heads=r_pred, player_id_fields=ids_pred))
        if use_poss == 0:
            keep = pd.DataFrame(dict(gid=rows["gid"].values, ev_row=rows["ev_row"].values))
            for k, v in o_gold.items():
                if v.ndim == 1:
                    keep[f"gold_{k}"] = v
            run.save_oof("attributes", keep)
    return run.save_result("attributes", dict(n_challenges=int(len(rows)), **res))


# =========================================================================== #
# FOULS
# =========================================================================== #
def task_fouls(run: Run):
    rows = run.table("foul_rows")
    X = run.table("foul_ev")
    gid = rows["gid"].values
    fl = feats(X)
    res = {}
    ft = rows["foul_type"].values
    any_foul = (~pd.isna(ft)).astype(int)
    given = np.array([v == "I" for v in ft]).astype(int)
    is_ch = (rows["pe_type"] == "CH").values
    c = cv(X[fl], given, gid)
    run.save_model("foul_given", c)
    res["foul_given_any_event"] = dict(all_events=M.binary_report(given, c["oof"]),
                                       challenges_only=M.binary_report(given[is_ch], c["oof"][is_ch]),
                                       non_challenge_events=M.binary_report(given[~is_ch], c["oof"][~is_ch]),
                                       top_features=importance(c))
    p_foul = c["oof"]
    c2 = cv(X[fl], any_foul, gid)
    run.save_model("foul_any", c2)
    res["any_infringement_incl_advantage_and_missed"] = M.binary_report(any_foul, c2["oof"])
    off = np.array([v == "O" for v in rows["offense"].values]).astype(int)
    not_ch = ~is_ch
    c3 = cv(X[fl], off, gid, train_mask=not_ch)
    run.save_model("offside", c3)
    res["offside"] = dict(non_challenge_events=M.binary_report(off[not_ch], c3["oof"][not_ch]), top_features=importance(c3))
    # attributes of an infringement, given that there is one
    isf = any_foul == 1

    def multi(name, values, labels, params=M.SMALL_PARAMS):
        y = codes(values, labels)
        ok = (y >= 0) & isf
        yc = np.clip(y, 0, None)
        cc = cv(X[fl], yc, gid, "multiclass", len(labels), train_mask=ok, params=params)
        run.save_model(f"foul_{name}", cc, dict(labels=labels))
        res[name] = M.clf_report(y, cc["oof"], labels, mask=ok)
        res[name]["top_features"] = importance(cc, 8)

    multi("foul_type_I_A_M", ft, FOUL_TYPES[1:])
    multi("offense_type", rows["offense"].values, OFFENSES)
    multi("foul_outcome_cards", rows["foul_outcome"].values, FOUL_OUTCOMES)
    for name, col in (("var_used", "var"),):
        v = pd.Series(rows[col].values).map({True: 1.0, False: 0.0}).to_numpy(float)
        ok = np.isfinite(v) & isf
        if ok.sum() > 50 and 0 < v[ok].sum() < ok.sum():
            cc = cv(X[fl], np.where(ok, v, 0).astype(int), gid, train_mask=ok, params=M.SMALL_PARAMS)
            res[name] = M.binary_report(v[ok].astype(int), cc["oof"][ok])
    # culprit / victim ranking
    cand = run.table("foul_cand")
    if len(cand):
        fc = feats(cand)
        for who in ("culprit", "victim"):
            y = cand[f"y_{who}"].astype(int).values
            has = cand.groupby("key")[f"y_{who}"].transform("max").to_numpy(bool)
            cc = cv(cand[fc], y, cand["gid"].values, train_mask=has, params=M.SMALL_PARAMS)
            run.save_model(f"foul_{who}", cc)
            rep = M.pick_report(cand["key"].values[has], cc["oof"][has], y[has])
            rep["answer_among_candidates"] = float(cand.groupby("key")[f"y_{who}"].max().mean())
            rep["baseline_nearest_player"] = M.pick_report(cand["key"].values[has], -cand["rank"].values[has], y[has])
            res[f"{who}_pick"] = rep
    run.save_oof("fouls", pd.DataFrame(dict(gid=gid, ev_row=rows["ev_row"].values, y_given=given, p_given=p_foul,
                                            y_off=off, p_off=c3["oof"])))
    return run.save_result("fouls", dict(n_events=int(len(rows)), n_infringements=int(isf.sum()), **res))


# =========================================================================== #
# SPOTTING
# =========================================================================== #
POS_WIN, IGNORE_WIN, NEG_KEEP = 0.5, 1.0, 0.2


def _spot_sample(df: pd.DataFrame) -> pd.DataFrame:
    """All frames near a challenge + a random 20% of the rest (deterministic per match)."""
    a = df["dt_ch"].abs().to_numpy()
    rng = np.random.RandomState(int(df["gid"].iloc[0]) % 100000)
    keep = (a <= 2.0) | (rng.rand(len(df)) < NEG_KEEP)
    return df[keep]


def peak_pick(score: np.ndarray, j: np.ndarray, period: np.ndarray, fps: float, min_sep_s=1.2, floor=0.02):
    """Local maxima of a smoothed per-frame score -> (frame index, peak score)."""
    s = np.convolve(score, np.ones(3) / 3, mode="same")
    order = np.argsort(-s)
    taken = np.zeros(len(s), bool)
    sep = min_sep_s * fps
    out_j, out_s = [], []
    for i in order:
        if s[i] < floor:
            break
        if taken[i]:
            continue
        out_j.append(j[i])
        out_s.append(s[i])
        lo = np.searchsorted(j, j[i] - sep)
        hi = np.searchsorted(j, j[i] + sep, side="right")
        taken[lo:hi] = True
    return np.array(out_j, np.int64), np.array(out_s, np.float32)


def match_detections(det_j, det_s, gt_j, tol_frames):
    """Greedy one-to-one matching in score order. Returns bool array 'detection is a true positive'."""
    used = np.zeros(len(gt_j), bool)
    tp = np.zeros(len(det_j), bool)
    which = np.full(len(det_j), -1, np.int64)
    if len(gt_j) == 0:
        return tp, which
    for i in np.argsort(-det_s):
        d = np.abs(gt_j - det_j[i]).astype(float)
        d[used] = np.inf
        k = int(d.argmin())
        if d[k] <= tol_frames:
            used[k] = True
            tp[i] = True
            which[i] = k
    return tp, which


def spotting_metrics(dets: pd.DataFrame, n_gt: int, tol: str) -> dict:
    """AP and best F1 from pooled detections (columns: score, tp_<tol>)."""
    d = dets.sort_values("score", ascending=False)
    tp = d[f"tp_{tol}"].to_numpy(bool)
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(d) + 1)
    rec = ctp / max(n_gt, 1)
    ap = float((prec * tp).sum() / max(n_gt, 1))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(f1.argmax()) if len(f1) else 0
    return dict(average_precision=ap, best_f1=float(f1[k]) if len(f1) else 0.0,
                precision=float(prec[k]) if len(f1) else 0.0, recall=float(rec[k]) if len(f1) else 0.0,
                threshold=float(d["score"].iloc[k]) if len(d) else 1.0, max_recall=float(rec[-1]) if len(rec) else 0.0)


def task_spotting(run: Run):
    import lightgbm as lgb
    samp = run.table("frames", frame_filter=_spot_sample)
    a = samp["dt_ch"].abs().to_numpy()
    y = (a <= POS_WIN).astype(int)
    train_mask = (a <= POS_WIN) | (a > IGNORE_WIN)
    w = np.where(a > 2.0, 1.0 / NEG_KEEP, 1.0).astype(np.float32)
    fl = feats(samp)
    params = dict(M.BASE_PARAMS, num_leaves=127, min_data_in_leaf=100)
    c = cv(samp[fl], y, samp["gid"].values, weight=w, train_mask=train_mask, params=params, rounds=2000)
    run.save_model("spotting", c, dict(pos_window_s=POS_WIN))
    frame_level = M.binary_report(y[train_mask], c["oof"][train_mask])
    del samp
    # full-match inference with the fold model that never saw the match
    rows = run.table("ch_rows", columns=["gid", "idx"])
    folds = dict(zip(run.games, M.fold_of(np.array(run.games))))
    dets, n_gt = [], 0
    base_dets = []
    for gid in run.games:
        df = run.table("frames", games=[gid])
        bst = c["fold_models"].get(int(folds[gid]), c["final"]) if not c["single_match"] else c["final"]
        score = bst.predict(df[fl].to_numpy(np.float32))
        fps = 29.97
        gt = np.sort(rows.loc[rows["gid"] == gid, "idx"].to_numpy())
        n_gt += len(gt)
        for name, sc, store in (("model", score, dets), ("base", 1.0 / (1.0 + df["pair_d_rmin"].fillna(9).to_numpy()
                                                                         + 0.5 * df["b_second_team"].fillna(9).to_numpy()), base_dets)):
            dj, ds = peak_pick(sc, df["j"].to_numpy(), df["period"].to_numpy(), fps, floor=0.02 if name == "model" else 0.15)
            d = pd.DataFrame(dict(gid=gid, j=dj, score=ds))
            for tol_s in (1, 2):
                d[f"tp_{tol_s}s"], wh = match_detections(dj, ds, gt, tol_s * fps)
                d[f"gt_{tol_s}s"] = np.where(wh >= 0, gt[np.clip(wh, 0, max(len(gt) - 1, 0))] if len(gt) else -1, -1)
            store.append(d)
    dets = pd.concat(dets, ignore_index=True)
    base_dets = pd.concat(base_dets, ignore_index=True)
    run.save_oof("spotting", dets)
    res = dict(n_challenges=int(n_gt), frame_level=frame_level,
               tolerance_1s=spotting_metrics(dets, n_gt, "1s"), tolerance_2s=spotting_metrics(dets, n_gt, "2s"),
               baseline_proximity_rule=dict(tolerance_1s=spotting_metrics(base_dets, n_gt, "1s"),
                                            tolerance_2s=spotting_metrics(base_dets, n_gt, "2s")),
               top_features=importance(c), single_match_debug=c["single_match"])
    with open(os.path.join(run.root, "models", "spotting_threshold.json"), "w") as f:
        json.dump(dict(threshold=res["tolerance_1s"]["threshold"], tolerance_s=1.0), f)
    return run.save_result("spotting", res)


# =========================================================================== #
# END TO END: tracking only -> challenge events with players and attributes
# =========================================================================== #
def task_e2e(run: Run):
    """Spot challenges, then identify participants and type on the DETECTED frames,
    always using the fold models that never saw the match."""
    sp = json.load(open(os.path.join(run.root, "results", "spotting.json")))
    thr = sp["tolerance_1s"]["threshold"]
    dets = run.load_oof("spotting")
    dets = dets[dets["score"] >= thr]
    rows = run.table("ch_rows")
    folds = dict(zip(run.games, M.fold_of(np.array(run.games))))
    single = len(run.games) == 1
    agg = dict(n_det=0, n_tp=0, both=0, one=0, type_ok=0, win_ok=0, win_n=0, struct_ok=0, foul_tp=0, foul_fp=0, foul_fn=0,
               full=0)
    n_gt = len(rows)
    for gid in run.games:
        d = dets[dets["gid"] == gid]
        if not len(d):
            continue
        k = None if single else int(folds[gid])
        m = load_match(game_path(run.compact, gid), with_events=False, ball_source=run.ball)
        r = rows[rows["gid"] == gid].reset_index(drop=True)
        idx = d["j"].to_numpy()
        part = F.challenge_participant_table(m, idx)
        bp, mp = run.load_model("participants_poss0", k)
        hs, as_ = F.pick_participants(part, bp.predict(part[mp["features"]].to_numpy(np.float32)), len(idx))
        attr = F.challenge_attr_table(m, idx, hs, as_)
        bt, mt = run.load_model("ch_type_poss0", k)
        ptype = np.array(CH_TYPES)[bt.predict(attr[mt["features"]].to_numpy(np.float32)).argmax(1)]
        bw, mw = run.load_model("ch_home_wins_poss0", k)
        pwin = bw.predict(attr[mw["features"]].to_numpy(np.float32)) > 0.5
        bf, mf = run.load_model("ch_foul_given_poss0", k)
        pf = bf.predict(attr[mf["features"]].to_numpy(np.float32)) > 0.5
        tp = d["tp_1s"].to_numpy(bool)
        gt_idx = d["gt_1s"].to_numpy()
        lut = {int(v): i for i, v in enumerate(r["idx"].values)}
        gi = np.array([lut.get(int(g), -1) for g in gt_idx])
        ok = tp & (gi >= 0)
        g = r.iloc[gi[ok]]
        agg["n_det"] += len(d)
        agg["n_tp"] += int(ok.sum())
        hh, aa = hs[ok] == g["home"].values, as_[ok] == g["away"].values
        agg["both"] += int((hh & aa).sum())
        agg["one"] += int((hh | aa).sum())
        agg["type_ok"] += int((ptype[ok] == g["ctype"].values).sum())
        agg["full"] += int((hh & aa & (ptype[ok] == g["ctype"].values)).sum())
        agg["struct_ok"] += int((np.isin(ptype[ok], F.DUEL_TYPES) == np.isin(g["ctype"].values, F.DUEL_TYPES)).sum())
        wv = np.isfinite(g["home_wins"].values.astype(float))
        agg["win_ok"] += int((pwin[ok][wv] == (g["home_wins"].values[wv] == 1)).sum())
        agg["win_n"] += int(wv.sum())
        gf = (g["foul_type"].values == "I")
        agg["foul_tp"] += int((pf[ok] & gf).sum())
        agg["foul_fp"] += int((pf[ok] & ~gf).sum()) + int(pf[~ok].sum())
        agg["foul_fn"] += int((~pf[ok] & gf).sum())
    ntp = max(agg["n_tp"], 1)
    total_fouls = int((rows["foul_type"] == "I").sum())
    res = dict(
        detection_threshold=thr, challenges_in_data=int(n_gt), detections=agg["n_det"], true_positives_within_1s=agg["n_tp"],
        detection_precision=agg["n_tp"] / max(agg["n_det"], 1), detection_recall=agg["n_tp"] / max(n_gt, 1),
        on_true_positives=dict(
            both_participants_correct=agg["both"] / ntp, at_least_one_participant_correct=agg["one"] / ntp,
            challenge_type_correct=agg["type_ok"] / ntp, duel_vs_tackle_structure_correct=agg["struct_ok"] / ntp,
            winning_team_correct=agg["win_ok"] / max(agg["win_n"], 1)),
        fully_correct_challenge_events=dict(
            note="detected within 1 s AND both players AND type correct, as a share of all real challenges",
            recall=agg["full"] / max(n_gt, 1)),
        challenge_fouls_from_tracking_only=dict(
            fouls_in_data=total_fouls, precision=agg["foul_tp"] / max(agg["foul_tp"] + agg["foul_fp"], 1),
            recall=agg["foul_tp"] / max(total_fouls, 1)),
    )
    return run.save_result("e2e", res)


TASKS = dict(pressure=task_pressure, participants=task_participants, attributes=task_attributes, fouls=task_fouls,
             spotting=task_spotting, e2e=task_e2e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--compact", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ball", default="raw", choices=["raw", "smoothed"])
    ap.add_argument("--tasks", default="pressure,participants,attributes,fouls,spotting,e2e")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--holdout", default="", help="comma separated match ids kept out of everything (final test)")
    a = ap.parse_args(argv)
    run = Run(a.tables, a.compact, a.out, a.ball, [g for g in a.holdout.split(",") if g])
    print(f"{len(run.games)} matches for training/validation, held out for the final test: {run.holdout}, ball={a.ball}", flush=True)
    for t in a.tasks.split(","):
        if run.done(t) and not a.overwrite:
            print(f"[{t}] already done - skipping", flush=True)
            continue
        t0 = time.time()
        TASKS[t](run)
        print(f"[{t}] finished in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

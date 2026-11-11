"""
field_models.py - turn the per-shot features (field_table.py) into PFF field values
===================================================================================
Small learned models (random forests) where there are enough examples, football rules where there
are too few.  Every learned model is trained on training matches only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

GROUP = {"G": "G", "O": "O", "S": "SAVE", "F": "SAVE", "B": "BLOCK", "C": "BLOCK", "L": "BLOCK"}
OUTCOME_FEATS = ["kick_dist", "k_y_line", "k_z_line", "k_on_target", "has_touch", "t1_dt", "t1_gk_d", "t1_def_d",
                 "t1_att_d", "t1_x", "t1_absy", "t1_z", "t1_dist_from_kick", "t1_y_line", "t1_z_line", "t1_on_target",
                 "t1_speed_drop", "t1_turn", "t1_behind_gk", "min_gk_d", "min_def_d", "line_seen", "line_y", "line_z",
                 "line_before_touch", "max_x_3s", "last_x_3s", "last_absy_3s", "lost_share_3s", "rs_dt", "rs_kickoff",
                 "rs_goalkick", "rs_corner", "rs_other", "rs_none", "ko_dt", "ball_speed_after_max", "ball_z_max_05",
                 "blocker_d", "fb1_d"]
FB_FEATS = ["fb1_d", "fb2_d", "fb_is_blocker", "blocker_d", "min_def_d", "kick_dist", "k_y_line", "t1_def_d",
            "t1_dt", "has_touch", "k_on_target", "ball_speed_after_max"]
MOVE_FEATS = ["sh_speed", "sh_v_to_goal", "sh_v_lat"]
HEIGHT_FEATS = ["line_z", "line_before_touch", "k_z_line", "t1_z_line", "ball_z_max_05", "ball_z_kick", "t1_z_med",
                "line_seen", "k_y_line"]
SAVEH_FEATS = ["t1_z_med", "t1_z", "line_z", "k_z_line", "t1_z_line", "ball_z_max_05", "ball_z_kick"]
NATURE_FEATS = ["ball_speed_after_max", "kick_dist", "ball_z_kick", "ball_speed_before", "k_y_line", "sh_speed"]


def rf(seed=20261110, leaf=2, balanced=False):
    """Random forest; no class re-weighting by default (re-weighting lowered held-out accuracy)."""
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf,
                                                class_weight="balanced" if balanced else None,
                                                random_state=seed, n_jobs=2))


def _X(T, cols):
    return T[cols].astype(float).values


def refine_outcome(group: str, r) -> str:
    """Sub-class inside a group from geometry (too few examples to learn S/F, B/C, L)."""
    # PFF: saves are on target 26:2, blocks on target 34:4, so the rare class needs strong evidence
    if group == "SAVE":
        far = (r.t1_y_line > 6.0) if not np.isnan(r.t1_y_line) else (r.k_y_line > 6.0 if not np.isnan(r.k_y_line) else False)
        return "F" if far else "S"
    if group == "BLOCK":
        if r.t1_behind_gk == 1.0 and r.t1_x > 50 and (np.isnan(r.t1_gk_d) or r.t1_gk_d > 2):
            return "L"
        return "C" if (not np.isnan(r.k_y_line) and r.k_y_line > 12.0) else "B"
    return group


def height_rule(z: float, rolling: bool = False) -> str | None:
    if z is None or np.isnan(z):
        return None
    if rolling or z < 0.2:
        return "G"
    for lim, lab in ((0.81, "BOTTOMTHIRD"), (1.63, "MIDDLETHIRD"), (2.13, "TOPTHIRD"), (2.44, "U"), (3.35, "N")):
        if z < lim:
            return lab
    return "O"


def rebound_zone(x, y, z, L=52.5, left_is_positive_y=True) -> str | None:
    if x is None or np.isnan(x):
        return None
    yy = y if left_is_positive_y else -y
    if x >= L:
        if z > 2.44:
            return "OC"
        return "GL" if yy > 0 else "GR"
    if x > L - 5.5 and abs(y) < 9.16:
        return "M6" if abs(yy) < 3.05 else ("L6" if yy > 0 else "R6")
    if x > L - 16.5 and abs(y) < 20.16:
        return "MA" if abs(yy) < 6.72 else ("LA" if yy > 0 else "RA")
    return "MO" if abs(yy) < 6.72 else ("LO" if yy > 0 else "RO")


class FieldModels:
    """All learned field models, fitted on a training table (one row per PFF shot)."""

    def fit(self, T: pd.DataFrame):
        self.outcome = rf().fit(_X(T, OUTCOME_FEATS), T.true_outcome.map(GROUP).values)
        self.fb = rf(leaf=3).fit(_X(T, FB_FEATS), T.true_failed_blocker_id.notna().values)
        self.move = rf(leaf=3).fit(_X(T, MOVE_FEATS), T.true_body_movement.values)
        h = T.true_shot_height.notna().values
        self.height = rf(leaf=3).fit(_X(T[h], HEIGHT_FEATS), T.true_shot_height[h].values)
        self.nature = rf(leaf=5).fit(_X(T, NATURE_FEATS), T.true_shot_nature.values)
        sh = T.true_save_height.notna().values
        self.saveh = rf(leaf=3).fit(_X(T[sh], SAVEH_FEATS), T.true_save_height[sh].values)
        # rebound zone: only ~1 example per match -> the rule did worse than the most common zone,
        # so the most common training zone is used (documented as a limit)
        reb = T.true_save_rebound.dropna()
        self.reb_major = reb.mode().iloc[0] if len(reb) else None
        self.ballmoving_thr = 1.0
        # rebound orientation learned from the training shots that have a rebound zone
        r = T[T.true_save_rebound.notna()]
        hits = {flag: sum(rebound_zone(a, b, c, left_is_positive_y=flag) == t
                          for a, b, c, t in zip(r.reb_x, r.reb_y, r.reb_z, r.true_save_rebound)) for flag in (True, False)}
        self.left_pos = hits[True] >= hits[False]
        return self

    def predict(self, T: pd.DataFrame) -> pd.DataFrame:
        P = pd.DataFrame(index=T.index)
        P["shooter_id"] = T.pred_shooter.values
        g = self.outcome.predict(_X(T, OUTCOME_FEATS))
        P["outcome"] = [refine_outcome(gg, r) for gg, (_, r) in zip(g, T.iterrows())]
        P["keeper_id"] = np.where(P.outcome.isin(["S", "F", "G"]), T.pred_keeper_any.values, None)
        P["blocker_id"] = np.where(P.outcome == "B", T.pred_blocker_any.values, None)
        P["gl_clearer_id"] = np.where(P.outcome == "L", T.pred_blocker_any.values, None)
        has_fb = self.fb.predict(_X(T, FB_FEATS)).astype(bool)
        fb = np.where((T.fb_is_blocker.values == 1) & P.outcome.isin(["B", "C"]).values, T.fb2.values, T.fb1.values)
        P["failed_blocker_id"] = np.where(has_fb, fb, None)
        P["body_movement"] = self.move.predict(_X(T, MOVE_FEATS))
        P["ball_moving"] = ~(T.ball_speed_before.fillna(5).values < self.ballmoving_thr)
        P["shot_height"] = self.height.predict(_X(T, HEIGHT_FEATS))
        P["save_height"] = np.where(P.outcome.isin(["S", "F"]), self.saveh.predict(_X(T, SAVEH_FEATS)), None)
        P["save_rebound"] = np.where(P.outcome.isin(["S", "F"]), self.reb_major, None)
        P["shot_nature"] = self.nature.predict(_X(T, NATURE_FEATS))
        return P

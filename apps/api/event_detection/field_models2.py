"""
field_models2.py - version 2 of the Member-4 field models (all learned on training matches only)
  shooter / blocker / failed blocker : learned candidate ranking (player_rank.py)
  outcome                            : gradient boosting (7 PFF types directly) on what happens after the kick,
                                       kick + context features and continuous kick-off / restart evidence
  save rebound zone                  : random forest on where the ball is 0.7 s after the save
  ball moving                        : gradient boosting
  body movement, shot height, save height, shot nature : as in version 1 (the bigger models were not better)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H

from .ball_physics import clean_ball
from .context import CTX_FEATURES
from .features import FEATURES as KICK_FEATURES
from .field_models import GROUP, OUTCOME_FEATS, FieldModels, refine_outcome, rf
from .player_rank import PlayerRankers, candidates, label
from .restart_feats import RS_FEATS, restart_features
from .tracking_reader import load_tracking

EXTRA = ["k_f_" + c for c in KICK_FEATURES + CTX_FEATURES]
OUT2 = OUTCOME_FEATS + EXTRA + RS_FEATS
REB = ["reb_x", "reb_y", "reb_z", "t1_x", "t1_absy", "t1_z_med"]
BM = ["ball_speed_before", "ball_speed_before_min"] + EXTRA
FB_THR = 0.30
_hgb = lambda: H(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=20261110)
_X = lambda T, c: T[c].astype(float).values


def enrich(mid: str, F: pd.DataFrame, T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add kick + context features to the per-shot table and build the player candidates (tracking only)."""
    Fi = F.drop_duplicates("idx").set_index("idx")
    cols = [c for c in KICK_FEATURES + CTX_FEATURES if c in Fi.columns]
    T = T.join(Fi[cols].add_prefix("k_f_"), on="kick_idx")
    for c in EXTRA:
        if c not in T.columns:
            T[c] = np.nan
    tr = load_tracking(mid); ball = clean_ball(tr)
    R = restart_features(tr, ball, T.kick_idx.tolist()); R.index = T.index
    T = pd.concat([T, R], axis=1)
    C = candidates(tr, ball, F.idx.values, T.kick_idx.tolist())
    return T, C


class FieldModels2:
    def fit(self, T: pd.DataFrame, C: pd.DataFrame):
        self.base = FieldModels().fit(T)
        self.rank = PlayerRankers().fit(label(C, T))
        self.outcome = _hgb().fit(_X(T, OUT2), T.true_outcome.astype(str).values)   # all 7 PFF types directly
        r = T.true_save_rebound.notna().values
        self.reb = rf(leaf=3).fit(_X(T[r], REB), T.true_save_rebound[r].astype(str).values)
        self.bm = _hgb().fit(_X(T, BM), T.true_ball_moving.astype(bool).values)
        return self

    def predict(self, T: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
        P = self.base.predict(T)
        pk = PlayerRankers.pick(self.rank.score(C)).drop_duplicates("kick_idx").set_index("kick_idx").reindex(T.kick_idx.values)
        P["shooter_id"] = pk.rk_shooter.values
        P["outcome"] = self.outcome.predict(_X(T, OUT2))
        o = P.outcome
        P["keeper_id"] = np.where(o.isin(["S", "F", "G"]), T.pred_keeper_any.values, None)
        P["blocker_id"] = np.where(o == "B", pk.rk_blocker.values, None)
        P["gl_clearer_id"] = np.where(o == "L", pk.rk_blocker.values, None)
        blocked = o.isin(["B", "C", "L"]).values
        same = blocked & (pk.rk_fb1.values == pk.rk_blocker.values)
        fb = np.where(same, pk.rk_fb2.values, pk.rk_fb1.values)
        fbp = np.where(same, pk.rk_fb2_p.values, pk.rk_fb1_p.values)
        P["failed_blocker_id"] = np.where(fbp > FB_THR, fb, None)
        P["save_rebound"] = np.where(o.isin(["S", "F"]), self.reb.predict(_X(T, REB)), None)
        P["ball_moving"] = self.bm.predict(_X(T, BM)).astype(bool)
        return P

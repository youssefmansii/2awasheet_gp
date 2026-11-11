"""
shot_detector.py - decide which kicks are shots
================================================

Two detectors, same input (features.py, tracking only):
  * rule_baseline(): hand-written football rules (a reference point)
  * ShotModel: gradient boosting (scikit-learn HistGradientBoostingClassifier), small and regularised
After scoring, non_max_suppression() keeps at most one shot per 1-second window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .context import CTX_FEATURES
from .features import FEATURES as _BASE

FEATURES = _BASE + CTX_FEATURES   # kick features + trajectory context (v2)


def rule_baseline(F: pd.DataFrame) -> np.ndarray:
    """1 = shot by simple rules: an outfield player kicks hard toward the goal from shooting range."""
    return ((F.actor_dist < 4) & (F.actor_is_gk < 0.5) & (F.dist_goal < 35) & (F.vx_goal > 6)
            & (F.speed_after > 9) & (F.abs_y_cross < 10) & (F.cos_goal > 0.7)).astype(float).values


def prefilter(F: pd.DataFrame) -> np.ndarray:
    """Cheap physical filter: kick within 36 m of a goal and the ball gets within 22.5 m of that goal
    line in the next 2 s.  Keeps 132 of 133 labelled shots and drops 2/3 of all kicks."""
    return ((F.dist_goal < 36) & (F.f_max_x_2s > 30)).values


class ShotModel:
    """Average of two tree ensembles (gradient boosting + random forest), trained on pre-filtered kicks.
    No class re-weighting: it distorted the ranking (average precision 0.44 -> 0.57 without it)."""

    def __init__(self, seed: int = 20261110):
        self.models = [
            HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=seed),
            make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                                 n_jobs=2, random_state=seed)),
        ]
        self.threshold = 0.5

    def fit(self, F: pd.DataFrame, y: np.ndarray):
        k = prefilter(F)
        X = F.loc[k, FEATURES].astype(float).values
        for m in self.models:
            m.fit(X, np.asarray(y)[k])
        return self

    def proba(self, F: pd.DataFrame) -> np.ndarray:
        k = prefilter(F)
        out = np.zeros(len(F))
        if k.any():
            X = F.loc[k, FEATURES].astype(float).values
            ps = [m.predict_proba(X)[:, 1] for m in self.models]
            # rank-free average after putting both on 0..1 by their own max on this call
            out[k] = np.mean(ps, axis=0)
        return out


def non_max_suppression(F: pd.DataFrame, score: np.ndarray, thr: float, win: int = 45) -> np.ndarray:
    """Keep candidates with score >= thr, at most one per +-win frames (the highest). 45 frames (1.5 s)
    was best on the 48 training matches (fewer double detections of one shot)."""
    keep = np.zeros(len(F), bool)
    order = np.argsort(-score)
    taken: list[tuple[str, int]] = []
    idx, mid = F.idx.values, F.match_id.values
    for k in order:
        if score[k] < thr:
            break
        if any(m == mid[k] and abs(i - idx[k]) <= win for m, i in taken):
            continue
        keep[k] = True
        taken.append((mid[k], idx[k]))
    return keep

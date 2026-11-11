"""fields_cv.py - 6-fold-by-match CV of every Member-4 field at PFF's shot moments (48 training matches).
Compares the first version (nearest-player rules, small feature set) with the improved models."""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as H, RandomForestClassifier as RF
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from .context import CTX_FEATURES, CTX_VERSION
from .features import FEAT_VERSION, FEATURES
from .field_models import GROUP, OUTCOME_FEATS, MOVE_FEATS, HEIGHT_FEATS, SAVEH_FEATS, NATURE_FEATS, refine_outcome
from .player_rank import PlayerRankers, label
from .tracking_reader import cache_dir

D = cache_dir(); folds = json.load(open(D / "cv48_folds.json")); TRAIN = sum(folds, [])


def load_all(matches=TRAIN, kind="oracle"):
    Ts, Cs = [], []
    for m in matches:
        T = pd.read_pickle(D / f"step4_{kind}_{m}.pkl")
        F = pd.concat([pd.read_pickle(D / f"features_{m}_v{FEAT_VERSION}.pkl"), pd.read_pickle(D / f"ctx_{m}_v{CTX_VERSION}.pkl")], axis=1)
        F = F.loc[:, ~F.columns.duplicated()].drop_duplicates("idx").set_index("idx")
        extra = [c for c in FEATURES + CTX_FEATURES if c not in T.columns]
        T = T.join(F[extra].add_prefix("k_f_"), on="kick_idx")
        Ts.append(T); Cs.append(pd.read_pickle(D / f"cand_{kind}_{m}.pkl"))
    T = pd.concat(Ts, ignore_index=True)
    if kind == "oracle" and (D / "rs_oracle_keys.pkl").exists():
        T = T.merge(pd.read_pickle(D / "rs_oracle_keys.pkl"), on=["match_id", "kick_idx"], how="left")
    return T, pd.concat(Cs, ignore_index=True)


EXTRA = ["k_f_" + c for c in FEATURES + CTX_FEATURES]
OUT2 = OUTCOME_FEATS + EXTRA
rf = lambda leaf=2: make_pipeline(SimpleImputer(strategy="median"), RF(n_estimators=200, min_samples_leaf=leaf, random_state=20261110, n_jobs=2))
hgb = lambda: H(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=20261110)


def cv_cls(T, cols, y, make, mask=None):
    pred = np.empty(len(T), object); mask = np.ones(len(T), bool) if mask is None else mask
    X = T[cols].astype(float).values
    for f in folds:
        te = T.match_id.isin(f).values; tr = ~te & mask & pd.notna(y)
        pred[te] = make().fit(X[tr], np.asarray(y)[tr].astype(str)).predict(X[te])
    return pred


def acc(p, t, m=None):
    p, t = np.asarray(p, object), np.asarray(t, object); m = pd.notna(t) if m is None else m
    return f"{(p[m] == t[m].astype(str)).mean():.1%} (n={int(m.sum())})"


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage != "players":
        T = pd.read_pickle(D / "fields_cv_T.pkl")
    else:
      T, C = load_all(); C = label(C, T)
      print("shots", len(T), "| candidate rows", len(C))
    if stage == "players":
      # ---- players -------------------------------------------------------------------------------
      picks = []
      for f in folds:
          te = C.match_id.isin(f).values
          r = PlayerRankers().fit(C[~te]); picks.append(PlayerRankers.pick(r.score(C[te])))
      Pk = pd.concat(picks); T = T.merge(Pk, on=["match_id", "kick_idx"], how="left")
      s = lambda v: np.array([None if pd.isna(x) else str(x) for x in v], object)
      print("SHOOTER   nearest attacker:", acc(s(T.pred_shooter), T.true_shooter_id), "| ranking model:", acc(s(T.rk_shooter), T.true_shooter_id))
      hb = T.true_blocker_id.notna().values
      print("BLOCKER   nearest on path :", acc(s(T.pred_blocker_any), T.true_blocker_id, hb), "| ranking model:", acc(s(T.rk_blocker), T.true_blocker_id, hb))
      hf = T.true_failed_blocker_id.notna().values
      hit = lambda a: np.mean([(x == t1) or (x == t2) for x, t1, t2 in zip(s(a)[hf], T.true_failed_blocker_id[hf], T.true_failed_blocker2_id[hf])])
      print(f"FAILED BLOCKER (one of PFF's two) nearest: {hit(T.fb1):.1%} | ranking model: {hit(T.rk_fb1):.1%} (n={hf.sum()})")
      best = max(((np.mean((T.rk_fb1_p.values > t) == hf)), t) for t in np.linspace(0.1, 0.7, 13))
      print(f"   'is there a failed blocker?' yes/no agreement {best[0]:.1%} at score > {best[1]:.2f} (always-no = {1 - hf.mean():.1%})")
      T.to_pickle(D / "fields_cv_T.pkl")
    if stage == "outcome":
      # ---- outcome ---------------------------------------------------------------------------------
      yg = T.true_outcome.map(GROUP)
      for name, cols, mk in (("v1 features, RF ", OUTCOME_FEATS, rf), ("+kick+context, GB", OUT2, hgb)):
          g = cv_cls(T, cols, yg, mk)
          o = np.array([refine_outcome(gg, r) for gg, (_, r) in zip(g, T.iterrows())], object)
          print(f"OUTCOME {name}: 4 groups {acc(g, yg)} | 7 types {acc(o, T.true_outcome)}")
          T["cv_outcome_" + name.strip()[-2:]] = o; T["cv_group"] = g
      T.to_pickle(D / "fields_cv_T.pkl")
    maj = lambda c: T["true_" + c].dropna().astype(str).mode().iloc[0]
    base = lambda c: f"{(T['true_' + c].dropna().astype(str) == maj(c)).mean():.1%}"
    if stage != "d2": sys.exit()
    # ---- delivery 2 ------------------------------------------------------------------------------
    for fld, cols in (("body_movement", MOVE_FEATS), ("shot_height", HEIGHT_FEATS), ("save_height", SAVEH_FEATS),
                      ("shot_nature", NATURE_FEATS), ("save_rebound", ["reb_x", "reb_y", "reb_z", "t1_x", "t1_absy", "t1_z_med"])):
        y = T["true_" + fld]; has = y.notna().values
        a = cv_cls(T, cols, y, lambda: rf(3), has); b = cv_cls(T, cols + EXTRA, y, hgb, has)
        print(f"{fld:14s} majority {base(fld)} | v1 features RF {acc(a, y)} | +kick+context GB {acc(b, y)}")
    bm = ~(T.ball_speed_before.fillna(5) < 1.0)
    print("ball_moving    majority", f"{T.true_ball_moving.mean():.1%}", "| rule", f"{(bm == T.true_ball_moving.astype(bool)).mean():.1%}",
          "| GB", acc(cv_cls(T, ["ball_speed_before", "ball_speed_before_min"] + EXTRA, T.true_ball_moving.astype(str), hgb), T.true_ball_moving.astype(str)))
    print("counts: deflector", int(T.true_deflector_id.notna().sum()) if "true_deflector_id" in T else "n/a", "| gl_clearer", int(T.true_gl_clearer_id.notna().sum()))

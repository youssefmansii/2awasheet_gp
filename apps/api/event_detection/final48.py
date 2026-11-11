"""
final48.py - train the final Member-4 model on ALL 48 training matches, then (once) test on the 16 unseen.
  python -m event_detection.final48 train      # shot model + field models; threshold = median of the 6 CV folds
  python -m event_detection.final48 features   # tracking-only features for the unseen matches (no answers read)
  python -m event_detection.final48 test       # ONE-TIME test on the 16 unseen matches
"""
import json, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from .eval_fields import oracle_table
from .field_models import FieldModels
from .run_member4 import Pipeline, end_to_end, field_report
from .shot_detector import ShotModel
from .tracking_reader import cache_dir
D = cache_dir()
SPLIT = json.load(open(D.parents[1] / "split_member4.json"))
TRAIN = [str(m) for m in sorted(SPLIT["train"])]
UNSEEN = [str(m) for m in sorted(SPLIT["unseen"])]
MODEL = D / "member4_models_v2.pkl"

cmd = sys.argv[1]
if cmd == "train":
    F = pd.read_pickle(D / "cv48_F.pkl")
    pipe = Pipeline()
    pipe.shot = ShotModel().fit(F, F.y.values)
    pipe.threshold = float(np.median([pickle.load(open(D / f"cv48_fold{i}.pkl", "rb"))["thr"] for i in range(6)]))
    from .field_models2 import FieldModels2
    from .fields_cv import load_all
    pipe.fields = FieldModels2().fit(*load_all(TRAIN))
    pipe.train_matches = TRAIN
    pickle.dump(pipe, open(MODEL, "wb"))
    print("saved", MODEL.name, "threshold", round(pipe.threshold, 3))
elif cmd == "test":
    log = D / "unseen_test_log_v2.json"
    if log.exists() and "--again" not in sys.argv:
        raise SystemExit("the unseen test was already run once (see unseen_test_log.json)")
    pipe = pickle.load(open(MODEL, "rb"))
    t0 = time.time(); start = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
    part = D / "unseen_parts_v2.pkl"
    done = pickle.load(open(part, "rb")) if part.exists() else {}
    for m in UNSEEN:
        if m in done:
            continue
        if time.time() - t0 > 140:
            break
        pred = pipe.predict(m, final=True)
        pred.to_csv(D / f"pred_shots_{m}.csv", index=False)
        done[m] = end_to_end(pred, m, final=True)
        pickle.dump(done, open(part, "wb"))
        print(m, {k: round(v, 3) if isinstance(v, float) else v for k, v in done[m][0].items()}, flush=True)
    if len(done) == len(UNSEEN):
        det = [d for d, _ in done.values()]
        tp = sum(d["tp"] for d in det); npred = sum(d["pred"] for d in det); ntrue = sum(d["true"] for d in det)
        P, R = tp / npred, tp / ntrue
        s = {"matches": len(det), "pred": npred, "true": ntrue, "tp": tp, "precision": P, "recall": R, "f1": 2 * P * R / (P + R)}
        fr = field_report(pd.concat([p for _, p in done.values()], ignore_index=True))
        print("UNSEEN TEST:", {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
        print(fr.round(3).to_string(index=False))
        json.dump({"summary": s, "time": time.strftime("%Y-%m-%d %H:%M")}, open(log, "w"), indent=1)
        fr.to_csv(D / "unseen_fields_v2.csv", index=False)
        pd.DataFrame([{"match": m, **{k: round(v, 3) for k, v in d.items()}} for m, (d, _) in done.items()]).to_csv(D / "unseen_detection_v2.csv", index=False)
    else:
        print("partial:", len(done), "of", len(UNSEEN), "- run again to continue")

"""Build/caches features (current FEAT_VERSION) for the training matches, sharded.  Tracking only."""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from .features import FEAT_VERSION, build_features
from .tracking_reader import cache_dir, load_tracking
t0 = time.time(); budget = float(sys.argv[1]); shard, n = map(int, sys.argv[2].split("/"))
which = sys.argv[3] if len(sys.argv) > 3 else "train"
split = json.load(open(cache_dir().parents[1] / "split_member4.json"))
ms = [str(m) for m in sorted(split[which])][shard::n]
done = []
for m in ms:
    p = cache_dir() / f"features_{m}_v{FEAT_VERSION}.pkl"
    if p.exists():
        continue
    if time.time() - t0 > budget - 12:
        break
    build_features(load_tracking(m))[0].to_pickle(p); done.append(m)
left = [m for m in ms if not (cache_dir() / f"features_{m}_v{FEAT_VERSION}.pkl").exists()]
print("built", len(done), "left", len(left))

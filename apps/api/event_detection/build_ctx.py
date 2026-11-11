import json, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from .context import CTX_VERSION, context_features
from .features import FEAT_VERSION
from .tracking_reader import cache_dir, load_tracking
t0 = time.time(); shard, n = map(int, sys.argv[1].split("/")); which = sys.argv[2] if len(sys.argv) > 2 else "train"
D = cache_dir(); split = json.load(open(D.parents[1] / "split_member4.json"))
built = 0
for m in [str(x) for x in sorted(split[which])][shard::n]:
    p = D / f"ctx_{m}_v{CTX_VERSION}.pkl"
    if p.exists(): continue
    if time.time() - t0 > 150: break
    F = pd.read_pickle(D / f"features_{m}_v{FEAT_VERSION}.pkl")
    context_features(load_tracking(m), F).to_pickle(p); built += 1
print("built", built, round(time.time() - t0), "s")

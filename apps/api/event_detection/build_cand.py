import json, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from .ball_physics import clean_ball
from .features import FEAT_VERSION
from .player_rank import candidates
from .tracking_reader import cache_dir, load_tracking
t0 = time.time(); shard, n = map(int, sys.argv[1].split("/"))
D = cache_dir(); split = json.load(open(D.parents[1] / "split_member4.json")); built = 0
for m in [str(x) for x in sorted(split["train"])][shard::n]:
    p = D / f"cand_oracle_{m}.pkl"
    if p.exists(): continue
    if time.time() - t0 > 150: break
    tr = load_tracking(m); T = pd.read_pickle(D / f"step4_oracle_{m}.pkl")
    F = pd.read_pickle(D / f"features_{m}_v{FEAT_VERSION}.pkl")
    candidates(tr, clean_ball(tr), F.idx.values, T.kick_idx.tolist()).to_pickle(p); built += 1
print("built", built, round(time.time() - t0), "s")

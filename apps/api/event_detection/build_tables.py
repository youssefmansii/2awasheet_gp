"""build_tables.py - build (and cache) the shot-candidate table and the per-shot field table for the
TRAINING matches only, in shards so each call stays short.
Run:  python -m event_detection.build_tables <seconds> <shard>/<nshard>"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
from .eval_fields import oracle_table
from .features import FEAT_VERSION
from .tracking_reader import cache_dir

t0 = time.time(); budget = float(sys.argv[1]); shard, n = map(int, sys.argv[2].split("/"))
split = json.load(open(cache_dir().parents[1] / "split_member4.json"))
train = [str(m) for m in sorted(split["train"])][shard::n]
todo = [m for m in train if not (cache_dir() / f"step4_oracle_{m}.pkl").exists()
        or not (cache_dir() / f"features_{m}_v{FEAT_VERSION}.pkl").exists()]
done = []
for m in todo:
    if time.time() - t0 > budget - 25:
        break
    oracle_table([m]); done.append(m)
print("built", done, "| left in this shard", len(todo) - len(done))

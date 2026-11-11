"""prep_cache.py - import finished downloads, then build tracking caches until a time budget runs out.
Only TRACKING is read here (no answer key), so it is safe for unseen matches too.
Run:  python -m event_detection.prep_cache <downloads_dir> <seconds>"""
import sys, time
from pathlib import Path
from .import_downloads import main as import_main
from .tracking_reader import READER_VERSION, cache_dir, data_dir, load_tracking

t0 = time.time(); budget = float(sys.argv[2])
shard, nshard = (map(int, sys.argv[3].split("/")) if len(sys.argv) > 3 else (0, 1))
if shard == 0:
    import_main(Path(sys.argv[1]), data_dir())
ids = sorted(p.name.split("_")[1].split(".")[0] for p in data_dir().glob("tracking_*.jsonl.bz2"))
todo = [m for m in ids if not (cache_dir() / f"tracking_{m}_v{READER_VERSION}.npz").exists()][shard::nshard]
done = []
for m in todo:
    if time.time() - t0 > budget - 40:
        break
    load_tracking(m); done.append(m)
print("cached now:", done, "| still to cache:", len(todo) - len(done))

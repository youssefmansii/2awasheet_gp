"""
Live demo, one command: raw tracking file in -> Member 5 event file out, ready for the replay app.

    python -m pff5.predict_match --tracking 3816.jsonl --events "3816(events).json" \
        --metadata 3816.json --run models_for_3816/raw --out 3816_member5_events.json

What it does
    1. reads the raw PFF tracking file (.jsonl or .jsonl.bz2) - about 1-2 minutes
    2. finds challenges from tracking alone and classifies them
    3. predicts pressure and fouls / offside at the event moments

The event file is used ONLY for the moments and the player on the ball (the part other
team members detect) and for player names. No labels are read from it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from . import export_events
from .preprocess import Source, convert_game


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", required=True, help="raw PFF tracking file, .jsonl or .jsonl.bz2")
    ap.add_argument("--events", required=True, help="PFF event file of the same match (moments + names only)")
    ap.add_argument("--metadata", default=None, help="PFF metadata file of the match (recommended)")
    ap.add_argument("--run", required=True, help="model folder, e.g. models_for_3816/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="where to keep the converted match (default: temp folder)")
    a = ap.parse_args(argv)

    t0 = time.time()
    meta = None
    if a.metadata:
        meta = json.load(open(a.metadata, encoding="utf-8"))
        meta = meta[0] if isinstance(meta, list) else meta
    if meta and meta.get("id"):
        gid = int(meta["id"])
    else:
        with open(a.events, "rb") as f:
            head = f.read(4000).decode("utf-8", "ignore")
        gid = int(head.split('"gameId":', 1)[1].split(",", 1)[0].strip())
    work = a.work or tempfile.mkdtemp(prefix="member5_")
    os.makedirs(work, exist_ok=True)
    print(f"[1/2] reading tracking for match {gid} (this takes 1-2 minutes) ...", flush=True)
    convert_game(str(gid), Source(a.tracking), Source(a.events), Source(a.metadata) if a.metadata else None, None,
                 work, overwrite=True)
    print(f"[2/2] running the Member 5 models ...", flush=True)
    export_events.main(["--run", a.run, "--compact", work, "--game", str(gid), "--out", a.out])
    print(f"done in {time.time() - t0:.0f}s -> load {a.out} as the events file in the replay app", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

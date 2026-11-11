"""
test_leakage.py - proves PFF's hand-made events can NOT reach the reader output.

Run (from apps/api):   python -m event_detection.test_leakage --match 3830
It parses the first N tracking lines twice: once as they are, once with every event field
scrambled or deleted.  The outputs must be byte-for-byte identical.  It also checks that no
feature module reads the forbidden keys or imports the answer key.
"""
from __future__ import annotations

import argparse
import bz2
import itertools
import json
import re
from pathlib import Path

import numpy as np

from .tracking_reader import DERIVED_KEYS, FORBIDDEN_KEYS, find_files, parse_lines

FEATURE_MODULES = ["tracking_reader.py", "ball_physics.py", "shot_detector.py", "shot_fields.py", "features.py", "field_models.py", "field_models2.py", "context.py",
                   "restart_feats.py"]


def _lines(match_id, n):
    p = find_files(match_id)["tracking"]
    f = bz2.open(p, "rb") if str(p).endswith(".bz2") else open(p, "rb")
    with f:
        return list(itertools.islice(f, n))


def _scramble(line: bytes, mode: int) -> bytes:
    o = json.loads(line)
    for k in FORBIDDEN_KEYS + DERIVED_KEYS:
        if mode == 0:
            o.pop(k, None)
        else:
            o[k] = {"possession_event_type": "SH", "start_frame": 1, "x": 99} if "event" in k and not k.endswith("_id") else 123
    return json.dumps(o).encode()


def run(match_id, n=3000) -> bool:
    base = _lines(match_id, n)
    a = parse_lines(base)
    ok = True
    for mode in (0, 1):
        b = parse_lines([_scramble(l, mode) for l in base])
        same = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a if k != "keys") and a["keys"] == b["keys"]
        print(f"  event fields {'deleted' if mode == 0 else 'scrambled'}: output identical = {same}")
        ok &= same
    here = Path(__file__).parent
    for name in FEATURE_MODULES:
        p = here / name
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        bad = re.findall(r"answer_key|load_events|load_shots|possessionEvents|\[\"(?:game|possession)_event\"\]|ball_s\b", src)
        print(f"  {name}: forbidden references = {bad or 'none'}")
        ok &= not bad
    print("LEAKAGE TEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="3830")
    run(ap.parse_args().match)

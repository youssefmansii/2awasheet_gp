"""
import_downloads.py - move PFF files downloaded by the browser (named m4_<kind>_<id>.<ext>) from
Downloads into data/, under the names the project uses, after checking each file is complete.
  tracking -> tracking_<id>.jsonl.bz2   (bz2 integrity test)
  events / metadata / roster -> <kind>_<id>.json   (must parse as JSON with the right structure)
Run:  python -m event_detection.import_downloads <downloads_dir>
"""
from __future__ import annotations

import bz2
import json
import re
import shutil
import sys
from pathlib import Path

PAT = re.compile(r"^m4_(tracking|events|metadata|roster)_(\d+)(?: \(\d+\))?\.(jsonl\.bz2|json)$")


def check(kind: str, p: Path) -> str | None:
    try:
        if kind == "tracking":
            # quick check here; the full decompression happens when the reader caches the file,
            # and a truncated .bz2 raises an error there
            with bz2.open(p, "rb") as f:
                json.loads(f.readline())
            return None
        d = json.load(open(p, encoding="utf-8"))
        if kind == "events" and not (isinstance(d, list) and d and "possessionEvents" in d[0]):
            return "not an event file"
        if kind == "roster" and not (isinstance(d, list) and d and "shirtNumber" in d[0]):
            return "not a roster"
        if kind == "metadata":
            m = d[0] if isinstance(d, list) else d
            if "homeTeam" not in m:
                return "not metadata"
        return None
    except Exception as e:  # incomplete / broken download
        return f"broken: {e.__class__.__name__}"


def main(src: Path, dst: Path):
    moved, bad = [], []
    for p in sorted(src.iterdir()):
        m = PAT.match(p.name)
        if not m:
            continue
        kind, mid, ext = m.groups()
        target = dst / (f"tracking_{mid}.jsonl.bz2" if kind == "tracking" else f"{kind}_{mid}.json")
        if target.exists():
            p.unlink(); continue
        err = check(kind, p)
        if err:
            bad.append((p.name, err)); continue
        shutil.copyfile(p, target)
        p.unlink()
        moved.append(target.name)
    print(f"moved {len(moved)}; problems {len(bad)}")
    for b in bad:
        print("  ", *b)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parents[3] / "data")

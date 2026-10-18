"""Small encoding contract fixture; real captures stay outside Git."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import _parse_tracking


class LoaderEncodingTests(unittest.TestCase):
    def test_utf8_bom_does_not_drop_first_tracking_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.jsonl'
            path.write_bytes(b'\xef\xbb\xbf{"frameNum":3056,"period":1,"videoTimeMs":1000,"homePlayers":[{"jerseyNum":10,"x":0,"y":0}]}\n')
            parsed = _parse_tracking(path, 'raw')
            self.assertEqual(parsed['malformed'], 0)
            self.assertEqual(int(parsed['meta'][0, 0]), 3056)

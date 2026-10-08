"""Checks on the committed Kvasir-SEG split files (splits/kvasir_*.txt).

These guard the split that every experiment depends on: sizes, no overlap, and
that the files are exactly what seed 42 produces from the sorted image ids.
"""

import random
import unittest
from pathlib import Path

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
SIZES = {"train": 700, "val": 100, "test": 200}
SEED = 42


def read_split(name):
    return (SPLITS_DIR / f"kvasir_{name}.txt").read_text().split()


class SplitFileTests(unittest.TestCase):
    def test_sizes(self):
        for name, size in SIZES.items():
            self.assertEqual(len(read_split(name)), size, name)

    def test_no_overlap_and_no_duplicates(self):
        all_ids = [i for name in SIZES for i in read_split(name)]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_each_file_is_sorted(self):
        for name in SIZES:
            ids = read_split(name)
            self.assertEqual(ids, sorted(ids), name)

    def test_reproducible_from_seed(self):
        all_ids = sorted(i for name in SIZES for i in read_split(name))
        shuffled = list(all_ids)
        random.Random(SEED).shuffle(shuffled)
        start = 0
        for name, size in SIZES.items():
            self.assertEqual(read_split(name), sorted(shuffled[start:start + size]), name)
            start += size

    def test_ids_look_like_kvasir_ids(self):
        for name in SIZES:
            for image_id in read_split(name):
                self.assertRegex(image_id, r"^[a-z0-9]+$", image_id)


if __name__ == "__main__":
    unittest.main()

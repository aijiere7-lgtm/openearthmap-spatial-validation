"""Portable tests for mIoU and the frozen archive."""
from pathlib import Path
from subprocess import run
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.metrics import ConfusionMatrix


class PortableTests(unittest.TestCase):
    def test_pooled_iou_ignores_padding_and_absent_classes(self):
        confusion = ConfusionMatrix(n_classes=8)
        confusion.update(np.array([0, 0, 1, 1, 2, 2, 0, 1]),
                         np.array([0, 0, 1, 2, 2, 1, 1, 0]))
        confusion.update(np.array([3]), np.array([255]), ignore_index=255)
        self.assertAlmostEqual(confusion.all_metrics()["miou"], (0.5 + 0.2 + 1/3) / 3)

    def test_archive_verifier(self):
        result = run([sys.executable, str(ROOT / "scripts/verify_records.py")],
                     cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified 75 JSON records", result.stdout)


if __name__ == "__main__":
    unittest.main()

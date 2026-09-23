"""Known-outcome checks for fixed-confidence validation diagnostics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval import bucket, match_at_fixed_conf, read_yolo, ROOT


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_size_buckets_use_normalized_sqrt_area(self):
        self.assertEqual(bucket(np.array([0., 0., .03, .03])), 0)
        self.assertEqual(bucket(np.array([0., 0., .075, .075])), 1)
        self.assertEqual(bucket(np.array([0., 0., .15, .15])), 2)
        self.assertEqual(bucket(np.array([0., 0., .25, .25])), 3)
        self.assertEqual(bucket(np.array([0., 0., .40, .40])), 4)

    def test_duplicate_detection_is_false_positive(self):
        gt = np.array([[0., 0., .1, .1], [.5, .5, .7, .7]])
        pred = np.array([[0., 0., .1, .1], [0., 0., .1, .1],
                         [.5, .5, .7, .7], [.5, .5, .7, .7]])
        conf = np.array([.9, .8, .7, .1])
        eligible, matches, matched = match_at_fixed_conf(gt, pred, conf)
        self.assertEqual(len(eligible), 3)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matched, {0, 2})

    def test_zero_height_low_confidence_prediction_is_readable(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'prediction.txt'
            path.write_text('0 0.141153 1 0.168534 0 0.00100633\n', encoding='utf-8')
            boxes, confidence = read_yolo(path, prediction=True)
            self.assertEqual(boxes.shape, (1, 4))
            self.assertEqual(confidence.shape, (1,))


if __name__ == '__main__':
    unittest.main()

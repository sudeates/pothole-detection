"""Small, known-outcome checks for the SAHI mAP matching step."""
import unittest

import numpy as np

from eval_p2_sahi import correct_matrix


class MatchingTests(unittest.TestCase):
    def test_duplicate_predictions_match_one_label_once(self):
        labels = np.array([[0., 0., 10., 10.]])
        predictions = np.array([[0., 0., 10., 10., .9],
                                [0., 0., 10., 10., .8]])
        correct = correct_matrix(labels, predictions)
        np.testing.assert_array_equal(correct.sum(axis=0), np.ones(10))

    def test_iou_thresholds_and_empty_image(self):
        labels = np.array([[0., 0., 10., 10.]])
        prediction = np.array([[2., 0., 12., 10., .8]])  # IoU = 2/3
        correct = correct_matrix(labels, prediction)
        np.testing.assert_array_equal(correct[0], [True, True, True, True,
                                                    False, False, False, False, False, False])
        self.assertEqual(correct_matrix(np.empty((0, 4)), prediction).sum(), 0)


if __name__ == '__main__':
    unittest.main()

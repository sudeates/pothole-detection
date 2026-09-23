"""Behavioral checks for crop geometry and deterministic negative selection."""
import unittest
from tile_yolo_dataset import project_box, retain_negative, starts


class TileGeometryTests(unittest.TestCase):
    def test_boundary_rule_is_strict(self):
        box = (500.0, 100.0, 700.0, 300.0)
        kept, clipped = project_box(box, 0, 0, 640, 640, 0.30)
        self.assertTrue(clipped)
        self.assertIsNotNone(kept)  # 70% of the box is inside.
        dropped, intersects = project_box(box, 640, 0, 1280, 640, 0.30)
        self.assertIsNone(dropped)  # Exactly 30% is not greater than 30%.
        self.assertTrue(intersects)
        values = list(map(float, kept.split()))
        self.assertAlmostEqual(values[1], 570 / 640)
        self.assertAlmostEqual(values[3], 140 / 640)

    def test_tiles_cover_right_edge_and_repeatably_sample_negatives(self):
        self.assertEqual(starts(1000, 640, 0.20), [0, 360])
        self.assertEqual(starts(500, 640, 0.20), [0])
        self.assertEqual(retain_negative('sample.jpg', 512, 0, 42, 0.10),
                         retain_negative('sample.jpg', 512, 0, 42, 0.10))


if __name__ == '__main__':
    unittest.main()

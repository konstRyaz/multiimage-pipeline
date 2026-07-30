from __future__ import annotations

import unittest

import numpy as np

from face_pipeline.clustering import automatic_groups


class PhotoClusteringTests(unittest.TestCase):
    def test_faces_from_same_photo_cannot_join_one_identity(self) -> None:
        tracks = [
            {
                "video_name": "2926_photo_1_a1b2c3",
                "start_frame": "0",
                "end_frame": "0",
            },
            {
                "video_name": "2926_photo_1_a1b2c3",
                "start_frame": "0",
                "end_frame": "0",
            },
            {
                "video_name": "6210_photo_1_d4e5f6",
                "start_frame": "0",
                "end_frame": "0",
            },
        ]
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )

        groups = automatic_groups(tracks, vectors, threshold=0.45)

        self.assertEqual(sorted(len(group) for group in groups), [1, 2])
        self.assertFalse(any({0, 1}.issubset(group) for group in groups))


if __name__ == "__main__":
    unittest.main()

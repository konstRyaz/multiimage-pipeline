from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from face_pipeline.analysis import run_analysis
from face_pipeline.clustering import automatic_groups, run_clustering
from face_pipeline.io import DataValidationError, load_face_run
from face_pipeline.selection import run_selection
from face_pipeline.tracking import run_tracking
from evaluate_synthetic import evaluate
from generate_synthetic import run as generate_synthetic
from process_faces import parse_frame


class PipelineTests(unittest.TestCase):
    def test_parse_frame_name(self) -> None:
        video, number = parse_frame(Path("05 2 What Do People Really Enjoymp4_frame_0117.jpg"))
        self.assertEqual(video, "05 2 What Do People Really Enjoymp4")
        self.assertEqual(number, 117)

    def test_same_frame_tracks_cannot_merge(self) -> None:
        rows = [
            {"video_name": "v", "start_frame": "10", "end_frame": "20"},
            {"video_name": "v", "start_frame": "15", "end_frame": "25"},
        ]
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        groups = automatic_groups(rows, np.stack([vector, vector]), threshold=0.45)
        self.assertEqual(len(groups), 2)

    def test_end_to_end_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "synthetic"
            generated = generate_synthetic(run_dir, seed=42, overwrite=False)
            self.assertEqual(generated["real_identities"], 3)
            rows, vectors = load_face_run(run_dir, require_images=True)
            self.assertEqual(len(rows), vectors.shape[0])
            self.assertEqual(vectors.shape[1], 512)
            analysis = run_analysis(run_dir)
            self.assertGreater(analysis["same_frame_pairs"], 0)
            tracking = run_tracking(run_dir)
            self.assertLess(tracking["tracks"], tracking["faces"])
            clustering = run_clustering(run_dir)
            self.assertEqual(clustering["known_clusters"], 3)
            selection = run_selection(run_dir)
            self.assertEqual(selection["identities"], 3)
            self.assertGreaterEqual(selection["selected_faces"], 3)
            result = evaluate(run_dir)
            self.assertTrue(result["passed"])

    def test_validation_rejects_bad_embedding_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "synthetic"
            generate_synthetic(run_dir, seed=1, overwrite=False)
            csv_path = run_dir / "faces.csv"
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]["embedding_index"] = rows[1]["embedding_index"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(DataValidationError):
                load_face_run(run_dir)


if __name__ == "__main__":
    unittest.main()

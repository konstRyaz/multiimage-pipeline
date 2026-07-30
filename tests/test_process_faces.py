from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "process_faces.py"
SPEC = importlib.util.spec_from_file_location("process_faces", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESS_FACES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS_FACES)


class ParseSourceTests(unittest.TestCase):
    def test_photos_with_same_filename_have_different_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            first = root / "2926" / "photo_1.jpg"
            second = root / "6210" / "photo_1.jpg"

            first_id, first_frame = PROCESS_FACES.parse_source(
                first, root, "photos"
            )
            second_id, second_frame = PROCESS_FACES.parse_source(
                second, root, "photos"
            )

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(first_frame, 0)
            self.assertEqual(second_frame, 0)
            self.assertIn("2926_photo_1", first_id)
            self.assertIn("6210_photo_1", second_id)

    def test_photo_source_id_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "2926" / "photo_1.jpg"
            first = PROCESS_FACES.parse_source(path, root, "photos")
            second = PROCESS_FACES.parse_source(path, root, "photos")
            self.assertEqual(first, second)

    def test_video_frame_parsing_remains_compatible(self) -> None:
        root = Path("/dataset")
        path = root / "clip_frame_000123.jpg"
        self.assertEqual(
            PROCESS_FACES.parse_source(path, root, "video-frames"),
            ("clip", 123),
        )


if __name__ == "__main__":
    unittest.main()

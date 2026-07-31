from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "baseline_datasets.py"
SPEC = importlib.util.spec_from_file_location("baseline_datasets", SCRIPT)
assert SPEC and SPEC.loader
datasets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(datasets)


class BaselineDatasetsTest(unittest.TestCase):
    def test_extract_zip_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(RuntimeError, "небезопасный путь"):
                datasets.extract_zip(archive, root / "out")
            self.assertFalse((root / "outside.txt").exists())

    def test_extract_zip_and_image_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("nested/a.JPG", b"image")
                zipped.writestr("nested/readme.txt", b"text")
            datasets.extract_zip(archive, root / "out")
            self.assertEqual(datasets.image_count(root / "out"), 1)

    def test_annotation_rows_supports_header_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.txt"
            path.write_text("2\na 1\nb 2\n", encoding="utf-8")
            self.assertEqual(datasets.annotation_rows(path), 2)

    def test_xqlfw_pair_count_supports_lfw_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.txt"
            path.write_text("10 300\nname 1 2\nother 1 name 2\n", encoding="utf-8")
            self.assertEqual(datasets.xqlfw_pair_count(path), 6000)


if __name__ == "__main__":
    unittest.main()

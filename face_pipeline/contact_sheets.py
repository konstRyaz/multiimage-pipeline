from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .io import resolve_path


def _font(size: int = 13) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_contact_sheet(
    items: Iterable[tuple[dict[str, str], str]],
    run_dir: Path,
    output_path: Path,
    columns: int = 8,
    thumb_size: int = 112,
) -> int:
    material = list(items)
    if not material:
        return 0
    label_height = 34
    rows_count = math.ceil(len(material) / columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows_count * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(12)
    for index, (row, label) in enumerate(material):
        x = (index % columns) * thumb_size
        y = (index // columns) * (thumb_size + label_height)
        image_path = resolve_path(row.get("aligned_path", ""), run_dir)
        if image_path is None:
            image = Image.new("RGB", (thumb_size, thumb_size), (210, 210, 210))
            missing = ImageDraw.Draw(image)
            missing.line((0, 0, thumb_size, thumb_size), fill="red", width=3)
            missing.line((thumb_size, 0, 0, thumb_size), fill="red", width=3)
        else:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_size, thumb_size), "black")
                canvas.paste(image, ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2))
                image = canvas
        sheet.paste(image, (x, y))
        draw.multiline_text((x + 3, y + thumb_size + 2), label[:42], fill="black", font=font, spacing=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    sheet.save(temp, format="JPEG", quality=90)
    temp.replace(output_path)
    return len(material)

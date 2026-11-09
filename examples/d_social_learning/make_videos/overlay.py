"""Text overlay for the stage videos: a fixed info panel (top left) and a
live bar (bottom). Pillow only, so it runs in both the ARIEL and the EvoGym
venv.

    panel = InfoPanel(lines, width, height)
    frame = panel.draw(frame, live="t = 5.3 s    displacement so far 0.42 m")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


class InfoPanel:
    """Static lines are drawn once into an RGBA layer, then composited onto
    every frame together with the live line."""

    def __init__(self, lines: list[str], width: int, height: int) -> None:
        self.width, self.height = width, height
        size = max(12, round(height / 40))
        self.font, self.bold = _font(size), _font(size, bold=True)
        self.pad, self.gap = round(size * 0.6), round(size * 0.35)

        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        fonts = [self.bold] + [self.font] * (len(lines) - 1)
        line_h = [d.textbbox((0, 0), "Ag", font=f)[3] for f in fonts]
        box_w = max(d.textlength(t, font=f) for t, f in zip(lines, fonts)) + 2 * self.pad
        box_h = sum(line_h) + self.gap * (len(lines) - 1) + 2 * self.pad
        d.rounded_rectangle((8, 8, 8 + box_w, 8 + box_h), radius=self.pad, fill=(0, 0, 0, 150))
        y = 8 + self.pad
        for text, f, h in zip(lines, fonts, line_h):
            d.text((8 + self.pad, y), text, font=f, fill=(255, 255, 255, 255))
            y += h + self.gap
        self.layer = layer
        self.live_h = line_h[-1] + 2 * self.pad

    def draw(self, frame: np.ndarray, live: str | None = None) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGBA")
        img.alpha_composite(self.layer)
        if live:
            # A separate layer, so the bar blends (drawing on img would overwrite)
            bar = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(bar)
            y0 = self.height - self.live_h
            d.rectangle((0, y0, self.width, self.height), fill=(0, 0, 0, 150))
            d.text((8 + self.pad, y0 + self.pad), live, font=self.font, fill=(255, 255, 255, 255))
            img.alpha_composite(bar)
        return np.asarray(img.convert("RGB"))


def fmt(v, digits: int = 2, unit: str = "") -> str:
    """Number with a fixed number of decimals, or an en dash when missing."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(f):
        return "–"
    return f"{f:.{digits}f}{unit}"

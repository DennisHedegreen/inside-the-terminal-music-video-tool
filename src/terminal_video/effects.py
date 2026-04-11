from __future__ import annotations

from dataclasses import dataclass
import math
import random

from PIL import Image, ImageDraw, ImageFilter

from .audio_analyzer import AudioAnalysis


@dataclass(frozen=True)
class EffectState:
    energy: float
    brightness: float
    beat: float
    glow: float
    cursor_alpha: int
    text_boost: float


def effect_state(analysis: AudioAnalysis, seconds: float, mode: str, config: dict) -> EffectState:
    effects = config.get("effects", {})
    energy = analysis.energy_at(seconds)
    brightness = analysis.brightness_at(seconds)
    beat = analysis.beat_proximity(seconds)
    mode_boost = {
        "boot": 0.15,
        "verse": 0.25,
        "interface": 0.35,
        "chorus": 0.62,
        "sidebar": 0.42,
        "shelter": 0.48,
        "bridge": -0.35,
        "final": 0.72,
    }.get(mode, 0.3)
    glow = max(
        0.05,
        effects.get("base_glow", 0.2)
        + energy * effects.get("rms_glow", 0.5)
        + beat * effects.get("beat_glow", 0.5)
        + mode_boost,
    )
    if mode == "bridge":
        glow *= 0.38 + beat * 0.35
    cursor_phase = math.sin(seconds * (5.5 + beat * 7.0))
    cursor_alpha = int(90 + max(0.0, cursor_phase) * 165)
    text_boost = min(1.0, brightness * effects.get("text_energy", 0.25) + beat * 0.15)
    return EffectState(
        energy=energy,
        brightness=brightness,
        beat=beat,
        glow=min(1.0, glow),
        cursor_alpha=cursor_alpha,
        text_boost=text_boost,
    )


def draw_scanlines(image: Image.Image, alpha: int) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    for y in range(0, height, 4):
        draw.line((0, y, width, y), fill=(0, 0, 0, alpha), width=1)
    image.alpha_composite(overlay)


def draw_noise(image: Image.Image, alpha: int, seed: int) -> None:
    random.seed(seed)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    for _ in range(900):
        x = random.randrange(0, width)
        y = random.randrange(0, height)
        shade = random.randrange(20, 95)
        draw.point((x, y), fill=(0, shade, 25, alpha))
    image.alpha_composite(overlay)


def glow_rect(image: Image.Image, rect: tuple[int, int, int, int], color: tuple[int, int, int], strength: float) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(90 * min(1.0, max(0.0, strength)))
    for width, blur in [(8, 14), (3, 5)]:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.rounded_rectangle(rect, radius=12, outline=(*color, alpha), width=width)
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        overlay.alpha_composite(layer)
    image.alpha_composite(overlay)

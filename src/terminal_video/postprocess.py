from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
import subprocess

from PIL import Image, ImageDraw, ImageFilter

from .mux import ffmpeg_path


@dataclass(frozen=True)
class PostPreset:
    name: str
    description: str
    contrast: float
    saturation: float
    gamma_green: float
    brightness_offset: float
    flicker_primary: float
    flicker_secondary: float
    chroma_blue_shift: int
    chroma_red_shift: int
    noise_strength: int
    vignette_angle: str
    ghost_opacity: float
    ghost_tail_weight: float
    ghost_blur_sigma: float
    green_curve: str
    grime_alpha: int
    grime_dust_count: int
    grime_smudge_count: int
    grime_streak_count: int
    grime_drift_x: int
    grime_drift_y: int


PRESETS: dict[str, PostPreset] = {
    "clean-distress": PostPreset(
        name="clean-distress",
        description="Subtle phosphor flicker, light ghosting, and a restrained green terminal rolloff.",
        contrast=1.05,
        saturation=0.92,
        gamma_green=1.03,
        brightness_offset=-0.012,
        flicker_primary=0.006,
        flicker_secondary=0.003,
        chroma_blue_shift=-1,
        chroma_red_shift=1,
        noise_strength=3,
        vignette_angle="PI/4.3",
        ghost_opacity=0.09,
        ghost_tail_weight=0.12,
        ghost_blur_sigma=0.7,
        green_curve="0/0 0.35/0.39 0.72/0.80 1/0.97",
        grime_alpha=0,
        grime_dust_count=0,
        grime_smudge_count=0,
        grime_streak_count=0,
        grime_drift_x=0,
        grime_drift_y=0,
    ),
    "dirty-terminal": PostPreset(
        name="dirty-terminal",
        description="Adds glass grime, dust, and a stronger terminal-body presence without collapsing the image.",
        contrast=1.08,
        saturation=0.89,
        gamma_green=1.04,
        brightness_offset=-0.018,
        flicker_primary=0.010,
        flicker_secondary=0.005,
        chroma_blue_shift=-2,
        chroma_red_shift=2,
        noise_strength=5,
        vignette_angle="PI/4.9",
        ghost_opacity=0.13,
        ghost_tail_weight=0.16,
        ghost_blur_sigma=1.1,
        green_curve="0/0 0.28/0.33 0.64/0.74 1/0.95",
        grime_alpha=42,
        grime_dust_count=1600,
        grime_smudge_count=12,
        grime_streak_count=18,
        grime_drift_x=6,
        grime_drift_y=4,
    ),
    "signal-damaged": PostPreset(
        name="signal-damaged",
        description="Heavier signal decay with stronger ghosting, chroma split, flicker, and monitor grime.",
        contrast=1.10,
        saturation=0.84,
        gamma_green=1.05,
        brightness_offset=-0.026,
        flicker_primary=0.016,
        flicker_secondary=0.009,
        chroma_blue_shift=-4,
        chroma_red_shift=4,
        noise_strength=8,
        vignette_angle="PI/5.3",
        ghost_opacity=0.18,
        ghost_tail_weight=0.24,
        ghost_blur_sigma=1.8,
        green_curve="0/0 0.22/0.28 0.55/0.68 1/0.93",
        grime_alpha=56,
        grime_dust_count=2400,
        grime_smudge_count=22,
        grime_streak_count=28,
        grime_drift_x=12,
        grime_drift_y=8,
    ),
}


@dataclass(frozen=True)
class PostprocessResult:
    input_path: Path
    output_path: Path
    report_path: Path
    preset: str
    intensity: float
    start_seconds: float | None
    duration_seconds: float | None
    texture_path: Path | None
    width: int
    height: int


def list_presets() -> list[PostPreset]:
    return [PRESETS[name] for name in sorted(PRESETS.keys())]


def get_preset(name: str) -> PostPreset:
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise KeyError(f"unknown preset '{name}'. available: {available}")
    return PRESETS[name]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _probe_dimensions(input_path: Path) -> tuple[int, int]:
    cmd = [ffmpeg_path(), "-hide_banner", "-i", str(input_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    probe_text = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"(\d{2,5})x(\d{2,5})", probe_text)
    if not match:
        raise RuntimeError(f"could not probe video dimensions for {input_path}")
    return int(match.group(1)), int(match.group(2))


def _grime_texture_needed(preset: PostPreset, intensity: float) -> bool:
    return preset.grime_alpha > 0 and intensity > 0.0


def _build_grime_texture(path: Path, width: int, height: int, preset: PostPreset, intensity: float, seed: int) -> None:
    rng = random.Random(seed)
    alpha_scale = _clamp(intensity, 0.35, 1.85)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    dust_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dust_draw = ImageDraw.Draw(dust_layer)
    dust_count = int(preset.grime_dust_count * alpha_scale)
    for _ in range(dust_count):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        radius = rng.choice((1, 1, 2, 2, 3))
        alpha = rng.randint(8, max(10, int(preset.grime_alpha * 0.65)))
        shade = rng.randint(135, 205)
        dust_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade, shade, shade, alpha))
    image.alpha_composite(dust_layer.filter(ImageFilter.GaussianBlur(0.35)))

    streak_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    streak_draw = ImageDraw.Draw(streak_layer)
    for _ in range(int(preset.grime_streak_count * alpha_scale)):
        x = rng.randrange(0, width)
        top = rng.randrange(0, max(1, height // 3))
        bottom = rng.randrange(max(top + 40, height // 2), height)
        line_width = rng.choice((1, 2, 2, 3))
        alpha = rng.randint(6, max(8, int(preset.grime_alpha * 0.45)))
        shade = rng.randint(120, 180)
        streak_draw.line((x, top, x, bottom), fill=(shade, shade, shade, alpha), width=line_width)
    image.alpha_composite(streak_layer.filter(ImageFilter.GaussianBlur(1.4)))

    smudge_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    smudge_draw = ImageDraw.Draw(smudge_layer)
    for _ in range(int(preset.grime_smudge_count * alpha_scale)):
        smudge_w = rng.randint(max(120, width // 12), max(180, width // 4))
        smudge_h = rng.randint(max(60, height // 18), max(120, height // 6))
        x0 = rng.randint(-smudge_w // 4, width - 1)
        y0 = rng.randint(-smudge_h // 4, height - 1)
        x1 = min(width + smudge_w // 4, x0 + smudge_w)
        y1 = min(height + smudge_h // 4, y0 + smudge_h)
        alpha = rng.randint(int(preset.grime_alpha * 0.25), max(1, int(preset.grime_alpha * 0.8)))
        if rng.random() < 0.55:
            fill = (rng.randint(18, 55), rng.randint(18, 55), rng.randint(18, 55), alpha)
        else:
            glow = rng.randint(105, 165)
            fill = (glow, glow, glow, alpha)
        smudge_draw.ellipse((x0, y0, x1, y1), fill=fill)
    image.alpha_composite(smudge_layer.filter(ImageFilter.GaussianBlur(12)))

    edge_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)
    edge_alpha = max(10, int(preset.grime_alpha * 0.55))
    edge_draw.rounded_rectangle(
        (18, 18, width - 18, height - 18),
        radius=24,
        outline=(0, 0, 0, edge_alpha),
        width=28,
    )
    image.alpha_composite(edge_layer.filter(ImageFilter.GaussianBlur(20)))

    image.save(path)


def _brightness_expression(preset: PostPreset, intensity: float) -> str:
    offset = preset.brightness_offset * intensity
    primary = preset.flicker_primary * intensity
    secondary = preset.flicker_secondary * intensity
    return (
        f"{offset:.5f}"
        f"+{primary:.5f}*sin(2*PI*t*0.37)"
        f"+{secondary:.5f}*sin(2*PI*t*2.11)"
    )


def _build_filter_complex(preset: PostPreset, intensity: float, texture_input: bool) -> str:
    tail_weight = preset.ghost_tail_weight * intensity
    ghost_opacity = _clamp(preset.ghost_opacity * intensity, 0.0, 0.95)
    ghost_scale = 1.0 + tail_weight
    blur_sigma = max(0.0, preset.ghost_blur_sigma * intensity)
    noise_strength = max(0, int(round(preset.noise_strength * intensity)))
    contrast = 1.0 + (preset.contrast - 1.0) * intensity
    saturation = 1.0 + (preset.saturation - 1.0) * intensity
    gamma_green = 1.0 + (preset.gamma_green - 1.0) * intensity

    filters = [
        (
            "[0:v]"
            f"eq=brightness='{_brightness_expression(preset, intensity)}':"
            f"contrast={contrast:.4f}:saturation={saturation:.4f}:gamma_g={gamma_green:.4f}:eval=frame,"
            f"curves=green='{preset.green_curve}',"
            f"chromashift=cbh={int(round(preset.chroma_blue_shift * intensity))}:"
            f"crh={int(round(preset.chroma_red_shift * intensity))}:edge=smear,"
            f"noise=alls={noise_strength}:allf=t+u,"
            "split=2[base][ghost]"
        ),
        (
            "[ghost]"
            f"tmix=frames=2:weights='1 {tail_weight:.4f}':scale={ghost_scale:.4f},"
            f"gblur=sigma={blur_sigma:.4f}"
            "[ghosted]"
        ),
        f"[base][ghosted]blend=all_mode=screen:all_opacity={ghost_opacity:.4f}[postghost]",
    ]

    current = "postghost"
    if texture_input:
        filters.append("[1:v]format=rgba[grime]")
        filters.append(
            f"[{current}][grime]"
            f"overlay=x='{preset.grime_drift_x}*sin(2*PI*t/23)':"
            f"y='{preset.grime_drift_y}*sin(2*PI*t/19)':format=auto"
            "[postgrime]"
        )
        current = "postgrime"

    filters.append(
        f"[{current}]vignette=angle={preset.vignette_angle}:x0=w/2:y0=h/2:eval=frame,format=yuv420p[vout]"
    )
    return ";".join(filters)


def default_output_path(tool_root: Path, input_path: Path, preset: PostPreset, start_seconds: float | None, duration_seconds: float | None) -> Path:
    output_dir = tool_root / "outputs" / "post"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = preset.name
    if start_seconds is not None or duration_seconds is not None:
        start_label = "full" if start_seconds is None else f"from-{start_seconds:0.1f}s"
        duration_label = "whole" if duration_seconds is None else f"for-{duration_seconds:0.1f}s"
        suffix = f"{suffix}.{start_label}.{duration_label}"
    return output_dir / f"{input_path.stem}.{suffix}.mp4"


def postprocess_video(
    *,
    tool_root: Path,
    input_path: Path,
    output_path: Path,
    preset: PostPreset,
    intensity: float = 1.0,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    seed: int = 17,
    crf: int = 17,
    ffmpeg_encode_preset: str = "slow",
) -> PostprocessResult:
    if not input_path.exists():
        raise FileNotFoundError(f"input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = _probe_dimensions(input_path)

    texture_path: Path | None = None
    texture_input = _grime_texture_needed(preset, intensity)
    if texture_input:
        texture_dir = tool_root / "outputs" / "post" / "_textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_path = texture_dir / f"{input_path.stem}.{preset.name}.png"
        _build_grime_texture(texture_path, width, height, preset, intensity, seed)

    filter_complex = _build_filter_complex(preset, intensity, texture_input)

    cmd = [ffmpeg_path(), "-y"]
    if start_seconds is not None:
        cmd.extend(["-ss", f"{max(0.0, start_seconds):0.3f}"])
    cmd.extend(["-i", str(input_path)])
    if texture_path is not None:
        cmd.extend(["-loop", "1", "-i", str(texture_path)])
    if duration_seconds is not None:
        cmd.extend(["-t", f"{max(0.1, duration_seconds):0.3f}"])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", ffmpeg_encode_preset,
        "-crf", str(crf),
        "-movflags", "+faststart",
        "-c:a", "copy",
        str(output_path),
    ])

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg postprocess failed:\n{result.stderr[-4000:]}")

    report_path = output_path.with_suffix(".report.json")
    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "preset": preset.name,
        "preset_description": preset.description,
        "intensity": intensity,
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "texture_path": str(texture_path) if texture_path else "",
        "width": width,
        "height": height,
        "ffmpeg_preset": ffmpeg_encode_preset,
        "crf": crf,
        "preset_values": asdict(preset),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PostprocessResult(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        preset=preset.name,
        intensity=intensity,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        texture_path=texture_path,
        width=width,
        height=height,
    )

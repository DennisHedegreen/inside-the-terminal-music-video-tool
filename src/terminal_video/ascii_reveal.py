from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .effects import draw_noise, draw_scanlines
from .mux import encode_raw_video


@dataclass(frozen=True)
class AsciiRevealResult:
    output_mp4: Path
    preview_png: Path
    frame_count: int


def hex_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def fit_with_letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def ease_in_out(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def lerp_pair(start: tuple[float, float], end: tuple[float, float], t: float) -> tuple[float, float]:
    return (lerp(start[0], end[0], t), lerp(start[1], end[1], t))


def fit_cover_with_motion(
    image: Image.Image,
    size: tuple[int, int],
    zoom: float = 1.0,
    pan: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    source = image.convert("RGBA")
    target_w, target_h = size
    scale = max(target_w / max(1, source.width), target_h / max(1, source.height)) * max(1.0, zoom)
    resized_w = max(target_w, int(round(source.width * scale)))
    resized_h = max(target_h, int(round(source.height * scale)))
    resized = source.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    overflow_x = max(0, resized_w - target_w)
    overflow_y = max(0, resized_h - target_h)
    pan_x = max(-1.0, min(1.0, pan[0]))
    pan_y = max(-1.0, min(1.0, pan[1]))
    left = int(round((pan_x + 1.0) * 0.5 * overflow_x))
    top = int(round((pan_y + 1.0) * 0.5 * overflow_y))
    return resized.crop((left, top, left + target_w, top + target_h))


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def build_priority_map(gray: np.ndarray) -> np.ndarray:
    inverted = 1.0 - gray
    grad_y, grad_x = np.gradient(gray)
    edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edges = normalize(edges)

    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dist = np.sqrt(((xx - cx) / max(1.0, width / 2.0)) ** 2 + ((yy - cy) / max(1.0, height / 2.0)) ** 2)
    center_weight = 1.0 - np.clip(dist, 0.0, 1.0)
    return (inverted * 0.48 + edges * 0.38 + center_weight * 0.14).astype(np.float32)


def glyph_for_intensity(intensity: float, glyphs: str) -> str:
    intensity = max(0.0, min(1.0, intensity))
    index = int(round(intensity * (len(glyphs) - 1)))
    return glyphs[index]


def prepare_ascii_cells(config: dict) -> tuple[dict, list[dict]]:
    style = config["style"]
    image_path = Path(config["source_image"])
    source = Image.open(image_path).convert("RGB")

    portrait_width = int(style["portrait_width"])
    portrait_height = int(style["portrait_height"])
    cell_width = int(style["cell_width"])
    cell_height = int(style["cell_height"])
    glyphs = style["glyphs"]

    portrait = fit_with_letterbox(source, (portrait_width, portrait_height))
    gray = np.asarray(portrait.convert("L"), dtype=np.float32) / 255.0
    priority = build_priority_map(gray)

    cols = portrait_width // cell_width
    rows = portrait_height // cell_height
    cells: list[dict] = []
    rng = random.Random(int(config.get("seed", 42)))

    for row in range(rows):
        for col in range(cols):
            x0 = col * cell_width
            y0 = row * cell_height
            x1 = min(portrait_width, x0 + cell_width)
            y1 = min(portrait_height, y0 + cell_height)
            block_gray = gray[y0:y1, x0:x1]
            block_priority = priority[y0:y1, x0:x1]
            intensity = 1.0 - float(block_gray.mean())
            score = float(block_priority.mean()) + rng.random() * 0.015
            cells.append(
                {
                    "col": col,
                    "row": row,
                    "intensity": intensity,
                    "score": score,
                    "glyph": glyph_for_intensity(intensity, glyphs),
                }
            )

    cells.sort(key=lambda item: item["score"], reverse=True)
    metadata = {
        "portrait_width": portrait_width,
        "portrait_height": portrait_height,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "cols": cols,
        "rows": rows,
    }
    return metadata, cells


def build_ascii_frame(
    *,
    config: dict,
    metadata: dict,
    cells: list[dict],
    visible_percentage: float,
    frame_index: int,
    frame_count: int,
    shot_index: int,
    total_shots: int,
    status_label: str,
) -> Image.Image:
    video = config["video"]
    style = config["style"]
    width = int(video["width"])
    height = int(video["height"])

    cols = metadata["cols"]
    rows = metadata["rows"]
    cell_width = metadata["cell_width"]
    cell_height = metadata["cell_height"]
    portrait_width = metadata["portrait_width"]
    portrait_height = metadata["portrait_height"]

    font_size = int(style["font_size"])
    font = ImageFont.truetype(style["font_regular"], font_size)
    title_font = ImageFont.truetype(style["font_bold"], int(style["title_font_size"]))
    bg = hex_color(style["background"])
    green = hex_color(style["green"])
    green_soft = hex_color(style["green_soft"])
    green_dim = hex_color(style["green_dim"])

    portrait_center_x = width // 2
    portrait_center_y = height // 2 - 10
    title = str(config["title"]).upper()
    max_visible = len(cells)
    visible_fraction = max(0.0, min(1.0, float(visible_percentage) / 100.0))
    visible_count = max(1, int(round(max_visible * visible_fraction)))
    visible_lookup = set(range(visible_count))

    progress = frame_index / max(1, frame_count - 1)
    zoom = 0.97 + progress * 0.06
    vertical_drift = -12 + progress * 16
    pulse = 0.55 + 0.45 * math.sin(frame_index / max(1, int(video["fps"])) * 2.8)

    image = Image.new("RGBA", (width, height), (*bg, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    border_rect = (156, 92, width - 156, height - 124)
    draw.rounded_rectangle(border_rect, radius=18, outline=(*green, 200), width=2)
    draw.text((border_rect[0] + 24, border_rect[1] + 16), title, font=title_font, fill=(*green_dim, 220))
    draw.text((border_rect[2] - 220, border_rect[1] + 18), f"step {shot_index + 1:02d}/{total_shots:02d}", font=font, fill=(*green_dim, 180))

    glow_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_overlay, "RGBA")
    glow_alpha = int(38 + pulse * 28)
    glow_draw.rounded_rectangle(border_rect, radius=18, outline=(*green, glow_alpha), width=8)
    image.alpha_composite(glow_overlay.filter(ImageFilter.GaussianBlur(18)))

    step_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    step_draw = ImageDraw.Draw(step_layer, "RGBA")

    scaled_cell_width = cell_width * zoom
    scaled_cell_height = cell_height * zoom
    scaled_width = portrait_width * zoom
    scaled_height = portrait_height * zoom
    left = portrait_center_x - scaled_width / 2
    top = portrait_center_y - scaled_height / 2 + vertical_drift

    for idx, cell in enumerate(cells):
        if idx not in visible_lookup:
            continue
        x = left + cell["col"] * scaled_cell_width
        y = top + cell["row"] * scaled_cell_height
        brightness = 0.42 + cell["intensity"] * 0.58
        if idx > visible_count - max(6, int(max_visible * 0.02)):
            color = green_soft
            alpha = int(140 + brightness * 90)
        else:
            color = green
            alpha = int(90 + brightness * 105)
        step_draw.text((x, y), cell["glyph"], font=font, fill=(*color, min(255, alpha)))

    image.alpha_composite(step_layer)

    draw.text((border_rect[0] + 24, border_rect[3] - 46), status_label, font=font, fill=(*green_dim, 200))
    draw.text((border_rect[2] - 198, border_rect[3] - 46), "nero ascii reveal", font=font, fill=(*green_dim, 160))

    scan = Image.new("RGBA", image.size, (0, 0, 0, 0))
    scan_draw = ImageDraw.Draw(scan)
    for y in range(0, height, 4):
        scan_draw.line((0, y, width, y), fill=(0, 0, 0, 26), width=1)
    image.alpha_composite(scan)
    draw_noise(image, 10, frame_index)
    return image


def build_ascii_plate(
    *,
    config: dict,
    metadata: dict,
    cells: list[dict],
    visible_percentage: float,
    transparent_background: bool = True,
    zoom: float = 1.0,
    vertical_offset: float = 0.0,
    glow_strength: float = 0.9,
) -> Image.Image:
    video = config["video"]
    style = config["style"]
    width = int(video["width"])
    height = int(video["height"])

    cell_width = metadata["cell_width"]
    cell_height = metadata["cell_height"]
    portrait_width = metadata["portrait_width"]
    portrait_height = metadata["portrait_height"]

    font_size = int(style["font_size"])
    font = ImageFont.truetype(style["font_regular"], font_size)
    green = hex_color(style["green"])
    green_soft = hex_color(style["green_soft"])
    bg = hex_color(style["background"])

    max_visible = len(cells)
    visible_fraction = max(0.0, min(1.0, float(visible_percentage) / 100.0))
    visible_count = max(1, int(round(max_visible * visible_fraction)))
    visible_lookup = set(range(visible_count))

    if transparent_background:
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (width, height), (*bg, 255))

    glyph_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glyph_draw = ImageDraw.Draw(glyph_layer, "RGBA")

    portrait_center_x = width // 2
    portrait_center_y = height // 2 + int(vertical_offset)
    scaled_cell_width = cell_width * zoom
    scaled_cell_height = cell_height * zoom
    scaled_width = portrait_width * zoom
    scaled_height = portrait_height * zoom
    left = portrait_center_x - scaled_width / 2
    top = portrait_center_y - scaled_height / 2

    tail_start = max(0, visible_count - max(8, int(max_visible * 0.025)))
    for idx, cell in enumerate(cells):
        if idx not in visible_lookup:
            continue
        x = left + cell["col"] * scaled_cell_width
        y = top + cell["row"] * scaled_cell_height
        brightness = 0.44 + cell["intensity"] * 0.56
        if idx >= tail_start:
            color = green_soft
            alpha = int(165 + brightness * 90)
        else:
            color = green
            alpha = int(118 + brightness * 92)
        glyph_draw.text((x, y), cell["glyph"], font=font, fill=(*color, min(255, alpha)))

    if glow_strength > 0.0:
        glow = glyph_layer.filter(ImageFilter.GaussianBlur(max(2, int(10 * glow_strength))))
        glow_alpha = glow.getchannel("A").point(lambda value: int(value * min(1.0, glow_strength) * 0.72))
        glow.putalpha(glow_alpha)
        image.alpha_composite(glow)
    image.alpha_composite(glyph_layer)
    return image


def build_cutaway_frame(
    *,
    config: dict,
    asset_path: Path,
    progress: float,
    label: str,
    zoom_start: float,
    zoom_end: float,
    pan_start: tuple[float, float],
    pan_end: tuple[float, float],
    flash_strength: float,
) -> Image.Image:
    video = config["video"]
    style = config["style"]
    width = int(video["width"])
    height = int(video["height"])
    font = ImageFont.truetype(style["font_regular"], int(style["font_size"]))
    title_font = ImageFont.truetype(style["font_bold"], int(style["title_font_size"]))
    bg = hex_color(style["background"])
    green = hex_color(style["green"])
    green_dim = hex_color(style["green_dim"])

    eased = ease_in_out(progress)
    zoom = lerp(zoom_start, zoom_end, eased)
    pan = lerp_pair(pan_start, pan_end, eased)
    base_asset = Image.open(asset_path).convert("RGBA")
    frame = fit_cover_with_motion(base_asset, (width, height), zoom=zoom, pan=pan)

    vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    vignette_draw.rectangle((0, 0, width, height), fill=(0, 0, 0, 24))
    vignette_draw.rounded_rectangle((48, 40, width - 48, height - 40), radius=40, outline=(*green, 36), width=2)
    frame.alpha_composite(vignette.filter(ImageFilter.GaussianBlur(22)))

    flash_phase = math.sin(progress * math.pi)
    flash_alpha = int(max(0.0, flash_strength) * flash_phase * 120)
    if flash_alpha > 0:
        flash = Image.new("RGBA", (width, height), (*green, flash_alpha))
        frame.alpha_composite(flash)

    draw_scanlines(frame, 20)
    draw_noise(frame, 8, int(progress * 1000) + len(label))

    draw = ImageDraw.Draw(frame, "RGBA")
    draw.text((84, 68), label.upper(), font=title_font, fill=(*green_dim, 180))
    draw.text((84, height - 72), "archive / sorting / discovery", font=font, fill=(*green_dim, 160))
    return frame


def render_ascii_reveal(config: dict, output_dir: Path) -> AsciiRevealResult:
    video = config["video"]
    style = config["style"]
    width = int(video["width"])
    height = int(video["height"])
    fps = int(video["fps"])
    crf = int(video.get("crf", 20))
    preset = str(video.get("preset", "veryfast"))
    slug = config["slug"]

    step_seconds = float(config["step_seconds"])
    visible_percentages = list(config["visible_percentages"])
    frame_count = int(round(len(visible_percentages) * step_seconds * fps))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug}.mp4"
    preview_path = output_dir / f"{slug}.preview.png"

    metadata, cells = prepare_ascii_cells(config)
    shot_frames = max(1, int(round(step_seconds * fps)))

    process = encode_raw_video(
        raw_frame_pipe=subprocess.PIPE,
        raw_video_path=output_path,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable for ascii reveal")

    preview_image: Image.Image | None = None
    try:
        for frame_index in range(frame_count):
            shot_index = min(len(visible_percentages) - 1, frame_index // shot_frames)
            image = build_ascii_frame(
                config=config,
                metadata=metadata,
                cells=cells,
                visible_percentage=float(visible_percentages[shot_index]),
                frame_index=frame_index,
                frame_count=frame_count,
                shot_index=shot_index,
                total_shots=len(visible_percentages),
                status_label=f"visible glyphs: {int(float(visible_percentages[shot_index])):02d}%",
            )

            if frame_index == frame_count // 2:
                preview_image = image.copy().convert("RGB")

            process.stdin.write(image.convert("RGB").tobytes())
    finally:
        process.stdin.close()

    stderr = process.stderr.read() if process.stderr else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed for ascii reveal:\n{stderr.decode(errors='ignore')[-4000:]}")

    if preview_image is None:
        preview_image = Image.new("RGB", (width, height), bg)
    preview_image.save(preview_path)
    return AsciiRevealResult(output_mp4=output_path, preview_png=preview_path, frame_count=frame_count)


def render_ascii_plates(config: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, cells = prepare_ascii_cells(config)
    visible_percentages = list(config["visible_percentages"])
    slug = str(config["slug"])
    zoom = float(config.get("plate_zoom", 1.0))
    vertical_offset = float(config.get("plate_vertical_offset", 0.0))
    glow_strength = float(config.get("plate_glow_strength", 0.9))
    transparent_background = bool(config.get("transparent_background", True))

    generated_paths: list[Path] = []
    preview_path = output_dir / f"{slug}.preview.png"
    preview_image: Image.Image | None = None

    for value in visible_percentages:
        plate = build_ascii_plate(
            config=config,
            metadata=metadata,
            cells=cells,
            visible_percentage=float(value),
            transparent_background=transparent_background,
            zoom=zoom,
            vertical_offset=vertical_offset,
            glow_strength=glow_strength,
        )
        safe_value = f"{int(round(float(value))):03d}"
        output_path = output_dir / f"{slug}-{safe_value}.png"
        plate.save(output_path)
        generated_paths.append(output_path)
        if int(round(float(value))) >= 100 and preview_image is None:
            preview_image = plate.copy()

    if preview_image is None and generated_paths:
        with Image.open(generated_paths[-1]) as image:
            preview_image = image.convert("RGBA")
    if preview_image is None:
        preview_image = Image.new("RGBA", (int(config["video"]["width"]), int(config["video"]["height"])), (0, 0, 0, 0))
    preview_image.save(preview_path)

    return {
        "preview_png": preview_path,
        "generated_paths": generated_paths,
        "frame_count": len(generated_paths),
    }


def render_ascii_hybrid(config: dict, output_dir: Path) -> AsciiRevealResult:
    video = config["video"]
    width = int(video["width"])
    height = int(video["height"])
    fps = int(video["fps"])
    crf = int(video.get("crf", 20))
    preset = str(video.get("preset", "veryfast"))
    slug = config["slug"]
    sequence = list(config.get("sequence", []))
    if not sequence:
        raise ValueError("hybrid ascii config requires a non-empty sequence")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug}.mp4"
    preview_path = output_dir / f"{slug}.preview.png"

    metadata, cells = prepare_ascii_cells(config)
    visible_percentages = list(config["visible_percentages"])

    frame_specs: list[dict] = []
    for scene in sequence:
        scene_type = str(scene.get("type", "")).strip()
        if scene_type == "ascii_steps":
            step_start = int(scene["step_start"])
            step_end = int(scene["step_end"])
            step_seconds = float(scene.get("step_seconds", config.get("step_seconds", 0.3)))
            hold_seconds = float(scene.get("hold_seconds", 0.0))
            for step_index in range(step_start, step_end + 1):
                frame_specs.append(
                    {
                        "type": "ascii",
                        "step_index": step_index - 1,
                        "duration_frames": max(1, int(round(step_seconds * fps))),
                    }
                )
            if hold_seconds > 0.0:
                frame_specs.append(
                    {
                        "type": "ascii_hold",
                        "step_index": step_end - 1,
                        "duration_frames": max(1, int(round(hold_seconds * fps))),
                    }
                )
        elif scene_type == "image_glimpse":
            frame_specs.append(
                {
                    "type": "glimpse",
                    "asset_path": Path(scene["asset_path"]),
                    "duration_frames": max(1, int(round(float(scene["duration_seconds"]) * fps))),
                    "zoom_start": float(scene.get("zoom_start", 1.0)),
                    "zoom_end": float(scene.get("zoom_end", 1.08)),
                    "pan_start": tuple(scene.get("pan_start", [0.0, 0.0])),
                    "pan_end": tuple(scene.get("pan_end", [0.0, 0.0])),
                    "label": str(scene.get("label", "glimpse")),
                    "flash_strength": float(scene.get("flash_strength", 0.18)),
                }
            )
        else:
            raise ValueError(f"unsupported hybrid scene type: {scene_type}")

    frame_count = sum(int(spec["duration_frames"]) for spec in frame_specs)
    process = encode_raw_video(
        raw_frame_pipe=subprocess.PIPE,
        raw_video_path=output_path,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable for ascii hybrid")

    preview_image: Image.Image | None = None
    absolute_frame = 0
    total_ascii_steps = len(visible_percentages)
    try:
        for spec in frame_specs:
            local_frames = int(spec["duration_frames"])
            for local_index in range(local_frames):
                local_progress = local_index / max(1, local_frames - 1)
                if spec["type"] in {"ascii", "ascii_hold"}:
                    step_index = int(spec["step_index"])
                    image = build_ascii_frame(
                        config=config,
                        metadata=metadata,
                        cells=cells,
                        visible_percentage=float(visible_percentages[step_index]),
                        frame_index=absolute_frame,
                        frame_count=frame_count,
                        shot_index=step_index,
                        total_shots=total_ascii_steps,
                        status_label=f"visible glyphs: {int(float(visible_percentages[step_index])):02d}%",
                    )
                else:
                    image = build_cutaway_frame(
                        config=config,
                        asset_path=Path(spec["asset_path"]),
                        progress=local_progress,
                        label=str(spec["label"]),
                        zoom_start=float(spec["zoom_start"]),
                        zoom_end=float(spec["zoom_end"]),
                        pan_start=(float(spec["pan_start"][0]), float(spec["pan_start"][1])),
                        pan_end=(float(spec["pan_end"][0]), float(spec["pan_end"][1])),
                        flash_strength=float(spec["flash_strength"]),
                    )
                if absolute_frame == frame_count // 2:
                    preview_image = image.copy().convert("RGB")
                process.stdin.write(image.convert("RGB").tobytes())
                absolute_frame += 1
    finally:
        process.stdin.close()

    stderr = process.stderr.read() if process.stderr else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed for ascii hybrid:\n{stderr.decode(errors='ignore')[-4000:]}")

    if preview_image is None:
        preview_image = Image.new("RGB", (width, height), hex_color(config["style"]["background"]))
    preview_image.save(preview_path)
    return AsciiRevealResult(output_mp4=output_path, preview_png=preview_path, frame_count=frame_count)

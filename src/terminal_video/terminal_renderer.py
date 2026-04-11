from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .audio_analyzer import AudioAnalysis
from .effects import EffectState, draw_noise, effect_state
from .lyrics_timeline import LineEvent, SceneEvent, Timeline


@dataclass(frozen=True)
class RenderedFrame:
    image: Image.Image
    mode: str


def hex_color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


class TerminalRenderer:
    def __init__(self, config: dict, analysis: AudioAnalysis, timeline: Timeline) -> None:
        self.config = config
        self.analysis = analysis
        self.timeline = timeline
        video = config["video"]
        style = config["style"]
        self.width = int(video["width"])
        self.height = int(video["height"])
        self.font = ImageFont.truetype(style["font_regular"], int(style["font_size"]))
        self.bold = ImageFont.truetype(style["font_bold"], int(style["font_size"]))
        self.small = ImageFont.truetype(style["font_regular"], int(style["small_font_size"]))
        self.title_font = ImageFont.truetype(style["font_bold"], int(style["title_font_size"]))
        self.bg = hex_color(style["background"])
        self.green = hex_color(style["green"])
        self.green_soft = hex_color(style["green_soft"])
        self.green_dim = hex_color(style["green_dim"])
        self.text = hex_color(style["text"])
        self.text_dim = hex_color(style["text_dim"])
        self.warning = hex_color(style["warning"])
        self.border = hex_color(style["terminal_border"])
        self.presentation = config.get("presentation", {})
        self.config_dir = Path(config.get("__config_dir__", "."))
        self._glow_cache: dict[tuple[tuple[int, int, int, int], int], Image.Image] = {}
        self._scanline_cache: dict[int, Image.Image] = {}
        self._asset_cache: dict[Path, Image.Image] = {}
        self._plane_layer_cache: dict[tuple, tuple[Image.Image | None, Image.Image | None, Image.Image]] = {}
        self.world_objects = {item["id"]: item for item in config.get("objects", []) if isinstance(item, dict) and item.get("id")}
        self.world_scenes = [item for item in config.get("scenes", []) if isinstance(item, dict)]
        self.text_layers = {item["id"]: item for item in config.get("text_layers", []) if isinstance(item, dict) and item.get("id")}
        self.ambient_layers = {item["id"]: item for item in config.get("ambient_layers", []) if isinstance(item, dict) and item.get("id")}

    def presentation_flag(self, key: str, default: bool) -> bool:
        value = self.presentation.get(key, default)
        if isinstance(value, str):
            return value.lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    def presentation_text(self, key: str, default: str = "") -> str:
        value = self.presentation.get(key, default)
        return str(value) if value is not None else ""

    def color_for_role(self, role: str) -> tuple[int, int, int]:
        return {
            "green": self.green_soft,
            "green_soft": self.green_soft,
            "green_dim": self.green_dim,
            "text": self.text,
            "text_dim": self.text_dim,
            "warning": self.warning,
            "border": self.border,
        }.get(role, self.green_soft)

    def scene_at(self, seconds: float) -> SceneEvent:
        for scene in self.timeline.scenes:
            if scene.start <= seconds < scene.end:
                return scene
        return self.timeline.scenes[-1]

    def world_scene_at(self, seconds: float) -> dict | None:
        for scene in self.world_scenes:
            start = float(scene.get("start", 0.0))
            end = float(scene.get("end", self.timeline.duration))
            if start <= seconds < end:
                return scene
        return None

    @staticmethod
    def lerp(start: float, end: float, ratio: float) -> float:
        return start + (end - start) * ratio

    def scene_progress(self, seconds: float, scene: dict | None) -> float:
        if not scene:
            return 0.0
        start = float(scene.get("start", 0.0))
        end = float(scene.get("end", start + 1.0))
        if end <= start:
            return 0.0
        return max(0.0, min(1.0, (seconds - start) / (end - start)))

    def scene_camera_state(self, seconds: float, scene: dict | None) -> dict:
        if not scene:
            return {"zoom": 1.0, "pan": (0.0, 0.0)}
        camera = scene.get("camera", {})
        progress = self.scene_progress(seconds, scene)
        pan_start = camera.get("pan_start", camera.get("pan", [0.0, 0.0]))
        pan_end = camera.get("pan_end", pan_start)
        while len(pan_start) < 2:
            pan_start.append(0.0)
        while len(pan_end) < 2:
            pan_end.append(0.0)
        zoom_start = float(camera.get("zoom_start", camera.get("zoom", 1.0)))
        zoom_end = float(camera.get("zoom_end", zoom_start))
        return {
            "zoom": self.lerp(zoom_start, zoom_end, progress),
            "pan": (
                self.lerp(float(pan_start[0]), float(pan_end[0]), progress),
                self.lerp(float(pan_start[1]), float(pan_end[1]), progress),
            ),
        }

    def scene_terminal_mode(self, scene: dict | None) -> str:
        if not scene:
            return "full"
        return str(scene.get("terminal_mode", "full"))

    def ambient_bounds(self, layer: dict, terminal_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        region = str(layer.get("region", "terminal"))
        if region == "screen":
            base_rect = (0, 0, self.width, self.height)
        elif region == "terminal_inner":
            inset = int(layer.get("region_inset", 28))
            base_rect = (
                terminal_rect[0] + inset,
                terminal_rect[1] + inset,
                terminal_rect[2] - inset,
                terminal_rect[3] - inset,
            )
        else:
            base_rect = terminal_rect
        rect = layer.get("rect", [0.0, 0.0, 1.0, 1.0])
        if not isinstance(rect, list) or len(rect) < 4:
            rect = [0.0, 0.0, 1.0, 1.0]
        bx0, by0, bx1, by1 = base_rect
        bw = max(1, bx1 - bx0)
        bh = max(1, by1 - by0)
        x0 = bx0 + int(float(rect[0]) * bw)
        y0 = by0 + int(float(rect[1]) * bh)
        x1 = bx0 + int(float(rect[2]) * bw)
        y1 = by0 + int(float(rect[3]) * bh)
        return (
            max(0, min(x0, self.width - 1)),
            max(0, min(y0, self.height - 1)),
            max(1, min(x1, self.width)),
            max(1, min(y1, self.height)),
        )

    @staticmethod
    def grid_route_points(waypoints: list[list[int]] | None, cols: int, rows: int, closed: bool = True) -> list[tuple[int, int]]:
        if not waypoints:
            return []
        cleaned: list[tuple[int, int]] = []
        for item in waypoints:
            if not isinstance(item, list) or len(item) < 2:
                continue
            cleaned.append((max(0, min(cols - 1, int(item[0]))), max(0, min(rows - 1, int(item[1])))))
        if not cleaned:
            return []
        route = [cleaned[0]]
        targets = cleaned[1:] + ([cleaned[0]] if closed else [])
        current_x, current_y = cleaned[0]
        for target_x, target_y in targets:
            while current_x != target_x:
                current_x += 1 if target_x > current_x else -1
                if route[-1] != (current_x, current_y):
                    route.append((current_x, current_y))
            while current_y != target_y:
                current_y += 1 if target_y > current_y else -1
                if route[-1] != (current_x, current_y):
                    route.append((current_x, current_y))
        return route

    @staticmethod
    def default_snake_waypoints(cols: int, rows: int) -> list[list[int]]:
        return [
            [1, max(1, rows - 3)],
            [max(2, cols - 3), max(1, rows - 3)],
            [max(2, cols - 3), max(1, rows // 2)],
            [max(2, cols // 2), max(1, rows // 2)],
            [max(2, cols // 2), 1],
            [max(1, cols // 3), 1],
            [max(1, cols // 3), max(1, rows - 4)],
            [1, max(1, rows - 4)],
        ]

    def animated_object(self, obj: dict, seconds: float, scene: dict | None) -> dict:
        motion = obj.get("motion", {})
        if not isinstance(motion, dict) or not motion:
            return obj

        render_obj = dict(obj)
        position = list(render_obj.get("position", [0.0, 0.0, 3.0]))
        while len(position) < 3:
            position.append(0.0)
        scale = list(render_obj.get("scale", [1.0, 1.0]))
        while len(scale) < 2:
            scale.append(scale[0] if scale else 1.0)
        rotation = list(render_obj.get("rotation", [0.0, 0.0, 0.0]))
        while len(rotation) < 3:
            rotation.append(0.0)

        progress = self.scene_progress(seconds, scene)
        phase = float(motion.get("phase", 0.0))
        oscillate_rate = float(motion.get("oscillate_rate", motion.get("pulse_rate", 0.45)))
        pulse = math.sin((seconds + phase) * math.tau * oscillate_rate)
        beat = self.analysis.beat_proximity(seconds, window=0.10)

        drift_start = motion.get("drift_start")
        drift_end = motion.get("drift_end")
        if isinstance(drift_start, list) and isinstance(drift_end, list):
            while len(drift_start) < 2:
                drift_start.append(0.0)
            while len(drift_end) < 2:
                drift_end.append(0.0)
            position[0] += self.lerp(float(drift_start[0]), float(drift_end[0]), progress)
            position[1] += self.lerp(float(drift_start[1]), float(drift_end[1]), progress)
        else:
            drift = motion.get("drift")
            if isinstance(drift, list):
                while len(drift) < 2:
                    drift.append(0.0)
                position[0] += self.lerp(0.0, float(drift[0]), progress)
                position[1] += self.lerp(0.0, float(drift[1]), progress)

        oscillate = motion.get("oscillate")
        if isinstance(oscillate, list):
            while len(oscillate) < 2:
                oscillate.append(0.0)
            position[0] += float(oscillate[0]) * pulse
            position[1] += float(oscillate[1]) * math.cos((seconds + phase) * math.tau * oscillate_rate)

        z_drift = motion.get("z_drift")
        if isinstance(z_drift, (int, float)):
            position[2] += float(z_drift) * progress

        pulse_scale = float(motion.get("pulse_scale", 0.0))
        beat_scale = float(motion.get("beat_scale", 0.0))
        scale_mult = 1.0 + max(0.0, pulse) * pulse_scale + beat * beat_scale
        scale[0] *= scale_mult
        scale[1] *= scale_mult

        rotation_wobble = motion.get("rotation_wobble")
        if isinstance(rotation_wobble, list):
            while len(rotation_wobble) < 3:
                rotation_wobble.append(0.0)
            rotation[0] += float(rotation_wobble[0]) * pulse
            rotation[1] += float(rotation_wobble[1]) * math.cos((seconds + phase) * math.tau * oscillate_rate)
            rotation[2] += float(rotation_wobble[2]) * pulse

        opacity = float(render_obj.get("opacity", 1.0))
        flicker = float(motion.get("flicker", 0.0))
        if flicker > 0.0:
            opacity *= max(0.0, min(1.0, 1.0 - flicker + flicker * (0.55 + 0.45 * max(0.0, pulse))))
        beat_opacity = float(motion.get("beat_opacity", 0.0))
        if beat_opacity > 0.0:
            opacity *= max(0.0, min(1.0, 1.0 + beat * beat_opacity))
        render_obj["opacity"] = max(0.0, min(1.0, opacity))

        glow_strength = float(render_obj.get("glow_strength", 0.0))
        beat_glow = float(motion.get("beat_glow", 0.0))
        if beat_glow > 0.0:
            render_obj["glow_strength"] = max(0.0, glow_strength + beat * beat_glow)

        render_obj["position"] = position
        render_obj["scale"] = scale
        render_obj["rotation"] = rotation
        return render_obj

    def active_lines(self, seconds: float, scene: SceneEvent) -> tuple[list[tuple[LineEvent, str]], str]:
        visible: list[tuple[LineEvent, str]] = []
        cursor_text = ""
        for event in self.timeline.lines:
            if event.scene_id != scene.scene_id:
                continue
            if seconds < event.start:
                continue
            if seconds >= event.end:
                visible.append((event, event.text))
                continue
            ratio = (seconds - event.start) / max(0.01, event.end - event.start)
            char_count = max(1, int(len(event.text) * ratio))
            text = event.text[:char_count]
            visible.append((event, text))
            cursor_text = text
            break
        return visible[-10:], cursor_text

    def wrap_line(self, text: str, width: int = 68) -> list[str]:
        if not text:
            return [""]
        return textwrap.wrap(text, width=width, break_long_words=False) or [text]

    def resolve_asset_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.config_dir / path).resolve()

    def load_asset(self, value: str) -> Image.Image | None:
        try:
            path = self.resolve_asset_path(value)
        except Exception:
            return None
        cached = self._asset_cache.get(path)
        if cached is not None:
            return cached
        try:
            with Image.open(path) as image:
                cached = image.convert("RGBA")
        except Exception:
            return None
        self._asset_cache[path] = cached
        return cached

    def world_to_screen(self, position: list[float], scale: list[float], camera_state: dict | None = None) -> tuple[int, int, int, int]:
        x = float(position[0]) if len(position) > 0 else 0.0
        y = float(position[1]) if len(position) > 1 else 0.0
        z = max(1.0, float(position[2]) if len(position) > 2 else 3.0)
        sx = float(scale[0]) if len(scale) > 0 else 1.0
        sy = float(scale[1]) if len(scale) > 1 else sx
        camera_state = camera_state or {"zoom": 1.0, "pan": (0.0, 0.0)}
        zoom = float(camera_state.get("zoom", 1.0))
        pan_x, pan_y = camera_state.get("pan", (0.0, 0.0))
        depth_scale = (3.0 / z) * zoom
        width = max(120, int(self.width * 0.26 * sx * depth_scale))
        height = max(80, int(self.height * 0.24 * sy * depth_scale))
        center_x = int(self.width * 0.5 + (x + float(pan_x)) * 320 * depth_scale)
        center_y = int(self.height * 0.5 - (y + float(pan_y)) * 240 * depth_scale)
        return (
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2,
        )

    def transform_plane(self, asset: Image.Image, obj: dict, rect: tuple[int, int, int, int]) -> Image.Image:
        width = max(8, rect[2] - rect[0])
        height = max(8, rect[3] - rect[1])
        fit_mode = obj.get("fit_mode", "cover")
        if fit_mode == "contain":
            plane = ImageOps.contain(asset, (width, height))
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            canvas.alpha_composite(plane, ((width - plane.width) // 2, (height - plane.height) // 2))
            plane = canvas
        else:
            plane = ImageOps.fit(asset, (width, height), method=Image.Resampling.LANCZOS)

        rotation = obj.get("rotation", [0.0, 0.0, 0.0])
        rx = math.radians(float(rotation[0]) if len(rotation) > 0 else 0.0)
        ry = math.radians(float(rotation[1]) if len(rotation) > 1 else 0.0)
        rz = float(rotation[2]) if len(rotation) > 2 else 0.0
        x_squash = max(0.12, abs(math.cos(ry)))
        y_squash = max(0.2, abs(math.cos(rx)))
        transformed = plane.resize(
            (max(8, int(width * x_squash)), max(8, int(height * y_squash))),
            Image.Resampling.LANCZOS,
        )
        if abs(rz) > 0.1:
            transformed = transformed.rotate(rz, resample=Image.Resampling.BICUBIC, expand=True)

        opacity = float(obj.get("opacity", 1.0))
        if opacity < 1.0:
            alpha = transformed.getchannel("A").point(lambda value: int(value * opacity))
            transformed.putalpha(alpha)
        return transformed

    def draw_plane_shadow(self, image: Image.Image, obj: dict, transformed: Image.Image, center: tuple[int, int]) -> None:
        shadow_strength = float(obj.get("shadow_strength", 0.0))
        if shadow_strength <= 0.0:
            return
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        alpha = transformed.getchannel("A").point(lambda value: int(value * min(0.9, shadow_strength) * 0.55))
        shadow = Image.new("RGBA", transformed.size, (0, 0, 0, 0))
        shadow.putalpha(alpha)
        left = int(center[0] - transformed.width / 2 + 18)
        top = int(center[1] - transformed.height / 2 + 22)
        layer.alpha_composite(shadow, (left, top))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(18)))

    def draw_plane_glow(self, image: Image.Image, obj: dict, transformed: Image.Image, center: tuple[int, int]) -> None:
        glow_strength = float(obj.get("glow_strength", 0.0))
        if glow_strength <= 0.0:
            return
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        alpha = transformed.getchannel("A").point(lambda value: int(value * min(1.0, glow_strength) * 0.35))
        glow = Image.new("RGBA", transformed.size, (*self.green, 0))
        glow.putalpha(alpha)
        left = int(center[0] - transformed.width / 2)
        top = int(center[1] - transformed.height / 2)
        layer.alpha_composite(glow, (left, top))
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(24)))

    def draw_image_plane(self, image: Image.Image, obj: dict, camera_state: dict | None = None, scene: dict | None = None, seconds: float = 0.0) -> None:
        obj = self.animated_object(obj, seconds, scene)
        asset_path = obj.get("asset_path")
        if not asset_path:
            return
        asset = self.load_asset(asset_path)
        if asset is None:
            return
        rect = self.world_to_screen(obj.get("position", [0.0, 0.0, 3.0]), obj.get("scale", [1.0, 1.0]), camera_state)
        rotation = tuple(float(value) for value in obj.get("rotation", [0.0, 0.0, 0.0]))
        scale = tuple(float(value) for value in obj.get("scale", [1.0, 1.0]))
        position = tuple(float(value) for value in obj.get("position", [0.0, 0.0, 3.0]))
        camera_pan = tuple(float(value) for value in (camera_state or {"pan": (0.0, 0.0)}).get("pan", (0.0, 0.0)))
        camera_zoom = float((camera_state or {"zoom": 1.0}).get("zoom", 1.0))
        cache_key = (
            str(asset_path),
            rect,
            rotation,
            scale,
            position,
            camera_pan,
            round(camera_zoom, 4),
            str(obj.get("fit_mode", "cover")),
            round(float(obj.get("opacity", 1.0)), 4),
            round(float(obj.get("shadow_strength", 0.0)), 4),
            round(float(obj.get("glow_strength", 0.0)), 4),
        )
        cached_layers = self._plane_layer_cache.get(cache_key)
        if cached_layers is None:
            transformed = self.transform_plane(asset, obj, rect)
            center = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
            left = int(center[0] - transformed.width / 2)
            top = int(center[1] - transformed.height / 2)

            shadow_layer: Image.Image | None = None
            if float(obj.get("shadow_strength", 0.0)) > 0.0:
                shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                alpha = transformed.getchannel("A").point(
                    lambda value: int(value * min(0.9, float(obj.get("shadow_strength", 0.0))) * 0.55)
                )
                shadow = Image.new("RGBA", transformed.size, (0, 0, 0, 0))
                shadow.putalpha(alpha)
                shadow_layer.alpha_composite(shadow, (left + 18, top + 22))
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(18))

            glow_layer: Image.Image | None = None
            if float(obj.get("glow_strength", 0.0)) > 0.0:
                glow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                alpha = transformed.getchannel("A").point(
                    lambda value: int(value * min(1.0, float(obj.get("glow_strength", 0.0))) * 0.35)
                )
                glow = Image.new("RGBA", transformed.size, (*self.green, 0))
                glow.putalpha(alpha)
                glow_layer.alpha_composite(glow, (left, top))
                glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(24))

            plane_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            plane_layer.alpha_composite(transformed, (left, top))
            cached_layers = (shadow_layer, glow_layer, plane_layer)
            self._plane_layer_cache[cache_key] = cached_layers

        shadow_layer, glow_layer, plane_layer = cached_layers
        if shadow_layer is not None:
            image.alpha_composite(shadow_layer)
        if glow_layer is not None:
            image.alpha_composite(glow_layer)
        image.alpha_composite(plane_layer)

    def draw_double_sided_plane(self, image: Image.Image, obj: dict, camera_state: dict | None = None, scene: dict | None = None, seconds: float = 0.0) -> None:
        obj = self.animated_object(obj, seconds, scene)
        flip_progress = float(obj.get("flip_progress", 0.0))
        front_path = obj.get("front_asset_path")
        back_path = obj.get("back_asset_path")
        angle = float(obj.get("rotation", [0.0, 0.0, 0.0])[1] if len(obj.get("rotation", [])) > 1 else 0.0) + 180.0 * flip_progress
        face_front = math.cos(math.radians(angle)) >= 0
        asset_path = front_path if face_front or not back_path or "__bind_" in str(back_path) else back_path
        if not asset_path:
            return
        render_obj = dict(obj)
        render_rotation = list(render_obj.get("rotation", [0.0, 0.0, 0.0]))
        while len(render_rotation) < 3:
            render_rotation.append(0.0)
        render_rotation[1] = angle
        render_obj["rotation"] = render_rotation
        render_obj["asset_path"] = asset_path
        render_obj["glow_strength"] = float(obj.get("front_glow" if face_front else "back_glow", obj.get("glow_strength", 0.0)))
        self.draw_image_plane(image, render_obj, camera_state=camera_state, scene=scene, seconds=seconds)

    def resolve_scene_object_ids(self, scene: dict | None) -> list[str]:
        if not scene:
            return []
        resolved: list[str] = []
        for object_id in scene.get("objects", []):
            obj = self.world_objects.get(object_id)
            if not obj:
                continue
            if obj.get("type") == "wall_cluster":
                for child_id in obj.get("children", []):
                    if child_id not in resolved:
                        resolved.append(child_id)
                continue
            if object_id not in resolved:
                resolved.append(object_id)
        return resolved

    def draw_world_text_layers(self, draw: ImageDraw.ImageDraw, scene: dict | None, state: EffectState) -> None:
        if not scene:
            return
        for layer_id in scene.get("text_layers", []):
            layer = self.text_layers.get(layer_id)
            if not layer:
                continue
            mode = layer.get("mode", "hold")
            content = str(layer.get("content", "")).strip()
            if not content:
                continue
            font_scale = float(layer.get("font_scale", 1.0))
            font = self.title_font if mode == "full-frame" else self.bold
            color_name = str(layer.get("color_role", "text"))
            color = {
                "text": self.text,
                "text_dim": self.text_dim,
                "warning": self.warning,
                "green": self.green_soft,
            }.get(color_name, self.text)
            alpha = int(120 + state.glow * 70)
            if mode == "ghost":
                alpha = int(70 + state.glow * 40)
                position = (150, self.height // 2 - 90)
            else:
                position = (self.width // 2, self.height // 2 - 20)

            if mode == "full-frame":
                font_size = max(32, int(self.config["style"]["title_font_size"] * font_scale))
                font = ImageFont.truetype(self.config["style"]["font_bold"], font_size)
            elif mode in {"hold", "ghost", "sidebar"}:
                font_size = max(22, int(self.config["style"]["font_size"] * font_scale))
                font = ImageFont.truetype(self.config["style"]["font_bold"], font_size if mode != "ghost" else max(18, font_size - 4))

            bbox = draw.textbbox((0, 0), content, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            if mode == "ghost":
                x, y = position
            elif mode == "sidebar":
                x, y = (self.width - 420, 180)
            else:
                x = position[0] - text_w // 2
                y = position[1] - text_h // 2
            draw.text((x, y), content, font=font, fill=(*color, alpha))

    def draw_led_grid(self, image: Image.Image, layer: dict, seconds: float, state: EffectState, bounds: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = bounds
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        grid = layer.get("grid", [12, 2])
        cols = max(1, int(grid[0] if len(grid) > 0 else 12))
        rows = max(1, int(grid[1] if len(grid) > 1 else 2))
        radius = float(layer.get("radius", 7.0))
        base_alpha = int(layer.get("base_alpha", 14))
        peak_alpha = int(layer.get("peak_alpha", 92))
        glow_alpha = int(layer.get("glow_alpha", 40))
        blink_rate = float(layer.get("blink_rate", 0.7))
        beat_weight = float(layer.get("beat_weight", 0.7))
        sweep_rate = float(layer.get("sweep_rate", 0.24))
        accent_every = max(1, int(layer.get("accent_every", 4)))
        opacity = max(0.0, min(1.0, float(layer.get("opacity", 1.0))))
        color = self.color_for_role(str(layer.get("color_role", "green")))
        draw = ImageDraw.Draw(image, "RGBA")

        for row in range(rows):
            for col in range(cols):
                cx = x0 + (width / 2 if cols == 1 else width * (col / max(1, cols - 1)))
                cy = y0 + (height / 2 if rows == 1 else height * (row / max(1, rows - 1)))
                index = row * cols + col
                phase = float(layer.get("phase", 0.0)) + index * 0.073 + row * 0.11
                blink = 0.5 + 0.5 * math.sin((seconds * blink_rate + phase) * math.tau)
                sweep = 0.5 + 0.5 * math.sin((seconds * sweep_rate) * math.tau - col * 0.42 + row * 0.2)
                accent = 1.18 if index % accent_every == 0 else 0.94
                intensity = min(1.0, 0.14 + blink * 0.48 + sweep * 0.18 + state.beat * beat_weight * 0.56)
                dot_alpha = int((base_alpha + (peak_alpha - base_alpha) * intensity * accent) * opacity)
                glow = int(glow_alpha * (0.35 + intensity * 0.9) * opacity)
                dot_radius = radius * (1.0 + state.beat * 0.16 + (0.08 if index % accent_every == 0 else 0.0))
                draw.ellipse(
                    (
                        int(cx - dot_radius * 1.8),
                        int(cy - dot_radius * 1.8),
                        int(cx + dot_radius * 1.8),
                        int(cy + dot_radius * 1.8),
                    ),
                    fill=(*color, max(0, min(255, glow))),
                )
                draw.ellipse(
                    (
                        int(cx - dot_radius),
                        int(cy - dot_radius),
                        int(cx + dot_radius),
                        int(cy + dot_radius),
                    ),
                    fill=(*color, max(0, min(255, dot_alpha))),
                )

    def draw_snake_overlay(self, image: Image.Image, layer: dict, seconds: float, state: EffectState, bounds: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = bounds
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        grid = layer.get("grid", [14, 10])
        cols = max(4, int(grid[0] if len(grid) > 0 else 14))
        rows = max(4, int(grid[1] if len(grid) > 1 else 10))
        cell_size = max(6.0, min(width / cols, height / rows))
        grid_width = cell_size * cols
        grid_height = cell_size * rows
        origin_x = x0 + (width - grid_width) / 2
        origin_y = y0 + (height - grid_height) / 2
        padding = float(layer.get("cell_padding", 0.18)) * cell_size
        opacity = max(0.0, min(1.0, float(layer.get("opacity", 0.4))))
        draw = ImageDraw.Draw(image, "RGBA")
        color = self.color_for_role(str(layer.get("color_role", "green_dim")))
        head_color = self.color_for_role(str(layer.get("head_color_role", "green")))
        food_color = self.color_for_role(str(layer.get("food_color_role", "warning")))
        grid_alpha = int(layer.get("grid_alpha", 18))
        body_alpha = int(layer.get("body_alpha", 52))
        head_alpha = int(layer.get("head_alpha", 120))
        glow_alpha = int(layer.get("glow_alpha", 28))
        speed = float(layer.get("speed", 4.0))
        beat_speed = float(layer.get("beat_speed", 0.28))
        trail_length = max(3, int(layer.get("trail_length", 12)))
        food_offset = max(3, int(layer.get("food_offset", 17)))
        waypoints = layer.get("path")
        if not isinstance(waypoints, list):
            waypoints = self.default_snake_waypoints(cols, rows)
        route = self.grid_route_points(waypoints, cols, rows, closed=bool(layer.get("closed", True)))
        if not route:
            return

        if layer.get("show_grid", True):
            for col in range(cols + 1):
                x = int(origin_x + col * cell_size)
                draw.line((x, int(origin_y), x, int(origin_y + grid_height)), fill=(*color, int(grid_alpha * opacity)), width=1)
            for row in range(rows + 1):
                y = int(origin_y + row * cell_size)
                draw.line((int(origin_x), y, int(origin_x + grid_width), y), fill=(*color, int(grid_alpha * opacity)), width=1)

        route_length = len(route)
        head_index = int((seconds + float(layer.get("phase", 0.0))) * speed * (1.0 + state.beat * beat_speed))
        food_index = (head_index + food_offset) % route_length
        food_x, food_y = route[food_index]
        food_rect = (
            int(origin_x + food_x * cell_size + padding * 0.9),
            int(origin_y + food_y * cell_size + padding * 0.9),
            int(origin_x + (food_x + 1) * cell_size - padding * 0.9),
            int(origin_y + (food_y + 1) * cell_size - padding * 0.9),
        )
        food_alpha = int((54 + (0.5 + 0.5 * math.sin(seconds * 3.6)) * 86) * opacity)
        draw.rounded_rectangle(food_rect, radius=max(2, int(cell_size * 0.18)), fill=(*food_color, max(0, min(255, food_alpha))))

        for segment_index in range(trail_length):
            cell_x, cell_y = route[(head_index - segment_index) % route_length]
            fade = 1.0 - (segment_index / max(1, trail_length))
            fill_color = head_color if segment_index == 0 else color
            alpha = int((head_alpha if segment_index == 0 else body_alpha) * fade * opacity)
            rect = (
                int(origin_x + cell_x * cell_size + padding),
                int(origin_y + cell_y * cell_size + padding),
                int(origin_x + (cell_x + 1) * cell_size - padding),
                int(origin_y + (cell_y + 1) * cell_size - padding),
            )
            if segment_index == 0 and glow_alpha > 0:
                draw.rounded_rectangle(
                    (
                        rect[0] - int(cell_size * 0.2),
                        rect[1] - int(cell_size * 0.2),
                        rect[2] + int(cell_size * 0.2),
                        rect[3] + int(cell_size * 0.2),
                    ),
                    radius=max(2, int(cell_size * 0.24)),
                    fill=(*head_color, int(glow_alpha * (0.7 + state.beat * 0.6) * opacity)),
                )
            draw.rounded_rectangle(rect, radius=max(2, int(cell_size * 0.18)), fill=(*fill_color, max(0, min(255, alpha))))

        if layer.get("show_score", False):
            score_text = str(layer.get("score_text", "SNAKE // SIGNAL"))
            score_alpha = int((84 + state.beat * 50) * opacity)
            draw.text((int(origin_x), int(origin_y - 30)), score_text, font=self.small, fill=(*self.green_dim, score_alpha))

    def draw_scene_ambient_layers(
        self,
        image: Image.Image,
        seconds: float,
        state: EffectState,
        scene: dict | None,
        terminal_rect: tuple[int, int, int, int],
        draw_order: str = "background",
    ) -> None:
        if not scene:
            return
        for layer_id in scene.get("ambient_layers", []):
            layer = self.ambient_layers.get(layer_id)
            if not layer:
                continue
            if str(layer.get("draw_order", "background")) != draw_order:
                continue
            bounds = self.ambient_bounds(layer, terminal_rect)
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            layer_type = str(layer.get("type", ""))
            if layer_type == "led_grid":
                self.draw_led_grid(overlay, layer, seconds, state, bounds)
            elif layer_type == "snake_overlay":
                self.draw_snake_overlay(overlay, layer, seconds, state, bounds)
            image.alpha_composite(overlay)

    def draw_world_scene(self, image: Image.Image, draw: ImageDraw.ImageDraw, seconds: float, state: EffectState) -> None:
        scene = self.world_scene_at(seconds)
        if not scene:
            return
        camera_state = self.scene_camera_state(seconds, scene)
        render_ids = self.resolve_scene_object_ids(scene)
        render_objects: list[dict] = []
        for object_id in render_ids:
            obj = self.world_objects.get(object_id)
            if obj is None or obj.get("type") == "terminal_window":
                continue
            render_objects.append(obj)
        render_objects.sort(key=lambda item: float(item.get("position", [0.0, 0.0, 3.0])[2] if len(item.get("position", [])) > 2 else 3.0), reverse=True)

        for obj in render_objects:
            if obj.get("type") == "image_plane":
                self.draw_image_plane(image, obj, camera_state=camera_state, scene=scene, seconds=seconds)
            elif obj.get("type") == "double_sided_plane":
                self.draw_double_sided_plane(image, obj, camera_state=camera_state, scene=scene, seconds=seconds)

    def draw_terminal_frame(self, draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], state: EffectState, mode: str, terminal_mode: str) -> None:
        x0, y0, x1, y1 = rect
        border_alpha = int(100 + state.glow * 140)
        fill_alpha = 32 if mode != "bridge" else 18
        if terminal_mode == "frame":
            fill_alpha = 8 if mode != "bridge" else 4
            border_alpha = int(border_alpha * 0.78)
        elif self.world_scene_at(getattr(self, "_render_seconds", 0.0)) is not None:
            fill_alpha = 18 if mode != "bridge" else 10
        draw.rounded_rectangle(rect, radius=16, fill=(0, 9, 2, fill_alpha), outline=(*self.border, border_alpha), width=3)
        if self.presentation_flag("show_terminal_header", True):
            header_y = y0 + 44
            draw.line((x0, header_y, x1, header_y), fill=(*self.border, int(60 + state.glow * 90)), width=1)
            left_label = self.presentation_text("terminal_header_left", "NERO / INSIDE THE TERMINAL")
            right_label = self.presentation_text("terminal_header_right", "SIGNAL CHANNEL: OPEN")
            if left_label:
                draw.text((x0 + 28, y0 + 14), left_label, font=self.small, fill=(*self.green_dim, 230))
            if right_label:
                right_bbox = draw.textbbox((0, 0), right_label, font=self.small)
                right_width = right_bbox[2] - right_bbox[0]
                draw.text((x1 - 28 - right_width, y0 + 14), right_label, font=self.small, fill=(*self.green_dim, 210))

    def apply_glow(self, image: Image.Image, rect: tuple[int, int, int, int], color: tuple[int, int, int], strength: float) -> None:
        glow_step = int(round(min(1.0, max(0.0, strength)) * 20))
        key = (rect, glow_step)
        overlay = self._glow_cache.get(key)
        if overlay is None:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            alpha = int(90 * (glow_step / 20))
            for width, blur in [(8, 14), (3, 5)]:
                layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                layer_draw = ImageDraw.Draw(layer)
                layer_draw.rounded_rectangle(rect, radius=12, outline=(*color, alpha), width=width)
                overlay.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
            self._glow_cache[key] = overlay
        image.alpha_composite(overlay)

    def apply_scanlines(self, image: Image.Image, alpha: int) -> None:
        scanline_alpha = max(0, min(80, int(alpha)))
        overlay = self._scanline_cache.get(scanline_alpha)
        if overlay is None:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for y in range(0, self.height, 4):
                draw.line((0, y, self.width, y), fill=(0, 0, 0, scanline_alpha), width=1)
            self._scanline_cache[scanline_alpha] = overlay
        image.alpha_composite(overlay)

    def draw_sidebar(self, draw: ImageDraw.ImageDraw, scene: SceneEvent, rect: tuple[int, int, int, int], seconds: float, state: EffectState) -> None:
        if not self.presentation_flag("show_sidebar", True):
            return
        if scene.mode not in {"sidebar", "final"}:
            return
        x0, y0, _x1, y1 = rect
        sidebar_x = x0 + 28
        sidebar_y = y0 + 88
        sidebar_label = self.presentation_text("sidebar_label", "TRACK DOORS").strip()
        if sidebar_label:
            draw.text((sidebar_x, sidebar_y), sidebar_label, font=self.small, fill=(*self.green_dim, 220))
            first_item_offset = 44
        else:
            first_item_offset = 10
        items = scene.sidebar_items or [
            "tomorrow in the wires",
            "signal made a room for me",
            "inside the terminal",
            "we never synced",
        ]
        for index, item in enumerate(items[:8]):
            y = sidebar_y + first_item_offset + index * 34
            pulse = 0.5 + 0.5 * math.sin(seconds * 2.2 + index)
            alpha = int(80 + pulse * 75 + state.beat * 60)
            label = item if len(item) <= 30 else f"{item[:27]}..."
            draw.text((sidebar_x, y), f"> {label}", font=self.small, fill=(*self.green_dim, alpha))

    def draw_side_annotations(self, draw: ImageDraw.ImageDraw, scene: SceneEvent, rect: tuple[int, int, int, int], state: EffectState) -> None:
        if not self.presentation_flag("show_side_annotations", True):
            return
        if not scene.side_annotations:
            return
        _x0, y0, x1, _y1 = rect
        x = x1 - 340
        y = y0 + 110
        for index, text in enumerate(scene.side_annotations):
            alpha = int(85 + index * 25 + state.beat * 50)
            draw.text((x, y + index * 36), text, font=self.small, fill=(*self.green_dim, alpha))

    def draw_depth_grid(self, draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], state: EffectState, mode: str) -> None:
        if mode not in {"shelter", "final"}:
            return
        x0, _y0, x1, y1 = rect
        floor_y = y1 - 150
        alpha = int(25 + state.glow * 45)
        for i in range(10):
            y = floor_y + i * 14
            draw.line((x0 + 24, y, x1 - 24, y + i * 5), fill=(*self.green_dim, alpha), width=1)
        for i in range(9):
            x = x0 + 120 + i * 145
            draw.line((x, floor_y, x0 + 780 + (i - 4) * 70, y1 - 28), fill=(*self.green_dim, alpha), width=1)

    def draw_progress(self, draw: ImageDraw.ImageDraw, seconds: float, state: EffectState) -> None:
        y = self.height - 82
        x0 = 270
        x1 = self.width - 270
        ratio = min(1.0, max(0.0, seconds / self.timeline.duration))
        draw.text((x0 - 96, y - 13), self.format_time(seconds), font=self.small, fill=(*self.green_dim, 230))
        draw.text((x1 + 24, y - 13), self.format_time(self.timeline.duration), font=self.small, fill=(*self.green_dim, 180))
        draw.line((x0, y, x1, y), fill=(*self.green_dim, 120), width=3)
        draw.line((x0, y, x0 + int((x1 - x0) * ratio), y), fill=(*self.green_soft, int(150 + state.glow * 90)), width=5)
        beat_window = [beat for beat in self.analysis.beats if 0 <= beat <= self.timeline.duration]
        for beat in beat_window[::2]:
            bx = x0 + int((x1 - x0) * (beat / self.timeline.duration))
            alpha = 70 if abs(seconds - beat) > 0.14 else 210
            draw.line((bx, y - 11, bx, y + 11), fill=(*self.green_dim, alpha), width=1)
        playhead = x0 + int((x1 - x0) * ratio)
        draw.ellipse((playhead - 8, y - 8, playhead + 8, y + 8), outline=(*self.green_soft, 230), width=2)

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def render(self, seconds: float, frame_index: int) -> RenderedFrame:
        # World-plane caches include full-frame RGBA layers keyed by camera state.
        # In v2 the camera moves almost every frame, so retaining this cache across
        # frames causes near-unbounded growth and can crash long renders.
        self._plane_layer_cache.clear()
        scene = self.scene_at(seconds)
        mode = scene.mode
        world_scene = self.world_scene_at(seconds)
        terminal_mode = self.scene_terminal_mode(world_scene)
        state = effect_state(self.analysis, seconds, mode, self.config)
        image = Image.new("RGBA", (self.width, self.height), (*self.bg, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        self._render_seconds = seconds

        terminal_rect = (210, 120, self.width - 210, self.height - 145)
        if mode == "bridge":
            draw.rectangle((0, 0, self.width, self.height), fill=(0, 0, 0, 235))

        if terminal_mode != "hidden":
            glow_strength = state.glow if terminal_mode == "full" else state.glow * 0.58
            self.apply_glow(image, terminal_rect, self.green, glow_strength)
            self.draw_terminal_frame(draw, terminal_rect, state, mode, terminal_mode)
        self.draw_scene_ambient_layers(image, seconds, state, world_scene, terminal_rect, draw_order="background")
        self.draw_world_scene(image, draw, seconds, state)
        self.draw_scene_ambient_layers(image, seconds, state, world_scene, terminal_rect, draw_order="foreground")
        if terminal_mode != "hidden":
            self.draw_depth_grid(draw, terminal_rect, state, mode)
            self.draw_sidebar(draw, scene, terminal_rect, seconds, state)
            self.draw_side_annotations(draw, scene, terminal_rect, state)

        text_left = terminal_rect[0] + (500 if mode in {"sidebar", "final"} else 62)
        text_top = terminal_rect[1] + (94 if self.presentation_flag("show_terminal_header", True) else 56)
        line_height = 42
        y = text_top
        if terminal_mode == "full":
            active_lines, cursor_text = self.active_lines(seconds, scene)
            if mode == "boot" and not active_lines:
                active_lines = []

            for event, rendered in active_lines:
                font = self.bold if event.emphasis or mode in {"chorus", "final"} and event.emphasis else self.font
                color = self.warning if event.emphasis else self.text
                if mode == "bridge":
                    color = self.green_dim if not event.emphasis else self.green_soft
                alpha = int(145 + state.text_boost * 70 + (60 if event.emphasis else 0))
                wrap_width = 46 if mode in {"sidebar", "final"} else 64
                for wrapped in self.wrap_line(rendered, width=wrap_width):
                    draw.text((text_left, y), wrapped, font=font, fill=(*color, min(255, alpha)))
                    y += line_height

            cursor_x = text_left + 14
            cursor_y = max(text_top, y - line_height)
            if active_lines:
                last_text = active_lines[-1][1]
                bbox = draw.textbbox((text_left, cursor_y), last_text, font=self.font)
                cursor_x = min(terminal_rect[2] - 90, bbox[2] + 10)
            draw.rectangle(
                (cursor_x, cursor_y + 8, cursor_x + 18, cursor_y + 40),
                fill=(*self.green_soft, state.cursor_alpha),
            )

        if mode == "boot" and self.presentation_flag("show_boot_overlay", True):
            boot_alpha = int(min(1.0, seconds / 8.0) * 220)
            draw.text((terminal_rect[0] + 62, terminal_rect[1] + 230), "signal channel opening", font=self.title_font, fill=(*self.green_soft, boot_alpha))
            draw.text((terminal_rect[0] + 62, terminal_rect[1] + 292), "nero presence detected", font=self.font, fill=(*self.green_dim, boot_alpha))

        if mode == "bridge" and self.presentation_flag("show_bridge_footer", True):
            draw.text((terminal_rect[0] + 62, terminal_rect[3] - 90), "pulse retained", font=self.small, fill=(*self.green_dim, 190 + int(state.beat * 60)))

        self.draw_world_text_layers(draw, self.world_scene_at(seconds), state)
        if self.presentation_flag("show_progress", True):
            self.draw_progress(draw, seconds, state)
        self.apply_scanlines(image, int(self.config["effects"].get("scanline_alpha", 24) + state.glow * 10))
        draw_noise(image, int(self.config["effects"].get("noise_alpha", 12)), frame_index)
        return RenderedFrame(image=image.convert("RGB"), mode=mode)

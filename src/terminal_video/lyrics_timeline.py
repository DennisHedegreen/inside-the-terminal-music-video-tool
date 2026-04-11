from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class LineEvent:
    text: str
    start: float
    end: float
    mode: str
    scene_id: str
    emphasis: bool = False


@dataclass(frozen=True)
class SceneEvent:
    scene_id: str
    mode: str
    start: float
    end: float
    side_annotations: list[str]
    sidebar_items: list[str]


@dataclass(frozen=True)
class Timeline:
    lines: list[LineEvent]
    scenes: list[SceneEvent]
    duration: float


def parse_lyrics_markdown(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        heading = re.match(r"^###\s+(.+)$", line)
        if heading:
            current = heading.group(1).strip()
            sections.setdefault(current, [])
            continue
        if not current or not line or line.startswith("#"):
            continue
        if current == "Lyrics":
            continue
        sections.setdefault(current, []).append(line.rstrip())

    return sections


def make_line_events(
    scene: dict,
    lines: list[str],
    scene_end: float,
    emphasis_phrases: list[str],
) -> list[LineEvent]:
    if not lines:
        return []

    scene_start = float(scene["start"])
    usable_start = scene_start + 0.8
    usable_end = max(usable_start + 1.0, scene_end - 0.8)
    total_chars = sum(max(8, len(line)) for line in lines)
    cursor = usable_start
    events: list[LineEvent] = []

    for index, line in enumerate(lines):
      weight = max(8, len(line)) / max(1, total_chars)
      slot = max(0.7, (usable_end - usable_start) * weight)
      start = cursor
      end = min(usable_end, start + slot * 0.78)
      cursor = min(usable_end, start + slot)
      events.append(
          LineEvent(
              text=line,
              start=start,
              end=max(start + 0.35, end),
              mode=scene.get("mode", "verse"),
              scene_id=scene["id"],
              emphasis=line in emphasis_phrases,
          )
      )

    return events


def build_timeline(config: dict, sections: dict[str, list[str]], duration: float) -> Timeline:
    scenes: list[SceneEvent] = []
    line_events: list[LineEvent] = []
    emphasis_phrases = config.get("emphasis_phrases", [])
    raw_timeline = config["timeline"]

    for index, scene in enumerate(raw_timeline):
        scene_start = float(scene["start"])
        explicit_end = scene.get("end")
        next_start = raw_timeline[index + 1]["start"] if index + 1 < len(raw_timeline) else duration
        scene_end = float(explicit_end) if explicit_end is not None else float(next_start or duration)
        scene_end = min(max(scene_start + 1.0, scene_end), duration)

        scenes.append(
            SceneEvent(
                scene_id=scene["id"],
                mode=scene.get("mode", "verse"),
                start=scene_start,
                end=scene_end,
                side_annotations=list(scene.get("side_annotations", [])),
                sidebar_items=list(scene.get("sidebar_items", [])),
            )
        )

        lines: list[str] = []
        lines.extend(scene.get("system_lines", []))
        for section_name in scene.get("source_sections", []):
            lines.extend(sections.get(section_name, []))

        line_events.extend(make_line_events(scene, lines, scene_end, emphasis_phrases))

    return Timeline(lines=line_events, scenes=scenes, duration=duration)

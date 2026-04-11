#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from render_inside_terminal import load_config, resolve_tool_path
from src.terminal_video.audio_analyzer import audio_duration_seconds
from src.terminal_video.render import render_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the public Inside the Terminal config in resumable parts.")
    parser.add_argument("--config", default="configs/inside-the-terminal.public-v1.json")
    parser.add_argument("--segment-seconds", type=float, default=6.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--list-only", action="store_true", help="Print segment plan without rendering.")
    parser.add_argument("--skip-concat", action="store_true", help="Render segments only and skip final concat/mux.")
    parser.add_argument("--concat-only", action="store_true", help="Only concatenate existing segments and mux audio.")
    return parser.parse_args()


def load_plan(tool_root: Path, config_arg: str, segment_seconds: float) -> dict:
    config_path = resolve_tool_path(tool_root, config_arg)
    config = load_config(config_path)
    audio_path = resolve_tool_path(tool_root, config["audio_path"])
    duration = audio_duration_seconds(audio_path)
    fps = int(config["video"]["fps"])
    _start_seconds, _end_seconds, start_frame, end_frame = render_bounds(config, duration, fps)
    frame_count = max(0, end_frame - start_frame)
    frames_per_segment = max(1, int(round(segment_seconds * fps)))
    segment_count = int(math.ceil(frame_count / frames_per_segment))

    return {
        "config": config,
        "config_path": config_path,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "segment_count": segment_count,
        "segment_seconds": segment_seconds,
    }


def command_for_segment(tool_root: Path, config_path: Path, segment_index: int, segment_seconds: float) -> list[str]:
    return [
        sys.executable,
        str(tool_root / "render_inside_terminal.py"),
        "--config",
        str(config_path),
        "--segment-index",
        str(segment_index),
        "--segment-seconds",
        str(segment_seconds),
    ]


def command_for_concat(tool_root: Path, config_path: Path, segment_seconds: float) -> list[str]:
    return [
        sys.executable,
        str(tool_root / "render_inside_terminal.py"),
        "--config",
        str(config_path),
        "--concat-only",
        "--segment-seconds",
        str(segment_seconds),
    ]


def main() -> int:
    args = parse_args()
    tool_root = Path(__file__).resolve().parent
    plan = load_plan(tool_root, args.config, args.segment_seconds)
    config = plan["config"]
    config_path = plan["config_path"]
    segment_count = int(plan["segment_count"])

    if segment_count <= 0:
        raise SystemExit("No segments to render for this config window.")

    start_index = max(0, args.start_index)
    end_index = segment_count - 1 if args.end_index is None else min(segment_count - 1, args.end_index)
    if end_index < start_index:
        raise SystemExit(f"Invalid segment range: {start_index}..{end_index}")

    print(f"title={config['title']}")
    print(f"slug={config['slug']}")
    print(f"config={config_path}")
    print(f"window={config.get('window', {'start': 0.0, 'end': plan['duration']})}")
    print(f"segment_seconds={args.segment_seconds}")
    print(f"segment_count={segment_count}")
    print(f"range={start_index}-{end_index}")

    if args.list_only:
        for index in range(segment_count):
            marker = "*" if start_index <= index <= end_index else "-"
            print(f"{marker} segment {index}")
        return 0

    if args.concat_only:
        subprocess.run(
            command_for_concat(tool_root, config_path, args.segment_seconds),
            cwd=tool_root,
            check=True,
        )
        return 0

    for index in range(start_index, end_index + 1):
        print(f"rendering segment {index}/{segment_count - 1}", flush=True)
        subprocess.run(
            command_for_segment(tool_root, config_path, index, args.segment_seconds),
            cwd=tool_root,
            check=True,
        )

    if not args.skip_concat:
        print("concatenating and muxing final output", flush=True)
        subprocess.run(
            command_for_concat(tool_root, config_path, args.segment_seconds),
            cwd=tool_root,
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

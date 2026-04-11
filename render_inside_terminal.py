#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.terminal_video.audio_analyzer import analyze_audio_cached
from src.terminal_video.lyrics_timeline import build_timeline, parse_lyrics_markdown
from src.terminal_video.render import concat_segments_and_mux, render_bounds, render_segment, render_video


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    extends_value = config.get("extends")
    if not extends_value:
        references = config.get("references", {})
        extends_value = references.get("base_config")

    if extends_value:
        base_path = (path.parent / extends_value).resolve()
        base_config = load_config(base_path)
        config = _deep_merge(base_config, config)

    config["__config_path__"] = str(path.resolve())
    config["__config_dir__"] = str(path.parent.resolve())
    return config


def resolve_tool_path(tool_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (tool_root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the public Inside the Terminal music video.")
    parser.add_argument("--config", default="configs/inside-the-terminal.public-v1.json")
    parser.add_argument("--preview-only", action="store_true", help="Render only preview PNG and report.")
    parser.add_argument("--segment-index", type=int, default=None, help="Render a single zero-based segment instead of the full video.")
    parser.add_argument("--segment-seconds", type=float, default=40.0, help="Segment length used with --segment-index or --concat-only.")
    parser.add_argument("--concat-only", action="store_true", help="Concatenate rendered segments and mux the original audio.")
    args = parser.parse_args()

    tool_root = Path(__file__).resolve().parent
    config_path = resolve_tool_path(tool_root, args.config)
    config = load_config(config_path)

    audio_path = resolve_tool_path(tool_root, config["audio_path"])
    lyrics_path = resolve_tool_path(tool_root, config["lyrics_path"])
    output_dir = resolve_tool_path(tool_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_cache_path = output_dir / f"{audio_path.stem}.audio-analysis.json"

    sections = parse_lyrics_markdown(lyrics_path)
    print(f"render: audio analysis ({analysis_cache_path.name})", flush=True)
    analysis = analyze_audio_cached(audio_path, cache_path=analysis_cache_path)
    timeline = build_timeline(config, sections, analysis.duration)

    if args.concat_only:
        _window_start, _window_end, window_start_frame, window_end_frame = render_bounds(
            config,
            timeline.duration,
            int(config["video"]["fps"]),
        )
        output_mp4, segment_paths = concat_segments_and_mux(
            config=config,
            audio_path=audio_path,
            output_dir=output_dir,
            timeline=timeline,
            segment_seconds=args.segment_seconds,
        )
        result = type("ConcatResult", (), {
            "output_mp4": output_mp4,
            "preview_png": output_dir / f"{config['slug']}.preview.png",
            "frame_count": max(0, window_end_frame - window_start_frame),
            "muxed": True,
        })()
        segment_report = [str(path) for path in segment_paths]
    elif args.segment_index is not None:
        result = render_segment(
            config=config,
            output_dir=output_dir,
            analysis=analysis,
            timeline=timeline,
            segment_index=args.segment_index,
            segment_seconds=args.segment_seconds,
        )
        segment_report = [str(result.output_mp4)] if result.output_mp4 else []
    else:
        result = render_video(
            config=config,
            audio_path=audio_path,
            output_dir=output_dir,
            analysis=analysis,
            timeline=timeline,
            preview_only=args.preview_only,
        )
        segment_report = []

    report = {
        "title": config["title"],
        "audio_path": str(audio_path),
        "lyrics_path": str(lyrics_path),
        "duration_seconds": analysis.duration,
        "render_window": config.get("window", {"start": 0.0, "end": analysis.duration}),
        "tempo_bpm": analysis.tempo,
        "beat_count": len(analysis.beats),
        "output_mp4": str(result.output_mp4) if result.output_mp4 else "",
        "preview_png": str(result.preview_png),
        "frame_count": result.frame_count,
        "fps": config["video"]["fps"],
        "resolution": [config["video"]["width"], config["video"]["height"]],
        "muxed": result.muxed,
        "preview_only": args.preview_only,
        "segment_index": args.segment_index,
        "segment_seconds": args.segment_seconds,
        "concat_only": args.concat_only,
        "segments": segment_report,
    }
    report_path = output_dir / "render-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

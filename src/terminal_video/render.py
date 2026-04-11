from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .audio_analyzer import AudioAnalysis
from .lyrics_timeline import Timeline
from .mux import concat_video_segments, encode_raw_video, mux_audio, mux_audio_window
from .terminal_renderer import TerminalRenderer


@dataclass(frozen=True)
class RenderResult:
    output_mp4: Path | None
    preview_png: Path
    frame_count: int
    muxed: bool


def render_bounds(config: dict, duration: float, fps: int) -> tuple[float, float, int, int]:
    window = config.get("window", {})
    start_seconds = float(window.get("start", 0.0))
    end_raw = window.get("end", duration)
    end_seconds = float(duration if end_raw is None else end_raw)
    start_seconds = max(0.0, min(start_seconds, duration))
    end_seconds = max(start_seconds + 0.1, min(end_seconds, duration))
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))
    return start_seconds, end_seconds, start_frame, end_frame


def _write_preview(
    config: dict,
    output_dir: Path,
    renderer: TerminalRenderer,
    timeline: Timeline,
) -> Path:
    slug = config["slug"]
    print("render: writing preview frame", flush=True)
    fps = int(config["video"]["fps"])
    start_seconds, end_seconds, _start_frame, _end_frame = render_bounds(config, timeline.duration, fps)
    midpoint = start_seconds + (end_seconds - start_seconds) * 0.5
    preview_frame = renderer.render(min(timeline.duration - 0.1, midpoint), 0).image
    preview_path = output_dir / f"{slug}.preview.png"
    preview_frame.save(preview_path)
    return preview_path


def _encode_frame_range(
    renderer: TerminalRenderer,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    label: str,
) -> None:
    frame_total = max(0, end_frame - start_frame)
    print(
        f"render: encoding {label}: frames {start_frame}-{end_frame - 1} "
        f"({frame_total} frames) at {width}x{height}/{fps}fps",
        flush=True,
    )
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
        raise RuntimeError("ffmpeg stdin unavailable")

    try:
        for absolute_frame in range(start_frame, end_frame):
            seconds = absolute_frame / fps
            frame = renderer.render(seconds, absolute_frame).image
            process.stdin.write(frame.tobytes())
            local_index = absolute_frame - start_frame
            if local_index % max(1, fps * 2) == 0:
                print(
                    f"rendered {label}: {local_index}/{frame_total} frames "
                    f"({seconds:0.1f}s absolute)",
                    flush=True,
                )
    finally:
        process.stdin.close()

    stderr = process.stderr.read() if process.stderr else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed for {label}:\n{stderr.decode(errors='ignore')[-4000:]}")


def render_video(
    config: dict,
    audio_path: Path,
    output_dir: Path,
    analysis: AudioAnalysis,
    timeline: Timeline,
    preview_only: bool = False,
) -> RenderResult:
    video = config["video"]
    width = int(video["width"])
    height = int(video["height"])
    fps = int(video["fps"])
    crf = int(video.get("crf", 20))
    preset = str(video.get("preset", "veryfast"))
    slug = config["slug"]
    _window_start, _window_end, start_frame, end_frame = render_bounds(config, timeline.duration, fps)
    frame_count = max(0, end_frame - start_frame)

    renderer = TerminalRenderer(config=config, analysis=analysis, timeline=timeline)
    preview_path = _write_preview(config, output_dir, renderer, timeline)

    if preview_only:
        return RenderResult(output_mp4=None, preview_png=preview_path, frame_count=frame_count, muxed=False)

    raw_video_path = output_dir / f"{slug}.raw.mp4"
    output_mp4 = output_dir / f"{slug}.mp4"
    _encode_frame_range(
        renderer=renderer,
        output_path=raw_video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
        label="full",
    )

    print("render: muxing audio", flush=True)
    if start_frame == 0 and end_frame >= int(round(timeline.duration * fps)):
        mux_audio(raw_video_path=raw_video_path, audio_path=audio_path, output_path=output_mp4)
    else:
        start_seconds = start_frame / fps
        duration_seconds = frame_count / fps
        mux_audio_window(
            raw_video_path=raw_video_path,
            audio_path=audio_path,
            output_path=output_mp4,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    return RenderResult(output_mp4=output_mp4, preview_png=preview_path, frame_count=frame_count, muxed=True)


def render_segment(
    config: dict,
    output_dir: Path,
    analysis: AudioAnalysis,
    timeline: Timeline,
    segment_index: int,
    segment_seconds: float,
) -> RenderResult:
    video = config["video"]
    width = int(video["width"])
    height = int(video["height"])
    fps = int(video["fps"])
    crf = int(video.get("crf", 20))
    preset = str(video.get("preset", "veryfast"))
    slug = config["slug"]
    _window_start, _window_end, window_start_frame, window_end_frame = render_bounds(config, timeline.duration, fps)
    frame_count = max(0, window_end_frame - window_start_frame)
    frames_per_segment = int(round(segment_seconds * fps))
    local_start_frame = segment_index * frames_per_segment
    local_end_frame = min(frame_count, local_start_frame + frames_per_segment)
    if local_start_frame >= frame_count:
        raise ValueError(f"segment {segment_index} starts beyond frame_count {frame_count}")
    start_frame = window_start_frame + local_start_frame
    end_frame = window_start_frame + local_end_frame

    renderer = TerminalRenderer(config=config, analysis=analysis, timeline=timeline)
    preview_path = _write_preview(config, output_dir, renderer, timeline)
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_path = segment_dir / f"{slug}.segment-{segment_index:03d}.mp4"
    _encode_frame_range(
        renderer=renderer,
        output_path=segment_path,
        start_frame=start_frame,
        end_frame=end_frame,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
        label=f"segment-{segment_index:03d}",
    )
    return RenderResult(output_mp4=segment_path, preview_png=preview_path, frame_count=local_end_frame - local_start_frame, muxed=False)


def concat_segments_and_mux(
    config: dict,
    audio_path: Path,
    output_dir: Path,
    timeline: Timeline,
    segment_seconds: float,
) -> tuple[Path, list[Path]]:
    fps = int(config["video"]["fps"])
    slug = config["slug"]
    _window_start, _window_end, window_start_frame, window_end_frame = render_bounds(config, timeline.duration, fps)
    frame_count = max(0, window_end_frame - window_start_frame)
    frames_per_segment = int(round(segment_seconds * fps))
    segment_count = int((frame_count + frames_per_segment - 1) // frames_per_segment)
    segment_dir = output_dir / "segments"
    segment_paths = [segment_dir / f"{slug}.segment-{index:03d}.mp4" for index in range(segment_count)]
    missing = [path for path in segment_paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing segment files: " + ", ".join(str(path) for path in missing))

    raw_video_path = output_dir / f"{slug}.raw.mp4"
    output_mp4 = output_dir / f"{slug}.mp4"
    concat_list_path = output_dir / f"{slug}.segments.txt"
    print(f"render: concatenating {len(segment_paths)} segments", flush=True)
    concat_video_segments(segment_paths=segment_paths, list_path=concat_list_path, output_path=raw_video_path)
    print("render: muxing audio", flush=True)
    if window_start_frame == 0 and window_end_frame >= int(round(timeline.duration * fps)):
        mux_audio(raw_video_path=raw_video_path, audio_path=audio_path, output_path=output_mp4)
    else:
        mux_audio_window(
            raw_video_path=raw_video_path,
            audio_path=audio_path,
            output_path=output_mp4,
            start_seconds=window_start_frame / fps,
            duration_seconds=frame_count / fps,
        )
    return output_mp4, segment_paths

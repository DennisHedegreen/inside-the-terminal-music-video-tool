from __future__ import annotations

from pathlib import Path
import subprocess

import imageio_ffmpeg


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_raw_video(
    raw_frame_pipe,
    raw_video_path: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_path(),
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),
        str(raw_video_path),
    ]
    return subprocess.Popen(cmd, stdin=raw_frame_pipe, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def mux_audio(raw_video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        ffmpeg_path(),
        "-y",
        "-i", str(raw_video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr[-4000:]}")


def mux_audio_window(
    raw_video_path: Path,
    audio_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    cmd = [
        ffmpeg_path(),
        "-y",
        "-i", str(raw_video_path),
        "-ss", f"{max(0.0, start_seconds):0.3f}",
        "-t", f"{max(0.1, duration_seconds):0.3f}",
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg window mux failed:\n{result.stderr[-4000:]}")


def concat_video_segments(segment_paths: list[Path], list_path: Path, output_path: Path) -> None:
    lines = [f"file '{path.resolve().as_posix()}'" for path in segment_paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        ffmpeg_path(),
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr[-4000:]}")

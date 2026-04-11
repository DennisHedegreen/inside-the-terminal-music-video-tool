from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioAnalysis:
    duration: float
    tempo: float
    beats: list[float]
    rms_times: list[float]
    rms: list[float]
    spectral_times: list[float]
    spectral_centroid: list[float]

    def energy_at(self, seconds: float) -> float:
        return sample_series(self.rms_times, self.rms, seconds)

    def brightness_at(self, seconds: float) -> float:
        return sample_series(self.spectral_times, self.spectral_centroid, seconds)

    def beat_proximity(self, seconds: float, window: float = 0.12) -> float:
        if not self.beats:
            return 0.0
        nearest = min(abs(seconds - beat) for beat in self.beats)
        return max(0.0, 1.0 - (nearest / window))

    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "tempo": self.tempo,
            "beats": self.beats,
            "rms_times": self.rms_times,
            "rms": self.rms,
            "spectral_times": self.spectral_times,
            "spectral_centroid": self.spectral_centroid,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AudioAnalysis":
        return cls(
            duration=float(payload["duration"]),
            tempo=float(payload["tempo"]),
            beats=[float(value) for value in payload.get("beats", [])],
            rms_times=[float(value) for value in payload.get("rms_times", [])],
            rms=[float(value) for value in payload.get("rms", [])],
            spectral_times=[float(value) for value in payload.get("spectral_times", [])],
            spectral_centroid=[float(value) for value in payload.get("spectral_centroid", [])],
        )


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def sample_series(times: list[float], values: list[float], seconds: float) -> float:
    if not times or not values:
        return 0.0
    idx = int(np.searchsorted(times, seconds, side="left"))
    if idx <= 0:
        return float(values[0])
    if idx >= len(values):
        return float(values[-1])
    before_t = times[idx - 1]
    after_t = times[idx]
    if after_t <= before_t:
        return float(values[idx])
    ratio = (seconds - before_t) / (after_t - before_t)
    return float(values[idx - 1] * (1 - ratio) + values[idx] * ratio)


def analyze_audio(audio_path: Path) -> AudioAnalysis:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        trim=False,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_norm = normalize(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)
    centroid_norm = normalize(centroid)

    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if tempo.size else 0.0
    else:
        tempo_value = float(tempo)

    return AudioAnalysis(
        duration=duration,
        tempo=tempo_value,
        beats=[float(beat) for beat in beat_times],
        rms_times=[float(t) for t in rms_times],
        rms=[float(v) for v in rms_norm],
        spectral_times=[float(t) for t in centroid_times],
        spectral_centroid=[float(v) for v in centroid_norm],
    )


def audio_duration_seconds(audio_path: Path) -> float:
    return float(sf.info(str(audio_path)).duration)


def _audio_signature(audio_path: Path) -> dict:
    stat = audio_path.stat()
    return {
        "audio_path": str(audio_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_cached_audio_analysis(cache_path: Path, audio_path: Path) -> AudioAnalysis | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    signature = payload.get("signature", {})
    current = _audio_signature(audio_path)
    if signature != current:
        return None

    analysis_payload = payload.get("analysis")
    if not isinstance(analysis_payload, dict):
        return None
    return AudioAnalysis.from_dict(analysis_payload)


def analyze_audio_cached(audio_path: Path, cache_path: Path | None = None) -> AudioAnalysis:
    if cache_path is not None:
        cached = load_cached_audio_analysis(cache_path, audio_path)
        if cached is not None:
            return cached

    analysis = analyze_audio(audio_path)
    if cache_path is not None:
        payload = {
            "signature": _audio_signature(audio_path),
            "analysis": analysis.to_dict(),
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return analysis

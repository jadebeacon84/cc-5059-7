"""Synthesis and audio sinks.

Offline rendering uses numpy into a stereo WAV (left = performer, right =
accompaniment), which is how replay runs are audited without audio hardware.
A live PortAudio sink is used when ``sounddevice`` and a device are present;
otherwise live runs fall back to a null sink and still emit every event
through the same pipeline.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np

SR = 22050


def midi_to_freq(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def _adsr(n: int, velocity: float) -> np.ndarray:
    a = max(1, int(0.008 * SR))
    r = max(1, int(0.09 * SR))
    env = np.ones(n, dtype=np.float64)
    if n <= a + r:
        a = max(1, n // 4)
        r = max(1, n // 4)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[a:n - r] = 1.0
    env[n - r:] = np.linspace(1.0, 0.0, r)
    return env * velocity


def render_note(pitch: int, seconds: float, velocity: int, velocity_scale: float = 1.0) -> np.ndarray:
    n = max(1, int(seconds * SR))
    t = np.arange(n) / SR
    f = midi_to_freq(pitch)
    # mellow additive tone: fundamental plus two decaying harmonics
    wave_sig = (
        np.sin(2 * math.pi * f * t)
        + 0.35 * np.sin(2 * math.pi * 2 * f * t)
        + 0.12 * np.sin(2 * math.pi * 3 * f * t)
    )
    gain = (velocity / 127.0) * velocity_scale
    return (wave_sig * _adsr(n, gain)).astype(np.float64)


@dataclass
class MidiLikeMessage:
    time: float
    kind: str          # "on" / "off"
    pitch: int
    velocity: int = 80
    velocity_scale: float = 1.0


class AudioBus:
    """Collects raw note messages on one clock; rendered offline or streamed."""

    def __init__(self) -> None:
        self.messages: list[MidiLikeMessage] = []

    def note_on(self, time: float, pitch: int, velocity: int, scale: float = 1.0) -> None:
        self.messages.append(MidiLikeMessage(time, "on", pitch, velocity, scale))

    def note_off(self, time: float, pitch: int) -> None:
        self.messages.append(MidiLikeMessage(time, "off", pitch))

    def render(self, tail: float = 0.6) -> np.ndarray:
        if not self.messages:
            return np.zeros((1, 1))
        # pair on/off events per pitch (last-off wins, re-press supported)
        active: dict[int, MidiLikeMessage] = {}
        segments: list[tuple[float, float, int, int, float]] = []
        for msg in self.messages:
            if msg.kind == "on":
                active[msg.pitch] = msg
            else:
                start_msg = active.pop(msg.pitch, None)
                if start_msg is not None:
                    segments.append(
                        (start_msg.time, msg.time, msg.pitch, start_msg.velocity,
                         start_msg.velocity_scale)
                    )
        end = max(m.time for m in self.messages) + tail
        for pitch, msg in active.items():
            segments.append((msg.time, end, pitch, msg.velocity, msg.velocity_scale))
        total = int(end * SR) + 1
        out = np.zeros(total, dtype=np.float64)
        for start, stop, pitch, vel, scale in segments:
            dur = max(0.05, stop - start)
            sig = render_note(pitch, dur, vel, scale)
            i0 = int(start * SR)
            i1 = min(total, i0 + len(sig))
            out[i0:i1] += sig[: i1 - i0]
        peak = np.max(np.abs(out)) if out.size else 0.0
        if peak > 0.95:
            out *= 0.95 / peak
        return out


def write_stereo_wav(path: str, performer: np.ndarray, accompaniment: np.ndarray) -> None:
    n = max(len(performer), len(accompaniment))
    stereo = np.zeros((n, 2), dtype=np.float64)
    stereo[: len(performer), 0] = performer
    stereo[: len(accompaniment), 1] = accompaniment
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


class LiveSink:
    """Best-effort live playback. Degrades silently without PortAudio."""

    def __init__(self) -> None:
        self.backend = None
        try:
            import sounddevice as sd  # noqa: F401
            sd.query_devices(kind="output")
            self.backend = sd
            self._streams: dict[int, object] = {}
        except Exception:
            self.backend = None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def note_on(self, pitch: int, velocity: int, velocity_scale: float) -> None:
        if self.backend is None:
            return
        sd = self.backend
        sig = render_note(pitch, 4.0, velocity, velocity_scale)
        stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
        stream.start()
        stream.write(sig.astype("float32"))
        self._streams[pitch] = stream

    def note_off(self, pitch: int) -> None:
        if self.backend is None:
            return
        stream = self._streams.pop(pitch, None)
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def close(self) -> None:
        if self.backend is None:
            return
        for stream in list(self._streams.values()):
            try:
                stream.close()
            except Exception:
                pass
        self._streams.clear()

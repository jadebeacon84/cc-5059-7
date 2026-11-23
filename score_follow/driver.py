"""Run drivers: one online pipeline shared by live input and offline replay.

The replay driver advances a virtual clock to each recorded event and only
then hands that event to the estimator. It never pre-reads future events for
tracking decisions; the file is read once into an ordered list, but the
tracker receives events strictly through an iterator, one at a time. Between
events the virtual clock still ticks, so stalls and timeouts behave exactly
as in a live run.
"""
from __future__ import annotations

import json
import os
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Iterator, Protocol

from .estimator import Estimator
from .scheduler import Scheduler, SystemClock, TimeSource
from .synthesis import AudioBus, LiveSink
from .types import NoteEvent


class EventSource(Protocol):
    def events(self) -> Iterator[NoteEvent]: ...
    def duration(self) -> float: ...


# --------------------------------------------------------------------- #
# Virtual clock for deterministic offline replay
# --------------------------------------------------------------------- #
class VirtualClock:
    def __init__(self) -> None:
        self._t = 0.0

    def now(self) -> float:
        return self._t

    def advance(self, t: float) -> None:
        if t < self._t:
            raise ValueError("virtual clock cannot move backwards")
        self._t = t


# --------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------- #
class ReplaySource:
    """Streams saved events strictly in arrival order."""

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._events = [
            NoteEvent(time=float(e["time"]), kind=e["kind"], pitch=int(e["pitch"]))
            for e in data["events"]
        ]
        self.meta = {k: v for k, v in data.items() if k != "events"}

    def events(self) -> Iterator[NoteEvent]:
        for ev in self._events:
            yield ev

    def duration(self) -> float:
        return self._events[-1].time if self._events else 0.0


@dataclass
class RunResult:
    clock: TimeSource
    estimates: list          # list[Estimate]
    scheduler: Scheduler
    performer_bus: AudioBus
    accompaniment_bus: AudioBus


def run_replay(
    score,
    source: ReplaySource,
    *,
    tick_step: float = 0.02,
) -> RunResult:
    clock = VirtualClock()
    estimator = Estimator(score)
    scheduler = Scheduler(score, clock)
    performer = AudioBus()
    accompaniment = AudioBus()
    estimates: list = []

    it = iter(source.events())
    pending = next(it, None)
    end_time = source.duration() + 2.0

    while True:
        if pending is not None and pending.time <= clock.now() + 1e-12:
            ev = pending
            estimate = estimator.handle_event(ev)
            _record_performer(performer, ev)
            for action, entry in scheduler.update(estimate):
                _emit(action, entry, clock, accompaniment)
            estimates.append(estimate)
            pending = next(it, None)
            continue
        # no event due: advance the virtual clock and poll timeouts
        if pending is None and clock.now() >= end_time:
            break
        nxt = pending.time if pending is not None else end_time
        clock.advance(min(nxt, clock.now() + tick_step))
        estimate = estimator.tick(clock.now())
        for action, entry in scheduler.poll(estimate):
            _emit(action, entry, clock, accompaniment)
        estimates.append(estimate)

    scheduler.finish()
    # close any still-held performer keys at the end for clean rendering
    held = {m.pitch for m in performer.messages if m.kind == "on"}
    for m in list(performer.messages):
        if m.kind == "off":
            held.discard(m.pitch)
    for pitch in held:
        performer.note_off(clock.now(), pitch)
    return RunResult(clock, estimates, scheduler, performer, accompaniment)


def _record_performer(bus: AudioBus, ev: NoteEvent) -> None:
    if ev.kind == "on":
        bus.note_on(ev.time, ev.pitch, 85)
    else:
        bus.note_off(ev.time, ev.pitch)


def _emit(action: str, entry, clock: TimeSource, bus: AudioBus) -> None:
    note = entry.note
    if action == "play":
        t = clock.now()
        bus.note_on(t, note.pitch, note.velocity, entry.velocity_scale)
        bus.note_off(max(t, entry.off_time), note.pitch)
    # "stall" emits nothing: accompaniment deliberately waits


# --------------------------------------------------------------------- #
# Live keyboard input (POSIX terminals): a..k maps to a C major octave
# --------------------------------------------------------------------- #
_KEY_MAP = {
    "a": 60, "w": 61, "s": 62, "e": 63, "d": 64, "f": 65,
    "t": 66, "g": 67, "y": 68, "h": 69, "u": 70, "j": 71,
    "k": 72, "o": 73, "l": 74, "p": 75, ";": 76,
}


class KeyboardSource:
    """Block-free terminal keyboard source using termios raw mode."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        self._held: set[int] = set()
        self._t0 = time.monotonic()

    def __enter__(self) -> "KeyboardSource":
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> list[NoteEvent]:
        import select
        events: list[NoteEvent] = []
        while select.select([self._fd], [], [], 0)[0]:
            ch = os.read(self._fd, 1).decode(errors="ignore").lower()
            if ch in ("\x03", "q"):
                raise KeyboardInterrupt
            pitch = _KEY_MAP.get(ch)
            if pitch is None:
                continue
            now = time.monotonic() - self._t0
            if pitch in self._held:
                self._held.discard(pitch)
                events.append(NoteEvent(now, "off", pitch))
            else:
                self._held.add(pitch)
                events.append(NoteEvent(now, "on", pitch))
        return events


class MidiSource:
    """Optional hardware MIDI source via mido. Raises if no port exists."""

    def __init__(self, port_name: str | None = None) -> None:
        import mido
        self.mido = mido
        self.port = mido.open_input(port_name)
        self._t0 = time.monotonic()

    def poll(self) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        for msg in self.port.iter_pending():
            if msg.type not in ("note_on", "note_off"):
                continue
            now = time.monotonic() - self._t0
            kind = "on" if msg.type == "note_on" and msg.velocity > 0 else "off"
            events.append(NoteEvent(now, kind, msg.note))
        return events

    def close(self) -> None:
        self.port.close()


def run_live(score, source, *, poll_interval: float = 0.01) -> RunResult:
    """Same pipeline as replay, driven by wall-clock polling."""
    clock = SystemClock()
    estimator = Estimator(score)
    scheduler = Scheduler(score, clock)
    performer = AudioBus()
    accompaniment = AudioBus()
    sink = LiveSink()
    estimates: list = []

    print(f"[live] audio backend: {'portaudio' if sink.available else 'unavailable (null sink)'}")
    print("[live] play with keys a s d f g h j k ; (q or Ctrl-C to stop)")
    try:
        while True:
            for ev in source.poll():
                estimate = estimator.handle_event(ev)
                _record_performer(performer, ev)
                for action, entry in scheduler.update(estimate):
                    _live_emit(action, entry, sink)
                estimates.append(estimate)
            tick_estimate = estimator.tick(clock.now())
            for action, entry in scheduler.poll(tick_estimate):
                _live_emit(action, entry, sink)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()
        scheduler.finish()
    return RunResult(clock, estimates, scheduler, performer, accompaniment)


def _live_emit(action: str, entry, sink: LiveSink) -> None:
    note = entry.note
    if action == "play":
        sink.note_on(note.pitch, note.velocity, entry.velocity_scale)
        time.sleep(max(0.05, entry.off_time - entry.play_time))
        sink.note_off(note.pitch)

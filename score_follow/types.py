"""Shared immutable data types.

All timestamps are seconds on a single monotonic clock per run; all score
positions are beats. Nothing here looks ahead at future events.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoreNote:
    """One note in a staff (solo or accompaniment)."""

    start: float          # beat position of onset
    duration: float       # beats
    pitch: int            # MIDI note number
    velocity: int = 80

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class NoteEvent:
    """A physical key action observed from the performer."""

    time: float           # seconds on the run clock
    kind: str             # "on" or "off"
    pitch: int

    def __post_init__(self) -> None:
        if self.kind not in ("on", "off"):
            raise ValueError(f"bad event kind {self.kind!r}")


@dataclass(frozen=True)
class Estimate:
    """Tracker output for one observation point.

    ``positions`` carries every live hypothesis (score beat, weight) so the
    scheduler can deliberately avoid committing when two readings are equally
    plausible (e.g. a phrase played twice with the tail beat skipped).
    """

    time: float
    positions: tuple[tuple[float, float], ...]
    best_beat: float
    best_index: int
    tempo: float                      # beats per second
    confidence: float                 # 0..1
    margin: float                     # weight gap to second-best reading
    expected_index: int
    event_count: int
    is_onset: bool = False


@dataclass
class ScheduledNote:
    """A planned accompaniment event with mutable lifecycle state.

    State advances pending -> armed -> played/cancelled. Once played, the
    entry is frozen: later estimates never rewrite or retract it.
    """

    note: ScoreNote
    play_time: float
    off_time: float
    state: str = "pending"           # pending, armed, played, cancelled
    velocity_scale: float = 1.0

    def freeze(self) -> None:
        object.__setattr__(self, "state", "played")

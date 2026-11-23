"""Accompaniment scheduler.

The scheduler owns its own clock (an injectable ``TimeSource``) and never
shares state with the estimator beyond reading ``Estimate`` objects.

Rules implemented here:

* a note inside ``lock_horizon`` seconds of sounding is frozen — later
  estimates can move or cancel only notes further in the future;
* once a note has sounded it can never be retracted or rewritten;
* when tracker confidence is low the scheduler either softens the entry
  (velocity scaled down) or waits instead of marching into the wrong phrase;
* every sounding and every stall is logged on the scheduler clock so the
  accompaniment lag can be measured offline against the ground-truth beat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .estimator import Estimator
from .score import Score
from .types import ScheduledNote, ScoreNote


class TimeSource(Protocol):
    def now(self) -> float: ...


class SystemClock:
    __slots__ = ("_t0",)

    def __init__(self) -> None:
        import time
        self._time = time
        self._t0 = time.monotonic()

    def now(self) -> float:
        return self._time.monotonic() - self._t0


@dataclass
class SchedulerLog:
    note: ScoreNote
    scheduled_time: float
    played_time: float | None
    target_beat: float
    confidence: float
    velocity_scale: float
    stalled: bool = False
    logged_at: float = 0.0


class Scheduler:
    def __init__(
        self,
        score: Score,
        clock: TimeSource,
        *,
        lock_horizon: float = 0.12,
        adjust_window: float = 1.2,
        soft_confidence: float = 0.55,
        wait_confidence: float = 0.3,
        stall_after: float = 2.5,
    ) -> None:
        self.score = score
        self.clock = clock
        self.lock_horizon = lock_horizon
        self.adjust_window = adjust_window
        self.soft_confidence = soft_confidence
        self.wait_confidence = wait_confidence
        self.stall_after = stall_after
        self.plan: list[ScheduledNote] = []
        self.log: list[SchedulerLog] = []
        self._last_event_time: float | None = None
        self._last_event_count = -1
        self._last_estimate: Estimate | None = None
        self._build_initial_plan()

    def _build_initial_plan(self) -> None:
        t0 = self.clock.now()
        tempo = self.score.initial_tempo_bpm / 60.0
        for note in self.score.accompaniment:
            self.plan.append(
                ScheduledNote(
                    note=note,
                    play_time=t0 + note.start / tempo,
                    off_time=t0 + note.end / tempo,
                )
            )

    # ------------------------------------------------------------------ #
    def update(self, estimate: Estimate) -> list[tuple[str, ScheduledNote]]:
        """Re-plan the movable window and emit any notes due now.

        Returns the actions taken this call: ("play", note) or ("stall", note).
        """
        now = self.clock.now()
        self._last_estimate = estimate
        if estimate.event_count > 0 and estimate.event_count != self._last_event_count:
            self._last_event_time = estimate.time
            self._last_event_count = estimate.event_count
        self._replan(estimate, now)
        return self._due(estimate, now)

    def poll(self, estimate: Estimate | None = None) -> list[tuple[str, ScheduledNote]]:
        """Clock-only wake-up: notes must still fire even with no new events.

        A fresh tick estimate (lower confidence after silence) supersedes the
        cached one so wait/soften policy applies on idle timeouts as well.
        """
        now = self.clock.now()
        if estimate is not None:
            self._last_estimate = estimate
        current = self._last_estimate
        if current is None:
            return []
        return self._due(current, now)

    # ------------------------------------------------------------------ #
    def _replan(self, estimate: Estimate, now: float) -> None:
        for entry in self.plan:
            if entry.state in ("played", "cancelled", "armed"):
                continue
            time_to_sound = entry.play_time - now
            if time_to_sound <= self.lock_horizon:
                # close enough to be concretely scheduled; the hard freeze
                # still only happens at sound time inside _due
                entry.state = "armed"
                continue
            if time_to_sound > self.adjust_window:
                continue  # too far ahead to plan concretely
            projected = self._project(entry.note.start, estimate, now)
            # never let a future note jump inside the lock horizon
            entry.play_time = max(now + self.lock_horizon, projected)
            entry.off_time = entry.play_time + entry.note.duration / max(estimate.tempo, 1e-6)
            entry.velocity_scale = self._velocity_scale(estimate)

    def _project(self, beat: float, estimate: Estimate, now: float) -> float:
        delta_beats = beat - estimate.best_beat
        return now + delta_beats / max(estimate.tempo, 1e-6)

    def _velocity_scale(self, estimate: Estimate) -> float:
        if estimate.confidence >= self.soft_confidence:
            return 1.0
        if estimate.confidence <= self.wait_confidence:
            return 0.25
        span = self.soft_confidence - self.wait_confidence
        frac = (estimate.confidence - self.wait_confidence) / span
        return 0.25 + 0.75 * frac

    def _due(self, estimate: Estimate, now: float) -> list[tuple[str, ScheduledNote]]:
        actions: list[tuple[str, ScheduledNote]] = []
        stalled = self._is_stalled(estimate, now)
        # global gate: nothing sounds before the first credible performer
        # evidence; until then every entry is held, not played
        gate_closed = self._last_event_time is None
        for entry in self.plan:
            if entry.state == "pending" and entry.play_time - now <= self.lock_horizon:
                entry.state = "armed"
            if entry.state != "armed":
                continue
            if entry.play_time > now:
                continue
            if gate_closed:
                entry.play_time = now + 0.1
                entry.off_time = entry.play_time + entry.note.duration / max(estimate.tempo, 1e-6)
                self.log.append(
                    SchedulerLog(
                        note=entry.note,
                        scheduled_time=entry.play_time,
                        played_time=None,
                        target_beat=entry.note.start,
                        confidence=estimate.confidence,
                        velocity_scale=0.0,
                        stalled=True,
                        logged_at=now,
                    )
                )
                actions.append(("stall", entry))
                continue
            if stalled:
                # explicit policy: do not keep playing confidently into a
                # wrong phrase; hold the entry and record the stall.
                # Keep its planned time aligned to the current beat clock so
                # that, when evidence recovers, it re-enters on the beat
                # instead of firing in a compressed burst.
                aligned = self._project(entry.note.start, estimate, now)
                entry.play_time = max(now + 0.15, aligned)
                entry.off_time = entry.play_time + entry.note.duration / max(estimate.tempo, 1e-6)
                self.log.append(
                    SchedulerLog(
                        note=entry.note,
                        scheduled_time=entry.play_time,
                        played_time=None,
                        target_beat=entry.note.start,
                        confidence=estimate.confidence,
                        velocity_scale=0.0,
                        stalled=True,
                        logged_at=now,
                    )
                )
                actions.append(("stall", entry))
                continue
            entry.velocity_scale = self._velocity_scale(estimate)
            entry.freeze()
            self.log.append(
                SchedulerLog(
                    note=entry.note,
                    scheduled_time=entry.play_time,
                    played_time=now,
                    target_beat=entry.note.start,
                    confidence=estimate.confidence,
                    velocity_scale=entry.velocity_scale,
                    logged_at=now,
                )
            )
            actions.append(("play", entry))
        return actions

    def _is_stalled(self, estimate: Estimate, now: float) -> bool:
        if estimate.confidence > self.wait_confidence:
            return False
        if self._last_event_time is None:
            return now >= self.stall_after
        return now - self._last_event_time >= self.stall_after

    def finish(self) -> None:
        """Cancel everything still unplayed once the score is exhausted."""
        for entry in self.plan:
            if entry.state in ("pending", "armed"):
                entry.state = "cancelled"

    @property
    def played_notes(self) -> list[SchedulerLog]:
        return [row for row in self.log if row.played_time is not None]

"""Score model: short pieces stored as JSON plus a built-in demo piece.

A score has a solo staff (what the performer plays, used by the estimator)
and an accompaniment staff (driven by the scheduler). Time is notated in
beats; seconds only enter through the performer clock.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .types import ScoreNote


@dataclass
class Score:
    name: str
    solo: list[ScoreNote]
    accompaniment: list[ScoreNote]
    initial_tempo_bpm: float = 90.0
    beats_per_bar: int = 4
    # repeat regions: (first_beat, last_beat_inclusive). A performer reaching
    # the end of a region may legitimately jump back to its first beat.
    repeats: tuple[tuple[float, float], ...] = ()

    @property
    def length_beats(self) -> float:
        ends = [n.end for n in self.solo] + [n.end for n in self.accompaniment]
        return max(ends) if ends else 0.0

    def note_at(self, index: int) -> ScoreNote | None:
        if 0 <= index < len(self.solo):
            return self.solo[index]
        return None

    def repeat_targets(self) -> tuple[float, ...]:
        return tuple(first for first, _ in self.repeats)


def _parse_staff(rows: list[dict]) -> list[ScoreNote]:
    notes = []
    for row in rows:
        notes.append(
            ScoreNote(
                start=float(row["start"]),
                duration=float(row.get("duration", 1.0)),
                pitch=int(row["pitch"]),
                velocity=int(row.get("velocity", 80)),
            )
        )
    notes.sort(key=lambda n: n.start)
    return notes


def load_score(path: str) -> Score:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Score(
        name=data.get("name", path),
        solo=_parse_staff(data["solo"]),
        accompaniment=_parse_staff(data.get("accompaniment", [])),
        initial_tempo_bpm=float(data.get("initial_tempo_bpm", 90.0)),
        beats_per_bar=int(data.get("beats_per_bar", 4)),
        repeats=tuple(
            (float(r[0]), float(r[1])) for r in data.get("repeats", [])
        ),
    )


def demo_score() -> Score:
    """Built-in piece.

    Phrase A (beats 0..8) is meant to be played twice; its last note lands on
    the tail beat (beat 8), which the test trajectory skips on the first pass.
    Phrase B (beats 10..16) is different, so arriving notes disambiguate the
    two retained readings.
    """
    solo_rows = [
        # phrase A
        (0.0, 1.0, 60), (1.0, 1.0, 62), (2.0, 1.0, 64), (3.0, 1.0, 65),
        (4.0, 1.0, 64), (5.0, 1.0, 62), (6.0, 1.0, 60), (7.0, 1.0, 62),
        # tail beat note: may be skipped
        (8.0, 1.0, 60),
        # phrase B (different contour -> fresh evidence)
        (10.0, 1.0, 67), (11.0, 1.0, 65), (12.0, 1.0, 64), (13.0, 1.0, 62),
        (14.0, 1.0, 60), (15.0, 1.0, 62), (16.0, 2.0, 60),
    ]
    solo = [ScoreNote(start=s, duration=d, pitch=p) for s, d, p in solo_rows]

    acc_rows: list[tuple[float, float, int, int]] = []
    # bass on bar starts, chord on beat 2 of each bar
    progression = [(48, 52), (43, 47), (50, 53), (45, 48), (48, 52)]
    for bar in range(5):
        bar_beat = bar * 4.0
        bass, chord_root = progression[bar]
        acc_rows.append((bar_beat, 1.6, bass, 70))
        acc_rows.append((bar_beat + 2.0, 1.2, chord_root + 12, 55))
        acc_rows.append((bar_beat + 2.0, 1.2, chord_root + 16, 50))
    acc = [
        ScoreNote(start=s, duration=d, pitch=p, velocity=v)
        for s, d, p, v in acc_rows
    ]
    return Score(
        name="demo_repeat_phrase",
        solo=solo,
        accompaniment=acc,
        initial_tempo_bpm=90.0,
        beats_per_bar=4,
        # phrase A spans beats 0..9 (tail note on beat 8); it is played twice
        repeats=((0.0, 9.0),),
    )

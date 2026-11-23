"""Synthetic performance trajectory generation (offline tool).

Generated trajectories are JSON files consumed by the *online* replay driver:
generation may plan the whole piece, but tracking still only ever sees one
event at a time. Every onset keeps a ground-truth score index for analysis.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .score import Score
from .types import NoteEvent


@dataclass
class GenConfig:
    seed: int = 7
    base_bpm: float | None = None
    tempo_drift: float = 0.0          # fractional tempo change across the piece
    timing_jitter: float = 0.03       # seconds of Gaussian onset jitter
    wrong_note_rate: float = 0.0
    missed_note_rate: float = 0.0
    repeat_press_rate: float = 0.0    # extra on-off pairs on a note
    pause_beats: tuple[float, ...] = ()   # score beats after which a pause is inserted
    pause_seconds: float = 1.5
    skip_indices: tuple[int, ...] = ()    # explicit solo indices to omit (tail beats)
    repeat_region: tuple[float, float] | None = None  # play this region twice
    repeat_skip_tail: bool = False    # on the second pass, drop the region's tail


def generate(score: Score, cfg: GenConfig) -> dict:
    rng = random.Random(cfg.seed)
    base_tempo = (cfg.base_bpm or score.initial_tempo_bpm) / 60.0
    events: list[dict] = []
    ground_truth: list[dict] = []

    # build the performance order; a repeat region inserts its note indices
    # a second time, optionally with the region's tail note dropped
    order: list[int] = list(range(len(score.solo)))
    if cfg.repeat_region is not None:
        first_beat, last_beat = cfg.repeat_region
        region_idx = [
            i for i, note in enumerate(score.solo)
            if first_beat <= note.start <= last_beat
        ]
        tail_idx = max(region_idx, key=lambda i: score.solo[i].start)
        insert_at = max(region_idx) + 1
        second_pass = [i for i in region_idx if not (cfg.repeat_skip_tail and i == tail_idx)]
        order = order[:insert_at] + second_pass + order[insert_at:]

    clock = 0.0
    prev_beat = 0.0
    total_steps = len(order)

    for step, i in enumerate(order):
        note = score.solo[i]
        progress = step / max(1, total_steps - 1)
        tempo = base_tempo * (1.0 + cfg.tempo_drift * progress)
        if step > 0 and note.start <= prev_beat:
            # backwards jump into a repeated region: one beat of reset gap
            gap_beats = 1.0
        else:
            gap_beats = note.start - prev_beat
        clock += gap_beats / tempo
        onset = max(0.0, clock + rng.gauss(0.0, cfg.timing_jitter))

        if note.start in cfg.pause_beats:
            clock += cfg.pause_seconds

        if i in cfg.skip_indices or rng.random() < cfg.missed_note_rate:
            ground_truth.append({"index": i, "beat": note.start, "played": False})
            prev_beat = note.start
            continue

        pitch = note.pitch
        if rng.random() < cfg.wrong_note_rate:
            pitch = note.pitch + rng.choice([-2, -1, 1, 2])

        events.append({"time": round(onset, 4), "kind": "on", "pitch": pitch,
                       "gt_index": i, "gt_beat": note.start})
        # durations jitter a little too; off before the next onset
        dur = max(0.12, note.duration / tempo * rng.uniform(0.7, 0.95))
        events.append({"time": round(onset + dur, 4), "kind": "off", "pitch": pitch})

        if rng.random() < cfg.repeat_press_rate:
            rep_on = onset + dur + 0.06
            events.append({"time": round(rep_on, 4), "kind": "on", "pitch": pitch,
                           "gt_index": i, "gt_beat": note.start, "repeated": True})
            events.append({"time": round(rep_on + 0.09, 4), "kind": "off", "pitch": pitch})

        ground_truth.append({"index": i, "beat": note.start, "played": True,
                             "time": round(onset, 4)})
        prev_beat = note.start

    events.sort(key=lambda e: (e["time"], 0 if e["kind"] == "on" else 1))
    return {
        "score": score.name,
        "config": {
            "seed": cfg.seed,
            "tempo_drift": cfg.tempo_drift,
            "timing_jitter": cfg.timing_jitter,
            "wrong_note_rate": cfg.wrong_note_rate,
            "missed_note_rate": cfg.missed_note_rate,
            "repeat_press_rate": cfg.repeat_press_rate,
            "pause_beats": list(cfg.pause_beats),
            "pause_seconds": cfg.pause_seconds,
            "skip_indices": list(cfg.skip_indices),
            "repeat_region": list(cfg.repeat_region) if cfg.repeat_region else None,
            "repeat_skip_tail": cfg.repeat_skip_tail,
        },
        "events": events,
        "ground_truth": ground_truth,
    }


def save_trajectory(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)

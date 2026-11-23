"""Offline analysis of a replay run against trajectory ground truth.

Metrics:
  * position error — |estimated score beat - ground-truth beat| per onset;
  * accompaniment lag — for every accompaniment note, its sounding time minus
    the projected ideal time (solo clock mapped through ground truth);
  * stall behaviour — scheduler waits/softens while confidence is low;
  * audio summary — rendered bus durations and RMS, confirming real sound.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .driver import RunResult
from .synthesis import SR


@dataclass
class Report:
    n_onsets: int
    position_errors_beats: list
    mean_abs_position_error: float
    max_abs_position_error: float
    correct_within_half_beat: float
    lag_seconds: list
    mean_abs_lag: float
    n_played_acc: int
    n_stalls: int
    min_confidence_on_play: float
    ambiguous_windows: int
    performer_seconds: float
    accompaniment_seconds: float
    performer_rms: float
    accompaniment_rms: float

    def to_dict(self) -> dict:
        return {
            "n_onsets": self.n_onsets,
            "mean_abs_position_error_beats": round(self.mean_abs_position_error, 4),
            "max_abs_position_error_beats": round(self.max_abs_position_error, 4),
            "fraction_within_half_beat": round(self.correct_within_half_beat, 3),
            "mean_abs_accompaniment_lag_seconds": round(self.mean_abs_lag, 4),
            "accompaniment_notes_played": self.n_played_acc,
            "accompaniment_stalls": self.n_stalls,
            "min_confidence_when_playing": round(self.min_confidence_on_play, 3),
            "ambiguous_position_windows": self.ambiguous_windows,
            "audio": {
                "performer_seconds": round(self.performer_seconds, 3),
                "accompaniment_seconds": round(self.accompaniment_seconds, 3),
                "performer_rms": round(self.performer_rms, 5),
                "accompaniment_rms": round(self.accompaniment_rms, 5),
            },
            "position_error_series": [round(x, 3) for x in self.position_errors_beats],
            "lag_series_seconds": [round(x, 3) for x in self.lag_seconds],
        }


def _ground_truth_onsets(trajectory: dict) -> list[tuple[int, float, float]]:
    # ordered (index, beat, time) for every actually played onset; a repeated
    # region legitimately yields multiple rows with the same index
    return [
        (int(row["index"]), float(row["beat"]), float(row["time"]))
        for row in trajectory["ground_truth"]
        if row.get("played")
    ]


def analyse(result: RunResult, trajectory: dict, score) -> Report:
    gt_onsets = _ground_truth_onsets(trajectory)

    # ---- position error: onset-only estimates vs ground-truth beat ------
    # tick estimates and repeated-press insertions are excluded; the tracker
    # is judged exactly when a new score-note onset evidence arrives.
    errors: list[float] = []
    ambiguous = 0
    gt_cursor = 0
    for est in result.estimates:
        if not getattr(est, "is_onset", False):
            continue
        # consume the next ground-truth onset at or before this estimate;
        # repeated presses (no matching GT onset) are skipped, not scored
        while gt_cursor < len(gt_onsets) and gt_onsets[gt_cursor][2] > est.time + 1e-6:
            break
        if gt_cursor >= len(gt_onsets):
            break
        idx, beat, gt_time = gt_onsets[gt_cursor]
        if abs(gt_time - est.time) > 0.35:
            continue  # insertion / repeated press with no score onset
        gt_cursor += 1
        errors.append(abs(est.best_beat - beat))
        if est.margin < 0.35 and len(est.positions) > 1:
            ambiguous += 1

    # ---- accompaniment lag, occurrence-aware for repeated beats ---------
    # Build the ideal accompaniment clock from the solo ground truth by
    # walking it in performance order. Each accompaniment note is matched to
    # the closest ground-truth occurrence of its target beat (a repeated beat
    # has several), so repeats do not create artificial multi-second lags.
    gt_by_beat: dict[float, list[float]] = {}
    for _, beat, t in gt_onsets:
        gt_by_beat.setdefault(beat, []).append(t)
    for times in gt_by_beat.values():
        times.sort()
    monotone_pts = sorted((beat, times[-1]) for beat, times in gt_by_beat.items())
    lags: list[float] = []
    for row in result.scheduler.played_notes:
        if row.played_time is None:
            continue
        candidates = gt_by_beat.get(row.target_beat)
        if candidates:
            ideal = min(candidates, key=lambda t: abs(t - row.played_time))
        else:
            ideal = _interp_beat_time(monotone_pts, row.target_beat)
        if ideal is not None:
            lags.append(row.played_time - ideal)

    played = result.scheduler.played_notes
    stalls = [r for r in result.scheduler.log if r.stalled]
    min_conf = min((r.confidence for r in played), default=1.0)

    perf_sig = result.performer_bus.render()
    acc_sig = result.accompaniment_bus.render()
    return Report(
        n_onsets=len(gt_onsets),
        position_errors_beats=errors,
        mean_abs_position_error=float(np.mean(errors)) if errors else math.nan,
        max_abs_position_error=float(np.max(errors)) if errors else math.nan,
        correct_within_half_beat=float(np.mean([e <= 0.5 for e in errors])) if errors else 0.0,
        lag_seconds=lags,
        mean_abs_lag=float(np.mean(np.abs(lags))) if lags else math.nan,
        n_played_acc=len(played),
        n_stalls=len(stalls),
        min_confidence_on_play=min_conf,
        ambiguous_windows=ambiguous,
        performer_seconds=len(perf_sig) / SR,
        accompaniment_seconds=len(acc_sig) / SR,
        performer_rms=float(np.sqrt(np.mean(perf_sig ** 2))) if perf_sig.size else 0.0,
        accompaniment_rms=float(np.sqrt(np.mean(acc_sig ** 2))) if acc_sig.size else 0.0,
    )


def _interp_beat_time(points: list[tuple[float, float]], beat: float) -> float | None:
    if not points:
        return None
    if beat <= points[0][0]:
        b0, t0 = points[0]
        b1, t1 = points[1] if len(points) > 1 else (b0 + 1.0, t0 + 1.0)
        return t0 + (beat - b0) * (t1 - t0) / max(1e-9, b1 - b0)
    if beat >= points[-1][0]:
        b1, t1 = points[-1]
        b0, t0 = points[-2] if len(points) > 1 else (b1 - 1.0, t1 - 1.0)
        return t1 + (beat - b1) * (t1 - t0) / max(1e-9, b1 - b0)
    for (b0, t0), (b1, t1) in zip(points, points[1:]):
        if b0 <= beat <= b1:
            return t0 + (beat - b0) * (t1 - t0) / max(1e-9, b1 - b0)
    return None


def write_report(path: str, report: Report) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

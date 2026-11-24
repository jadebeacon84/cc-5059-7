"""Performance sample for the candidate-update path (committed reference runs).

Every scenario below is generated, then replayed through the *online*
pipeline (estimator + scheduler + audio buses), so the accompaniment run
stays in the loop as the reference frame for every measurement:

  jump_after_repeat  phrase A played twice (tail skipped on the repeat),
                     phrase B, then a sudden forward jump to phrase D -
                     stresses mis-tracking (错追) and relock latency;
  load               full piece with 15% missed notes and a 50% tempo
                     acceleration - raises candidate load on the estimator;
  late_and_missed    late input (a 2 s pause) plus 12% missed notes - the
                     tracker must keep its uncertainty instead of marching
                     on at a fixed speed.

A report is only written after a complete run and carries the per-onset
evidence (event latency, candidate counts, position error, confidence) that
each aggregate metric is computed from, so committed artifacts trace back to
execution. Reports are never edited by hand.

Usage:
  python -m score_follow.benchmark run --variant baseline --outdir data/benchmark
  python -m score_follow.benchmark compare --outdir data/benchmark
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict

from .analysis import analyse
from .driver import ReplaySource, run_replay
from .estimator import Estimator
from .score import Score
from .types import ScoreNote

BENCH_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "benchmark")
)

# Estimator kwargs per candidate-update variant, all explicit so the
# reference run stays reproducible no matter which variant later becomes
# the Estimator default:
#   baseline  pre-optimization update: every hypothesis expands its full
#             forward window and may propose repeat jumps;
#   budget    fixed far-jump budget tier: a reserved repeat-jump budget the
#             local tier cannot starve, plus a capped local budget spent on
#             the strongest readings first;
#   entropy   candidate-entropy-adaptive window: narrow when the beam is
#             confident, widened only while the reading is undecided.
VARIANTS: dict[str, dict] = {
    "baseline": {
        "local_budget": None,
        "far_jump_budget": None,
        "far_jump_proposers": None,
        "adaptive_window": False,
    },
    "budget": {
        "local_budget": 128,
        "far_jump_budget": 24,
        "far_jump_proposers": 8,
        "adaptive_window": False,
    },
    "entropy": {
        "local_budget": None,
        "far_jump_budget": None,
        "far_jump_proposers": None,
        "adaptive_window": True,
    },
}


def bench_score() -> Score:
    """Four-phrase piece; phrase A is a notated repeat with a tail beat."""
    solo_rows = [
        # phrase A (beats 0..8, tail note on beat 8)
        (0, 60), (1, 62), (2, 64), (3, 65), (4, 64), (5, 62), (6, 60),
        (7, 62), (8, 60),
        # phrase B (beats 10..17)
        (10, 67), (11, 65), (12, 64), (13, 62), (14, 60), (15, 62),
        (16, 64), (17, 65),
        # phrase C (beats 18..25)
        (18, 66), (19, 64), (20, 62), (21, 60), (22, 62), (23, 64),
        (24, 66), (25, 67),
        # phrase D (beats 26..33)
        (26, 72), (27, 71), (28, 69), (29, 67), (30, 65), (31, 64),
        (32, 62), (33, 60),
    ]
    solo = [ScoreNote(float(b), 1.0, p) for b, p in solo_rows]
    acc: list[ScoreNote] = []
    roots = [48, 45, 50, 43, 48, 45, 50, 43, 48]
    for bar in range(9):
        bar_beat = float(bar * 4)
        acc.append(ScoreNote(bar_beat, 1.8, roots[bar], 70))
        acc.append(ScoreNote(bar_beat + 2.0, 1.2, roots[bar] + 7, 55))
        acc.append(ScoreNote(bar_beat + 2.0, 1.2, roots[bar] + 12, 50))
    return Score(
        name="bench_sections",
        solo=solo,
        accompaniment=acc,
        initial_tempo_bpm=120.0,
        beats_per_bar=4,
        repeats=((0.0, 9.0),),
    )


@dataclass
class LoadConfig:
    seed: int = 21
    base_bpm: float = 120.0
    accel: float = 0.0               # fractional tempo increase across the run
    timing_jitter: float = 0.02      # seconds of Gaussian onset jitter
    missed_note_rate: float = 0.0
    repeat_press_rate: float = 0.0
    pause_after_beats: tuple = ()    # score beats after which input arrives late
    pause_seconds: float = 2.0


# performance order as solo-note indices; phrase A is 0..8, B is 9..16,
# C is 17..24, D is 25..32
_ORDERS = {
    # A, A again without the tail note, B, then a sudden jump over C into D
    "jump_after_repeat": (
        list(range(0, 9)) + list(range(0, 8)) + list(range(9, 17))
        + list(range(25, 33))
    ),
    "load": list(range(0, 9)) + list(range(0, 8)) + list(range(9, 33)),
    "late_and_missed": list(range(0, 9)) + list(range(0, 8)) + list(range(9, 33)),
}

SCENARIOS = {
    "jump_after_repeat": LoadConfig(
        seed=21, accel=0.35, missed_note_rate=0.06, timing_jitter=0.02,
    ),
    "load": LoadConfig(
        seed=22, accel=0.5, missed_note_rate=0.15, repeat_press_rate=0.1,
        timing_jitter=0.03,
    ),
    "late_and_missed": LoadConfig(
        seed=23, accel=0.2, missed_note_rate=0.12, pause_after_beats=(12.0,),
        pause_seconds=2.0,
    ),
}

# first beat of phrase D: the target of the sudden jump in jump_after_repeat
JUMP_TARGET_BEAT = 26.0


def build_trajectory(score: Score, order: list[int], cfg: LoadConfig) -> dict:
    """Deterministic trajectory for an explicit performance order.

    Same ground-truth format as score_follow.trajectory.generate so the
    offline analysis works unchanged; repeated indices legitimately appear
    several times.
    """
    rng = random.Random(cfg.seed)
    base_tempo = cfg.base_bpm / 60.0
    events: list[dict] = []
    ground_truth: list[dict] = []
    clock = 0.0
    prev_beat = 0.0
    total = len(order)
    for step, i in enumerate(order):
        note = score.solo[i]
        tempo = base_tempo * (1.0 + cfg.accel * step / max(1, total - 1))
        if step > 0 and note.start <= prev_beat:
            gap_beats = 1.0  # backwards jump into the repeated region
        else:
            gap_beats = note.start - prev_beat
        clock += gap_beats / tempo
        onset = max(0.0, clock + rng.gauss(0.0, cfg.timing_jitter))
        if note.start in cfg.pause_after_beats:
            clock += cfg.pause_seconds  # late input: the next onset is late
        if rng.random() < cfg.missed_note_rate:
            ground_truth.append({"index": i, "beat": note.start, "played": False})
            prev_beat = note.start
            continue
        events.append({"time": round(onset, 4), "kind": "on", "pitch": note.pitch,
                       "gt_index": i, "gt_beat": note.start})
        dur = max(0.12, note.duration / tempo * 0.8)
        events.append({"time": round(onset + dur, 4), "kind": "off",
                       "pitch": note.pitch})
        if rng.random() < cfg.repeat_press_rate:
            rep_on = onset + dur + 0.06
            events.append({"time": round(rep_on, 4), "kind": "on",
                           "pitch": note.pitch, "gt_index": i,
                           "gt_beat": note.start, "repeated": True})
            events.append({"time": round(rep_on + 0.09, 4), "kind": "off",
                           "pitch": note.pitch})
        ground_truth.append({"index": i, "beat": note.start, "played": True,
                             "time": round(onset, 4)})
        prev_beat = note.start
    events.sort(key=lambda e: (e["time"], 0 if e["kind"] == "on" else 1))
    return {
        "score": score.name,
        "config": asdict(cfg),
        "events": events,
        "ground_truth": ground_truth,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def _relock_and_mistrack(per_onset: list[dict]) -> dict:
    """Mis-tracking and relock latency around the sudden section jump.

    Counts onsets after the jump target where the tracker is confidently
    wrong (>2 beats off with confidence >0.5) and how many onsets pass
    before the reading settles within half a beat for two in a row.
    """
    after = [row for row in per_onset if row["gt_beat"] is not None
             and row["gt_beat"] >= JUMP_TARGET_BEAT - 1e-9]
    mistracked = sum(
        1 for row in after
        if abs(row["best_beat"] - row["gt_beat"]) > 2.0 and row["confidence"] > 0.5
    )
    relock = None
    streak = 0
    for n, row in enumerate(after):
        if abs(row["best_beat"] - row["gt_beat"]) <= 0.5:
            streak += 1
            if streak >= 2:
                relock = n + 1
                break
        else:
            streak = 0
    return {
        "onsets_after_jump": len(after),
        "mistracked_onsets": mistracked,
        "relock_events_after_jump": relock,
    }


def _run_once(score: Score, traj: dict, traj_path: str, variant: str,
              profile: bool):
    """One complete replay of the same trajectory; returns wall time, the
    estimator (with its cost counters), per-onset evidence and the result."""
    estimator = Estimator(score, profile=profile, **VARIANTS[variant])
    per_onset: list[dict] = []
    gt_by_time = sorted(
        (float(g["time"]), float(g["beat"]))
        for g in traj["ground_truth"] if g.get("played")
    )
    gt_cursor = [0]
    cand_snap = [0]

    def on_event(ev, estimate, latency):
        if ev.kind != "on":
            return
        # match the nearest not-yet-consumed ground-truth onset (same rule
        # as analysis.analyse) so each evidence row carries its true beat
        gt_beat = None
        while gt_cursor[0] < len(gt_by_time):
            t, b = gt_by_time[gt_cursor[0]]
            if t < ev.time - 0.35:
                gt_cursor[0] += 1  # stale unmatched row; skip it
                continue
            if abs(t - ev.time) <= 0.35:
                gt_beat = b
                gt_cursor[0] += 1
            break
        stats = estimator.stats
        cand_total = (
            stats["cand_insertion"] + stats["cand_forward"] + stats["cand_repeat"]
        )
        per_onset.append({
            "t": round(ev.time, 4),
            "pitch": ev.pitch,
            "latency_ms": round(latency * 1000.0, 4),
            "candidates": cand_total - cand_snap[0],
            "best_beat": round(estimate.best_beat, 4),
            "gt_beat": gt_beat,
            "confidence": round(estimate.confidence, 6),
            "expected_index": estimate.expected_index,
        })
        cand_snap[0] = cand_total

    started = time.perf_counter()
    result = run_replay(
        score, ReplaySource(traj_path), estimator=estimator, on_event=on_event
    )
    wall = time.perf_counter() - started
    return wall, estimator, per_onset, result


def run_scenario(
    scenario: str, variant: str, outdir: str, *, profile: bool = True,
    repeats: int = 3,
) -> dict:
    """Full replay runs; the report comes from the median-wall-time run.

    The trajectory is deterministic, so every repeat produces identical
    behaviour; only wall-clock measurements differ, and the median run is
    the one written down.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    score = bench_score()
    cfg = SCENARIOS[scenario]
    traj = build_trajectory(score, _ORDERS[scenario], cfg)
    os.makedirs(outdir, exist_ok=True)
    traj_path = os.path.join(outdir, f"{scenario}_trajectory.json")
    with open(traj_path, "w", encoding="utf-8") as fh:
        json.dump(traj, fh, indent=1)

    runs = [
        _run_once(score, traj, traj_path, variant, profile)
        for _ in range(max(1, repeats))
    ]
    runs.sort(key=lambda row: row[0])
    wall, estimator, per_onset, result = runs[len(runs) // 2]
    report = analyse(result, traj, score)

    on_latencies = [row["latency_ms"] for row in per_onset]
    behavior = {
        "expected_index_series": [row["expected_index"] for row in per_onset],
        "confidence_series": [row["confidence"] for row in per_onset],
        "ambiguous_windows": report.ambiguous_windows,
        "n_stalls": report.n_stalls,
        "n_played_acc": report.n_played_acc,
    }
    return {
        "scenario": scenario,
        "variant": variant,
        "score": score.name,
        "load": asdict(cfg),
        "wall_seconds": round(wall, 4),
        "n_events": len(traj["events"]),
        "n_estimates": len(result.estimates),
        "event_latency_ms": {
            "mean": round(statistics.fmean(on_latencies), 4) if on_latencies else math.nan,
            "p95": round(_percentile(on_latencies, 0.95), 4),
            "max": round(max(on_latencies), 4) if on_latencies else math.nan,
        },
        "candidate_stats": dict(estimator.stats),
        "phase_seconds": {k: round(v, 6) for k, v in estimator.phase_seconds.items()},
        "position": {
            "mean_abs_error_beats": round(report.mean_abs_position_error, 4),
            "max_abs_error_beats": round(report.max_abs_position_error, 4),
            "within_half_beat": round(report.correct_within_half_beat, 4),
            **_relock_and_mistrack(per_onset),
        },
        "accompaniment": {
            "mean_abs_lag_seconds": round(report.mean_abs_lag, 4),
            "played": report.n_played_acc,
            "stalls": report.n_stalls,
        },
        "behavior": behavior,
        "evidence": {
            "per_onset": per_onset,
            "audio": {
                "performer_seconds": round(report.performer_seconds, 3),
                "accompaniment_seconds": round(report.accompaniment_seconds, 3),
                "accompaniment_rms": round(report.accompaniment_rms, 5),
            },
        },
    }


def write_report(report: dict, outdir: str) -> str:
    path = os.path.join(outdir, f"{report['scenario']}__{report['variant']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return path


def _behavior_close(base: dict, new: dict, *, tol: float = 0.02) -> bool:
    """Behaviour may only move within noise outside the two metrics the
    speedup is allowed to affect (position error and event latency).

    Qualitative behaviour (stalls, played accompaniment, ambiguous windows)
    must match exactly; the tracked reading must match on every *scored*
    onset (insertions are not scored, matching analysis.analyse); confidence
    must stay within noise tolerance so late input and missed notes keep
    their original uncertainty.
    """
    b, n = base["behavior"], new["behavior"]
    if b["n_stalls"] != n["n_stalls"] or b["n_played_acc"] != n["n_played_acc"]:
        return False
    if b["ambiguous_windows"] != n["ambiguous_windows"]:
        return False
    scored_b, scored_n = [], []
    for row in base["evidence"]["per_onset"]:
        if row["gt_beat"] is not None:
            scored_b.append(row["expected_index"])
    for row in new["evidence"]["per_onset"]:
        if row["gt_beat"] is not None:
            scored_n.append(row["expected_index"])
    if scored_b != scored_n:
        return False
    return all(
        abs(x - y) <= tol
        for x, y in zip(b["confidence_series"], n["confidence_series"])
    )


def compare_reports(base: dict, new: dict) -> dict:
    """Metric deltas of ``new`` against the ``base`` reference run."""
    def pct(a: float, b: float) -> float | None:
        if a == 0 or math.isnan(a) or math.isnan(b):
            return None
        return round((b - a) / a * 100.0, 2)

    return {
        "scenario": base["scenario"],
        "base_variant": base["variant"],
        "new_variant": new["variant"],
        "latency_mean_pct": pct(base["event_latency_ms"]["mean"],
                                new["event_latency_ms"]["mean"]),
        "latency_p95_pct": pct(base["event_latency_ms"]["p95"],
                               new["event_latency_ms"]["p95"]),
        "wall_pct": pct(base["wall_seconds"], new["wall_seconds"]),
        "pos_err_delta": round(new["position"]["mean_abs_error_beats"]
                               - base["position"]["mean_abs_error_beats"], 4),
        "relock_delta": (
            None if base["position"]["relock_events_after_jump"] is None
            or new["position"]["relock_events_after_jump"] is None
            else new["position"]["relock_events_after_jump"]
            - base["position"]["relock_events_after_jump"]
        ),
        "mistrack_delta": (new["position"]["mistracked_onsets"]
                           - base["position"]["mistracked_onsets"]),
        "behavior_equal": base["behavior"] == new["behavior"],
        "behavior_close": _behavior_close(base, new),
        "candidates_delta": {
            k: new["candidate_stats"][k] - base["candidate_stats"][k]
            for k in ("cand_forward", "cand_repeat", "merged_in")
        },
    }


def _cmd_run(args) -> None:
    variants = sorted(VARIANTS) if args.variant == "all" else [args.variant]
    scenarios = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        for variant in variants:
            report = run_scenario(scenario, variant, args.outdir)
            path = write_report(report, args.outdir)
            pos = report["position"]
            lat = report["event_latency_ms"]
            print(f"[{scenario} / {variant}] "
                  f"pos_err={pos['mean_abs_error_beats']} beats, "
                  f"latency mean={lat['mean']}ms p95={lat['p95']}ms, "
                  f"candidates={report['candidate_stats']['merged_in']}, "
                  f"wall={report['wall_seconds']}s -> {path}")


def _cmd_compare(args) -> None:
    scenarios = sorted(SCENARIOS)
    new_variant = args.against
    rows = []
    for scenario in scenarios:
        base_path = os.path.join(args.outdir, f"{scenario}__{args.base}.json")
        new_path = os.path.join(args.outdir, f"{scenario}__{new_variant}.json")
        with open(base_path, encoding="utf-8") as fh:
            base = json.load(fh)
        with open(new_path, encoding="utf-8") as fh:
            new = json.load(fh)
        rows.append(compare_reports(base, new))
    out = {"base": args.base, "against": new_variant, "rows": rows}
    path = os.path.join(args.outdir, f"comparison__{args.base}__{new_variant}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {path}")


# Selection rules applied to the comparison rows. A variant is eligible only
# if, on every scenario, behaviour outside the two sanctioned metrics stays
# intact and the sanctioned metrics do not regress; among eligible variants
# the fastest mean event latency wins. No eligible variant means the speedup
# is withdrawn and the baseline update order stays.
def select_variant(comparisons: dict[str, list[dict]]) -> dict:
    eligible = []
    for variant, rows in comparisons.items():
        ok = all(
            row["behavior_close"]
            and row["pos_err_delta"] <= 0.01
            and row["mistrack_delta"] <= 0
            and (row["relock_delta"] is None or row["relock_delta"] <= 0)
            for row in rows
        )
        if ok:
            mean_latency_pct = sum(
                row["latency_mean_pct"] for row in rows
                if row["latency_mean_pct"] is not None
            ) / max(1, sum(1 for row in rows if row["latency_mean_pct"] is not None))
            eligible.append((mean_latency_pct, variant))
    if not eligible:
        return {
            "selected": "baseline",
            "rationale": "no variant kept behaviour intact on every scenario; "
                         "the candidate-update speedup is withdrawn",
            "candidates": sorted(comparisons),
        }
    eligible.sort()
    best_pct, best = eligible[0]
    return {
        "selected": best,
        "rationale": f"kept behaviour intact on all scenarios with "
                     f"{best_pct:.1f}% mean event-latency change vs baseline",
        "candidates": sorted(comparisons),
    }


def _cmd_select(args) -> None:
    comparisons: dict[str, list[dict]] = {}
    for variant in args.variants:
        path = os.path.join(
            args.outdir, f"comparison__{args.base}__{variant}.json"
        )
        with open(path, encoding="utf-8") as fh:
            comparisons[variant] = json.load(fh)["rows"]
    verdict = select_variant(comparisons)
    verdict["base"] = args.base
    path = os.path.join(args.outdir, "selection.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="score-follow-benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run scenarios and write reports")
    r.add_argument("--variant", default="baseline",
                   help="variant name or 'all'")
    r.add_argument("--scenario", default="all",
                   help="scenario name or 'all'")
    r.add_argument("--outdir", default=BENCH_DIR)
    r.set_defaults(func=_cmd_run)
    c = sub.add_parser("compare", help="compare a variant against a reference")
    c.add_argument("--base", default="baseline")
    c.add_argument("--against", required=True)
    c.add_argument("--outdir", default=BENCH_DIR)
    c.set_defaults(func=_cmd_compare)
    s = sub.add_parser("select", help="pick the winning variant from comparisons")
    s.add_argument("--base", default="baseline")
    s.add_argument("--variants", nargs="+", default=["budget", "entropy"])
    s.add_argument("--outdir", default=BENCH_DIR)
    s.set_defaults(func=_cmd_select)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

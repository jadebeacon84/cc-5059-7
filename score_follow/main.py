"""Command line entry point for the score-following accompaniment prototype.

Subcommands
  gen      generate a synthetic performance trajectory (JSON)
  replay   run the *online* pipeline over a saved trajectory, render WAV + report
  live     real-time local demo with terminal keyboard (or MIDI port)
  demo     end-to-end: generate scenarios, replay each, write all artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .analysis import analyse, write_report
from .driver import ReplaySource, KeyboardSource, MidiSource, run_live, run_replay
from .score import Score, demo_score, load_score
from .synthesis import write_stereo_wav
from .trajectory import GenConfig, generate, save_trajectory


def _load_or_demo(path: str | None) -> Score:
    return load_score(path) if path else demo_score()


def cmd_gen(args) -> None:
    score = _load_or_demo(args.score)
    cfg = GenConfig(
        seed=args.seed,
        tempo_drift=args.tempo_drift,
        timing_jitter=args.jitter,
        wrong_note_rate=args.wrong_notes,
        missed_note_rate=args.missed_notes,
        repeat_press_rate=args.repeat_press,
        pause_beats=tuple(args.pause_beats),
        pause_seconds=args.pause_seconds,
        skip_indices=tuple(args.skip_indices),
        repeat_region=tuple(args.repeat_region) if args.repeat_region else None,
        repeat_skip_tail=args.repeat_skip_tail,
    )
    data = generate(score, cfg)
    save_trajectory(args.out, data)
    print(f"wrote {args.out}: {len(data['events'])} events, "
          f"{sum(1 for g in data['ground_truth'] if g['played'])} played onsets")


def cmd_replay(args) -> None:
    score = _load_or_demo(args.score)
    source = ReplaySource(args.trajectory)
    result = run_replay(score, source)
    trajectory = json.load(open(args.trajectory, encoding="utf-8"))
    report = analyse(result, trajectory, score)
    write_report(args.report, report)
    write_stereo_wav(args.wav, result.performer_bus.render(),
                     result.accompaniment_bus.render())
    print(json.dumps(report.to_dict(), indent=2)[:1200])
    print(f"\nwrote {args.wav} and {args.report}")


def cmd_live(args) -> None:
    score = _load_or_demo(args.score)
    if args.midi_port:
        source = MidiSource(args.midi_port)
        try:
            run_live(score, source)
        finally:
            source.close()
    else:
        if not sys.stdin.isatty():
            sys.exit("live keyboard mode needs a TTY; use 'replay' for offline runs")
        with KeyboardSource() as source:
            run_live(score, source)


def cmd_demo(args) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    score = demo_score()
    score_path = os.path.join(args.outdir, "score.json")
    with open(score_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "name": score.name,
                "initial_tempo_bpm": score.initial_tempo_bpm,
                "beats_per_bar": score.beats_per_bar,
                "solo": [vars(n) for n in score.solo],
                "accompaniment": [vars(n) for n in score.accompaniment],
            },
            fh, indent=1,
        )

    # tail-beat note of phrase A is solo index 8 (beat 8.0)
    scenarios = {
        "clean": GenConfig(seed=1),
        "repeat_skip_tail": GenConfig(
            seed=2, skip_indices=(8,), repeat_press_rate=0.25,
            pause_beats=(8.0,), pause_seconds=1.2,
        ),
        "twice_repeated_phrase": GenConfig(
            seed=6, repeat_region=(0.0, 9.0), repeat_skip_tail=True,
            repeat_press_rate=0.2,
        ),
        "errors_and_drift": GenConfig(
            seed=3, tempo_drift=0.25, wrong_note_rate=0.12,
            missed_note_rate=0.1, repeat_press_rate=0.15,
            pause_beats=(4.0,), pause_seconds=1.0,
        ),
    }
    summary = {}
    for name, cfg in scenarios.items():
        traj_path = os.path.join(args.outdir, f"{name}_trajectory.json")
        wav_path = os.path.join(args.outdir, f"{name}.wav")
        report_path = os.path.join(args.outdir, f"{name}_report.json")
        data = generate(score, cfg)
        save_trajectory(traj_path, data)
        result = run_replay(score, ReplaySource(traj_path))
        report = analyse(result, data, score)
        write_report(report_path, report)
        write_stereo_wav(wav_path, result.performer_bus.render(),
                         result.accompaniment_bus.render())
        summary[name] = report.to_dict()
        print(f"[{name}] pos_err={report.mean_abs_position_error:.3f} beats, "
              f"lag={report.mean_abs_lag:.3f}s, stalls={report.n_stalls}, "
              f"acc_notes={report.n_played_acc}, wav={wav_path}")
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nall artifacts in {args.outdir}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="score-follow")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a performance trajectory")
    g.add_argument("--score")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--tempo-drift", type=float, default=0.0)
    g.add_argument("--jitter", type=float, default=0.03)
    g.add_argument("--wrong-notes", type=float, default=0.0)
    g.add_argument("--missed-notes", type=float, default=0.0)
    g.add_argument("--repeat-press", type=float, default=0.0)
    g.add_argument("--pause-beats", type=float, nargs="*", default=[])
    g.add_argument("--pause-seconds", type=float, default=1.5)
    g.add_argument("--skip-indices", type=int, nargs="*", default=[])
    g.add_argument("--repeat-region", type=float, nargs=2, metavar=("FIRST", "LAST"),
                   help="play this score region a second time")
    g.add_argument("--repeat-skip-tail", action="store_true",
                   help="drop the region's tail note on the second pass")
    g.set_defaults(func=cmd_gen)

    r = sub.add_parser("replay", help="online pipeline over a saved trajectory")
    r.add_argument("--score")
    r.add_argument("--trajectory", required=True)
    r.add_argument("--wav", required=True)
    r.add_argument("--report", required=True)
    r.set_defaults(func=cmd_replay)

    l = sub.add_parser("live", help="real-time local performance")
    l.add_argument("--score")
    l.add_argument("--midi-port", default=None)
    l.set_defaults(func=cmd_live)

    d = sub.add_parser("demo", help="end-to-end offline demonstration")
    d.add_argument("--outdir", default="data/demo_run")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

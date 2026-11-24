"""Regression tests for the selected candidate-update speedup.

The budgeted update (default) is compared against the pre-speedup baseline
update order on identical event streams: the tracked response and position
must agree on every scored onset, late input and missed notes must keep
their original uncertainty (no fixed-speed marching), and the reserved
far-jump budget must keep legitimate jump explanations alive.
"""
import json
import os
import unittest

from score_follow.benchmark import (
    SCENARIOS,
    _ORDERS,
    bench_score,
    build_trajectory,
)
from score_follow.estimator import Estimator
from score_follow.types import NoteEvent

BASELINE_KWARGS = {
    "local_budget": None,
    "far_jump_budget": None,
    "far_jump_proposers": None,
}

ARTIFACT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "benchmark")
)


def _events(traj):
    return [NoteEvent(float(e["time"]), e["kind"], int(e["pitch"]))
            for e in traj["events"]]


def _scored_onsets(traj):
    return [g for g in traj["ground_truth"] if g.get("played")]


class _GtCursor:
    """Walks played ground-truth onsets; only advances on a time match,
    so insertions (repeated presses) never consume a real onset."""

    def __init__(self, traj):
        self.rows = [(float(g["time"]), g) for g in _scored_onsets(traj)]
        self.i = 0

    def match(self, t: float):
        while self.i < len(self.rows) and self.rows[self.i][0] < t - 0.35:
            self.i += 1
        if self.i < len(self.rows) and abs(self.rows[self.i][0] - t) <= 0.35:
            row = self.rows[self.i][1]
            self.i += 1
            return row
        return None


class SameTrajectoryTests(unittest.TestCase):
    """Same performance, two update orders: response and position compared."""

    def _run_pair(self, scenario):
        score = bench_score()
        traj = build_trajectory(score, _ORDERS[scenario], SCENARIOS[scenario])
        base = Estimator(score, **BASELINE_KWARGS)
        fast = Estimator(score)  # selected budgeted update
        rows = []
        for ev in _events(traj):
            a = base.handle_event(ev)
            b = fast.handle_event(ev)
            rows.append((ev, a, b))
        return traj, rows

    def test_jump_scenario_positions_agree_on_scored_onsets(self):
        traj, rows = self._run_pair("jump_after_repeat")
        gt = _GtCursor(traj)
        checked = 0
        for ev, a, b in rows:
            if ev.kind != "on":
                continue
            if gt.match(ev.time) is None:
                continue  # insertion: not a scored onset
            self.assertAlmostEqual(a.best_beat, b.best_beat, delta=0.5,
                                   msg=f"position diverged at t={ev.time}")
            self.assertAlmostEqual(a.confidence, b.confidence, delta=0.02)
            checked += 1
        self.assertGreater(checked, 20)

    def test_load_scenario_positions_agree_on_scored_onsets(self):
        traj, rows = self._run_pair("load")
        gt = _GtCursor(traj)
        for ev, a, b in rows:
            if ev.kind != "on":
                continue
            if gt.match(ev.time) is None:
                continue
            self.assertAlmostEqual(a.best_beat, b.best_beat, delta=0.5)
        self.assertEqual(rows[-1][2].expected_index, rows[-1][1].expected_index)

    def test_late_input_and_misses_keep_uncertainty(self):
        # late input (2 s pause) plus missed notes: neither update order may
        # fake real-time following by marching on at a fixed speed
        score = bench_score()
        traj = build_trajectory(
            score, _ORDERS["late_and_missed"], SCENARIOS["late_and_missed"]
        )
        events = _events(traj)
        finals = []
        for kwargs in (BASELINE_KWARGS, {}):
            est = Estimator(score, **kwargs)
            prev_on_time = None
            before_best = None
            before_conf = None
            pause_checked = False
            for ev in events:
                if (ev.kind == "on" and prev_on_time is not None
                        and ev.time - prev_on_time > 1.5 and not pause_checked):
                    # deep inside the input gap, just before the next onset
                    during = est.tick(ev.time - 0.1)
                    # position must not advance without evidence...
                    self.assertEqual(during.best_beat, before_best)
                    # ...and uncertainty must grow, not stay pinned
                    self.assertLess(during.confidence, before_conf)
                    pause_checked = True
                estimate = est.handle_event(ev)
                if ev.kind == "on":
                    prev_on_time = ev.time
                    before_best = estimate.best_beat
                    before_conf = estimate.confidence
            self.assertTrue(pause_checked)
            finals.append(est.last_estimate.expected_index)
        # both update orders end at the same final reading
        self.assertEqual(finals[0], finals[1])

    def test_far_jump_budget_keeps_repeat_reading_alive(self):
        # repeated phrase with skipped tail: the backwards-jump reading must
        # survive next to the forward one under the reserved far-jump budget
        score = bench_score()
        traj = build_trajectory(
            score, _ORDERS["jump_after_repeat"], SCENARIOS["jump_after_repeat"]
        )
        est = Estimator(score)
        saw_repeat_reading = False
        for ev in _events(traj):
            est_now = est.handle_event(ev)
            beats = {b for b, _ in est_now.positions}
            if any(b <= 7.0 for b in beats) and any(b >= 10.0 for b in beats):
                saw_repeat_reading = True
        self.assertTrue(saw_repeat_reading)


class CommittedArtifactTests(unittest.TestCase):
    """The committed performance sample must match what the code produces;
    a hand-edited or stale report fails here (no fake reports)."""

    def test_baseline_artifact_matches_execution(self):
        path = os.path.join(ARTIFACT_DIR, "load__baseline.json")
        if not os.path.exists(path):
            self.skipTest("committed benchmark artifact not present")
        with open(path, encoding="utf-8") as fh:
            artifact = json.load(fh)
        score = bench_score()
        traj = build_trajectory(score, _ORDERS["load"], SCENARIOS["load"])
        # the artifact was produced by a full replay (ticks interleaved), so
        # the check replays the same trajectory through the same driver
        import tempfile

        from score_follow.driver import ReplaySource, run_replay
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = os.path.join(tmp, "traj.json")
            with open(traj_path, "w", encoding="utf-8") as fh:
                json.dump(traj, fh)
            est = Estimator(score, **BASELINE_KWARGS)
            result = run_replay(score, ReplaySource(traj_path), estimator=est)
        gt = _GtCursor(traj)
        idx_series = []
        for estimate in result.estimates:
            if not estimate.is_onset:
                continue
            if gt.match(estimate.time) is None:
                continue
            idx_series.append(estimate.expected_index)
        artifact_idx = [
            row["expected_index"] for row in artifact["evidence"]["per_onset"]
            if row["gt_beat"] is not None
        ]
        self.assertEqual(idx_series, artifact_idx)
        self.assertEqual(est.stats["cand_forward"],
                         artifact["candidate_stats"]["cand_forward"])
        self.assertEqual(est.stats["cand_repeat"],
                         artifact["candidate_stats"]["cand_repeat"])

    def test_selection_artifact_chose_a_real_variant(self):
        path = os.path.join(ARTIFACT_DIR, "selection.json")
        if not os.path.exists(path):
            self.skipTest("selection artifact not present")
        with open(path, encoding="utf-8") as fh:
            verdict = json.load(fh)
        self.assertEqual(verdict["selected"], "budget")
        # the selected variant is the running default
        est = Estimator(bench_score())
        self.assertEqual(est.local_budget, 128)
        self.assertEqual(est.far_jump_budget, 24)


if __name__ == "__main__":
    unittest.main()

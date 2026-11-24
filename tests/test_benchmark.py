"""Smoke tests for the committed performance sample."""
import json
import os
import tempfile
import unittest

from score_follow.benchmark import (
    SCENARIOS,
    VARIANTS,
    _ORDERS,
    bench_score,
    build_trajectory,
    compare_reports,
    run_scenario,
)


class BenchmarkTests(unittest.TestCase):
    def test_trajectory_is_deterministic(self):
        score = bench_score()
        cfg = SCENARIOS["jump_after_repeat"]
        a = build_trajectory(score, _ORDERS["jump_after_repeat"], cfg)
        b = build_trajectory(score, _ORDERS["jump_after_repeat"], cfg)
        self.assertEqual(a, b)
        # the sudden jump to phrase D is actually in the performance order
        beats = [g["beat"] for g in a["ground_truth"]]
        self.assertIn(26.0, beats)
        self.assertNotIn(18.0, beats)  # phrase C is jumped over

    def test_baseline_run_produces_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_scenario("jump_after_repeat", "baseline", tmp)
        # metrics trace back to per-onset execution evidence
        self.assertGreater(len(report["evidence"]["per_onset"]), 0)
        self.assertTrue(all(r["latency_ms"] >= 0.0
                            for r in report["evidence"]["per_onset"]))
        self.assertGreater(report["event_latency_ms"]["mean"], 0.0)
        self.assertGreater(report["candidate_stats"]["cand_forward"], 0)
        self.assertGreater(report["candidate_stats"]["cand_repeat"], 0)
        # the tracker follows the jump within a handful of onsets
        self.assertIsNotNone(report["position"]["relock_events_after_jump"])
        # accompaniment actually sounded: no fake report
        self.assertGreater(report["evidence"]["audio"]["accompaniment_rms"], 0.0)
        self.assertGreater(report["accompaniment"]["played"], 0)

    def test_compare_reports_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = run_scenario("late_and_missed", "baseline", tmp)
            new = run_scenario("late_and_missed", "baseline", tmp)
        row = compare_reports(base, new)
        self.assertEqual(row["scenario"], "late_and_missed")
        self.assertTrue(row["behavior_equal"])
        self.assertEqual(row["pos_err_delta"], 0.0)
        self.assertEqual(row["mistrack_delta"], 0)

    def test_all_scenarios_and_variants_are_wired(self):
        for name in SCENARIOS:
            self.assertIn(name, _ORDERS)
        self.assertIn("baseline", VARIANTS)


if __name__ == "__main__":
    unittest.main()

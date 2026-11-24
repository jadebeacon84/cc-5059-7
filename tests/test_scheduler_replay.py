"""Scheduler and end-to-end replay tests."""
import json
import os
import tempfile
import unittest

from score_follow.analysis import analyse
from score_follow.driver import ReplaySource, VirtualClock, run_replay
from score_follow.estimator import Estimator
from score_follow.scheduler import Scheduler
from score_follow.score import demo_score
from score_follow.trajectory import GenConfig, generate, save_trajectory
from score_follow.types import NoteEvent


class VirtualClockTests(unittest.TestCase):
    def test_clock_is_monotonic(self):
        clk = VirtualClock()
        clk.advance(1.0)
        clk.advance(2.5)
        self.assertEqual(clk.now(), 2.5)
        with self.assertRaises(ValueError):
            clk.advance(2.0)


class SystemClockTests(unittest.TestCase):
    def test_wall_clock_advances(self):
        from score_follow.scheduler import SystemClock
        clk = SystemClock()
        self.assertGreaterEqual(clk.now(), 0.0)
        self.assertLess(clk.now(), 5.0)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.score = demo_score()
        self.clock = VirtualClock()
        self.sched = Scheduler(self.score, self.clock)
        self.est = Estimator(self.score)

    def _feed(self, ev):
        estimate = self.est.handle_event(ev)
        return self.sched.update(estimate)

    def test_played_notes_are_never_rewritten(self):
        # arm and play a note, then push a wildly different estimate
        tempo = 90 / 60.0
        events = []
        for i, note in enumerate(self.score.solo[:6]):
            t = note.start / tempo
            events.append(NoteEvent(t, "on", note.pitch))
            events.append(NoteEvent(t + 0.3, "off", note.pitch))
        played_snapshots = []
        for ev in events:
            self.clock.advance(ev.time)
            actions = self._feed(ev)
            for action, entry in actions:
                if action == "play":
                    played_snapshots.append((entry.note.pitch, entry.play_time))
        # everything marked played stays played; no entry returns to pending
        for entry in self.sched.plan:
            if entry.state == "played":
                self.assertIn((entry.note.pitch, entry.play_time), played_snapshots)
        self.assertTrue(len(played_snapshots) > 0)

    def test_future_plan_only_moves_within_window(self):
        # far-future notes stay on the initial tempo mapping until close
        far = [e for e in self.sched.plan if e.play_time - self.clock.now() > 5.0]
        self.assertTrue(any(far))
        initial_times = {id(e): e.play_time for e in far}
        # a single new event must not yank notes beyond the adjust window
        ev = NoteEvent(0.01, "on", self.score.solo[0].pitch)
        self.clock.advance(0.01)
        self._feed(ev)
        for e in far:
            self.assertEqual(initial_times[id(e)], e.play_time)

    def test_low_confidence_softens_or_waits(self):
        # build an estimate with zero confidence and let a note come due
        ev = NoteEvent(0.0, "on", self.score.solo[0].pitch)
        self.clock.advance(0.0)
        self._feed(ev)
        # force the first accompaniment note to be due while confidence is low
        target = self.sched.plan[0]
        target.state = "armed"
        target.play_time = 0.05
        # estimator with idle time: advance clock far beyond stall threshold
        self.clock.advance(4.0)
        est = self.est.tick(self.clock.now())
        self.assertLess(est.confidence, 0.4)
        actions = self.sched.poll(est)
        kinds = {a for a, _ in actions}
        # policy says wait (stall), never a confident play
        self.assertIn("stall", kinds)
        self.assertNotIn("play", kinds)


class ReplayEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.score = demo_score()
        self.tmp = tempfile.mkdtemp()

    def _run(self, cfg: GenConfig):
        data = generate(self.score, cfg)
        path = os.path.join(self.tmp, "traj.json")
        save_trajectory(path, data)
        result = run_replay(self.score, ReplaySource(path))
        report = analyse(result, data, self.score)
        return result, report, data

    def test_clean_run_position_error_is_small(self):
        _, report, _ = self._run(GenConfig(seed=11, timing_jitter=0.01))
        self.assertLess(report.mean_abs_position_error, 0.15)
        self.assertGreater(report.correct_within_half_beat, 0.9)
        self.assertGreater(report.n_played_acc, 0)
        self.assertGreater(report.accompaniment_rms, 0.0)

    def test_repeat_phrase_skipped_tail_does_not_derail(self):
        cfg = GenConfig(seed=5, skip_indices=(8,), repeat_press_rate=0.3,
                        pause_beats=(8.0,), pause_seconds=1.2)
        result, report, data = self._run(cfg)
        # final position reaches the end of the piece
        self.assertEqual(result.estimates[-1].expected_index, len(self.score.solo))
        self.assertLess(report.mean_abs_position_error, 0.75)
        # ambiguity was actually observed along the way (two readings kept)
        self.assertGreater(report.ambiguous_windows, 0)

    def test_errors_drift_and_pauses(self):
        cfg = GenConfig(seed=9, tempo_drift=0.25, wrong_note_rate=0.12,
                        missed_note_rate=0.1, repeat_press_rate=0.15,
                        pause_beats=(4.0,), pause_seconds=1.0)
        _, report, _ = self._run(cfg)
        self.assertLess(report.mean_abs_position_error, 1.0)
        self.assertEqual(report.n_onsets,
                         sum(1 for g in generate(self.score, cfg)["ground_truth"]
                             if g["played"]))

    def test_wav_is_rendered_with_two_buses(self):
        from score_follow.synthesis import write_stereo_wav
        import wave
        result, _, _ = self._run(GenConfig(seed=4))
        wav_path = os.path.join(self.tmp, "out.wav")
        write_stereo_wav(wav_path, result.performer_bus.render(),
                         result.accompaniment_bus.render())
        with wave.open(wav_path) as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertGreater(wf.getnframes(), 1000)

    def test_replay_uses_only_arrived_events(self):
        # estimates produced up to the truncation timestamp must be identical
        # whether or not later events exist in the file — no lookahead leaks.
        cfg = GenConfig(seed=13)
        data = generate(self.score, cfg)
        cutoff = data["events"][len(data["events"]) // 2]["time"]
        truncated = dict(data, events=[e for e in data["events"] if e["time"] <= cutoff])
        full_path = os.path.join(self.tmp, "full.json")
        trunc_path = os.path.join(self.tmp, "trunc.json")
        save_trajectory(full_path, data)
        save_trajectory(trunc_path, truncated)
        full = run_replay(self.score, ReplaySource(full_path))
        trunc = run_replay(self.score, ReplaySource(trunc_path))
        # compare only event-driven onset/off estimates: these depend on
        # nothing but events at or before their own timestamp
        full_events = [e for e in full.estimates
                       if getattr(e, "is_onset", False) and e.time <= cutoff + 1e-9]
        trunc_events = [e for e in trunc.estimates if getattr(e, "is_onset", False)]
        self.assertGreater(len(trunc_events), 0)
        self.assertEqual(len(trunc_events), len(full_events))
        for a, b in zip(trunc_events, full_events):
            self.assertEqual(a.best_beat, b.best_beat)
            self.assertEqual(a.event_count, b.event_count)
            self.assertEqual(a.tempo, b.tempo)
            self.assertEqual(a.is_onset, b.is_onset)


    def test_wait_policy_fires_in_full_pipeline(self):
        # dense accompaniment guarantees an entry is due inside the pause;
        # the pipeline must stall it rather than play into silence
        from score_follow.score import Score
        from score_follow.types import ScoreNote
        from score_follow.trajectory import GenConfig as GC
        dense_solo = [ScoreNote(float(i), 1.0, 60 + (i % 5)) for i in range(12)]
        dense_acc = [ScoreNote(float(i), 0.8, 48, 70) for i in range(12)]
        dense = Score("dense", dense_solo, dense_acc, initial_tempo_bpm=120)
        data = generate(dense, GC(seed=2, pause_beats=(3.0,), pause_seconds=4.0))
        path = os.path.join(self.tmp, "dense.json")
        save_trajectory(path, data)
        result = run_replay(dense, ReplaySource(path))
        stalls = [r for r in result.scheduler.log if r.stalled]
        self.assertGreater(len(stalls), 0)
        # while fully stalled, no NEW accompaniment note may start
        stall_times = [r.logged_at for r in stalls if r.target_beat >= 9.0]
        first_stall = min(stall_times)
        times = {g["index"]: g["time"] for g in data["ground_truth"] if g.get("played")}
        resume_time = times[4]
        # no NEW accompaniment onset between the full-stall point and resume
        full_stall_onsets = [
            r for r in result.scheduler.played_notes
            if first_stall <= (r.played_time or 0.0) < resume_time
        ]
        self.assertEqual(len(full_stall_onsets), 0)
        # notes that do sound as confidence decays only get softer, never louder
        softening = [
            r.velocity_scale for r in result.scheduler.played_notes
            if times[3] < r.played_time < first_stall
        ]
        self.assertTrue(all(a >= b for a, b in zip(softening, softening[1:])))
        # after resume, accompaniment recovers and notes sound again
        post_resume_plays = [
            r for r in result.scheduler.played_notes if r.played_time >= resume_time
        ]
        self.assertGreater(len(post_resume_plays), 0)


if __name__ == "__main__":
    unittest.main()

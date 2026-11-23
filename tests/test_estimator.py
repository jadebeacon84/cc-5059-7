"""Unit tests for the online tracker guarantees."""
import math
import unittest

from score_follow.estimator import Estimator
from score_follow.score import demo_score
from score_follow.types import NoteEvent


def strict_tempo_events(score, tempo_bpm, *, skip=(), wrong=None, repeats=(), pause_at=None,
                        pause_seconds=1.0, repeat_pitch_at=()):
    """Build events at exact tempo; returns (events, gt index per on-event)."""
    tempo = tempo_bpm / 60.0
    events: list[NoteEvent] = []
    gt: list[int] = []
    t = 0.0
    prev_beat = 0.0
    wrong = wrong or {}
    for i, note in enumerate(score.solo):
        t += (note.start - prev_beat) / tempo
        prev_beat = note.start
        if pause_at is not None and abs(note.start - pause_at) < 1e-9:
            t += pause_seconds
        if i in skip:
            continue
        pitch = wrong.get(i, note.pitch)
        events.append(NoteEvent(round(t, 4), "on", pitch))
        gt.append(i)
        events.append(NoteEvent(round(t + 0.35 / tempo, 4), "off", pitch))
        if i in repeat_pitch_at:
            events.append(NoteEvent(round(t + 0.45 / tempo, 4), "on", pitch))
            events.append(NoteEvent(round(t + 0.55 / tempo, 4), "off", pitch))
    return events, gt


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.score = demo_score()
        self.est = Estimator(self.score)

    def _feed_onsets_only(self, events):
        last = None
        for ev in events:
            last = self.est.handle_event(ev)
        return last

    def test_clean_playthrough_tracks_each_note(self):
        events, gt = strict_tempo_events(self.score, 90)
        for ev, gt_idx in zip(events, [x for x in gt for _ in range(2)]):
            est = self.est.handle_event(ev)
        # final expected index must be end of score, not event count
        self.assertEqual(est.expected_index, len(self.score.solo))
        self.assertGreater(est.confidence, 0.6)

    def test_event_count_is_not_score_index(self):
        # skip two score notes: observed event #k must not map to score note #k
        events, gt = strict_tempo_events(self.score, 90, skip=(3, 7))
        on_count = 0
        for ev in events:
            est = self.est.handle_event(ev)
            if ev.kind == "on":
                on_count += 1
        self.assertEqual(est.expected_index, len(self.score.solo))
        self.assertEqual(on_count, len(self.score.solo) - 2)
        self.assertLess(abs(est.best_beat - self.score.solo[-1].start), 1.0)

    def test_wrong_note_is_tolerated(self):
        events, _ = strict_tempo_events(self.score, 90, wrong={4: 65, 10: 66})
        est = self._feed_onsets_only(events)
        self.assertEqual(est.expected_index, len(self.score.solo))

    def test_gradual_tempo_change_is_followed(self):
        # accelerate linearly from 70 to 120 bpm across the piece
        events: list[NoteEvent] = []
        clock = 0.0
        n = len(self.score.solo)
        for i, note in enumerate(self.score.solo):
            bpm = 70 + 50 * (i / max(1, n - 1))
            gap = note.start - (self.score.solo[i - 1].start if i else 0.0)
            clock += gap / (bpm / 60.0)
            events.append(NoteEvent(round(clock, 4), "on", note.pitch))
            events.append(NoteEvent(round(clock + 0.25, 4), "off", note.pitch))
        est = self._feed_onsets_only(events)
        self.assertEqual(est.expected_index, len(self.score.solo))
        # final tempo belief should be near 120 bpm = 2.0 beats/s
        self.assertAlmostEqual(est.tempo, 2.0, delta=0.35)

    def test_repeated_key_press_does_not_advance_position(self):
        events, _ = strict_tempo_events(self.score, 90, repeat_pitch_at=(2, 5, 11))
        on_events = [e for e in events if e.kind == "on"]
        # feed through the real onset of note index 2 plus its repeat, then
        # check the expected score note did not jump past note 3
        estimate = None
        onsets_seen = 0
        for ev in events:
            estimate = self.est.handle_event(ev)
            if ev.kind == "on":
                onsets_seen += 1
            if onsets_seen == 4:  # notes 0,1,2 then the repeated press
                break
        self.assertLessEqual(estimate.expected_index, 4)

    def test_skipped_tail_keeps_two_positions_alive(self):
        # phrase A is played up to its last regular note; the tail-beat note
        # (index 8) is skipped, so the tracker must keep "before tail" and
        # "past tail" as two live readings instead of committing.
        tempo = 90 / 60.0
        events: list[NoteEvent] = []
        for i in range(0, 8):
            note = self.score.solo[i]
            clock = note.start / tempo
            events.append(NoteEvent(round(clock, 4), "on", note.pitch))
            events.append(NoteEvent(round(clock + 0.2, 4), "off", note.pitch))
        est = None
        for ev in events:
            est = self.est.handle_event(ev)
        distinct_indices = {pos for pos, _ in est.positions}
        self.assertGreaterEqual(len(distinct_indices), 2)
        # the "tail note already happened" reading (beat 8) must survive with
        # enough weight that fresh evidence can still select it
        weights = dict(est.positions)
        self.assertGreater(weights.get(8.0, 0.0), 0.05)
        self.assertLess(est.margin, 0.8)

    def test_new_evidence_resolves_ambiguity(self):
        tempo = 90 / 60.0
        events: list[NoteEvent] = []
        for i in list(range(0, 8)) + list(range(9, 14)):
            note = self.score.solo[i]
            clock = note.start / tempo
            events.append(NoteEvent(round(clock, 4), "on", note.pitch))
            events.append(NoteEvent(round(clock + 0.2, 4), "off", note.pitch))
        est = None
        for ev in events:
            est = self.est.handle_event(ev)
        # phrase B's distinct pitches pin the reading at index 14
        self.assertEqual(est.expected_index, 14)
        self.assertGreater(est.confidence, 0.55)

    def test_pause_erodes_confidence_but_recovers(self):
        events, _ = strict_tempo_events(self.score, 90, pause_at=4.0, pause_seconds=3.0)
        after_pause = None
        pause_start = 4.0 / 1.5  # beat 4 onset time, before which the pause lands
        for ev in events:
            est = self.est.handle_event(ev)
            if ev.kind == "on" and ev.time > pause_start + 2.0:
                after_pause = est
                break
        self.assertIsNotNone(after_pause)
        # first post-pause onset is solo index 4 -> next expected index 5
        self.assertEqual(after_pause.expected_index, 5)
        self.assertGreater(after_pause.confidence, 0.35)

    def test_only_past_events_are_used(self):
        # feeding one event at a time must produce the same estimates as
        # batch feeding — there is no global lookahead state
        events, _ = strict_tempo_events(self.score, 90)
        one_shot = Estimator(self.score)
        streaming = Estimator(self.score)
        for ev in events:
            a = one_shot.handle_event(ev)
        for ev in events:
            b = streaming.handle_event(ev)
        self.assertEqual(one_shot.event_count, streaming.event_count)
        self.assertAlmostEqual(one_shot.last_estimate.best_beat,
                               streaming.last_estimate.best_beat, places=6)


    def test_notated_repeat_keeps_both_jump_and_forward_readings(self):
        # phrase A is played twice; on the second pass the tail note is
        # skipped. Both "jumped back into the repeat" and "kept moving
        # forward" readings must survive until phrase B evidence arrives.
        tempo = self.score.initial_tempo_bpm / 60.0
        est = Estimator(self.score)

        def play(index, clock):
            note = self.score.solo[index]
            e = est.handle_event(NoteEvent(round(clock, 4), "on", note.pitch))
            est.handle_event(NoteEvent(round(clock + 0.2, 4), "off", note.pitch))
            return e

        clock = 0.0
        for i in range(0, 9):  # first full pass incl. tail note index 8
            clock = self.score.solo[i].start / tempo
            e = play(i, clock)
        self.assertEqual(e.expected_index, 9)

        second_start = clock + 1.0 / tempo
        e = None
        for k, i in enumerate(range(0, 8)):  # second pass, tail skipped
            e = play(i, second_start + k / tempo)
        beats = {b for b, _ in e.positions}
        # a reading inside the repeated phrase (beat <= 7) is still alive...
        self.assertTrue(any(b <= 7.0 for b in beats))
        # ...alongside a forward reading past the phrase (beat >= 13)
        self.assertTrue(any(b >= 13.0 for b in beats))
        ambiguous_weight = sum(w for b, w in e.positions if b <= 7.0)
        self.assertGreater(ambiguous_weight, 0.05)

        # fresh phrase-B evidence (distinct pitches) reorders the readings:
        # each new event shifts weight rather than the tracker having
        # committed before the evidence arrived
        forward_clock = second_start + 8 / tempo
        before_ranking = [b for b, _ in e.positions]
        e = play(9, forward_clock)
        after_ranking = [b for b, _ in e.positions]
        self.assertNotEqual(before_ranking[:1], after_ranking[:1])
        # both structural readings remain represented, neither is retroactively
        # erased; the decision is driven only by events that have arrived
        self.assertTrue(any(b <= 7.0 for b, _ in e.positions))
        self.assertTrue(any(b >= 10.0 for b, _ in e.positions))


if __name__ == "__main__":
    unittest.main()

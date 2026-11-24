"""Online position estimator.

The tracker never maps "the k-th observed event" to "the k-th score note".
Instead it keeps a beam of hypotheses, each one a belief about where in the
score the performer is, with its own tempo estimate and a log weight. A new
key event advances every hypothesis through several interpretations:

* the key matches the next score note (exact or chromatically wrong),
* score notes were skipped (missed notes, including a skipped tail beat),
* the key is an extra insertion (repeated press, ornament, noise).

Ambiguous readings stay alive together, so a repeated phrase whose tail was
skipped leaves two score positions in the beam until later evidence settles
it. Output confidence reflects that ambiguity and the scheduler reacts to it.

Candidate updates are budgeted (see ``_handle_on`` for the exact order): a
local continuation tier scores forward matches strongest-reading-first up to
``local_budget`` per event, and a reserved far-jump tier lets the strongest
readings propose repeat-region openings, so a legitimate jump explanation is
never starved by local pressure. The budget split was selected against an
entropy-adaptive window variant on the committed benchmark runs
(data/benchmark/); the pre-speedup update order stays reproducible with
``local_budget=None, far_jump_budget=None, far_jump_proposers=None``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

from .score import Score
from .types import Estimate, NoteEvent


@dataclass
class Hypothesis:
    index: int                 # next solo note index expected
    anchor_beat: float         # score beat of last matched onset
    anchor_time: float         # clock time of that onset
    tempo: float               # beats / second
    log_weight: float
    held: frozenset[int]       # pitches currently held down

    @property
    def weight(self) -> float:
        return math.exp(self.log_weight)


def _pitch_cost(semitone_distance: int, scale: float) -> float:
    # Gaussian-ish penalty on |semitones|; 0 distance costs 0.
    return scale * semitone_distance * semitone_distance


class Estimator:
    def __init__(
        self,
        score: Score,
        *,
        beam_width: int = 32,
        pitch_scale: float = 0.55,
        insertion_cost: float = 2.4,
        reinsertion_cost: float = 0.9,
        skip_base_cost: float = 0.65,
        skip_per_note: float = 0.45,
        timing_scale: float = 0.7,
        idle_cost_per_sec: float = 0.25,
        tempo_smoothing: float = 0.35,
        max_tempo_ratio: float = 2.5,
        lookahead_notes: int = 10,
        repeat_jump_cost: float = 0.85,
        local_budget: int | None = 128,
        far_jump_budget: int | None = 24,
        far_jump_proposers: int | None = 8,
        adaptive_window: bool = False,
        lookahead_min: int = 5,
        lookahead_max: int = 16,
        entropy_threshold: float = 0.65,
        profile: bool = False,
    ) -> None:
        self.score = score
        self.beam_width = beam_width
        self.pitch_scale = pitch_scale
        self.insertion_cost = insertion_cost
        self.reinsertion_cost = reinsertion_cost
        self.skip_base_cost = skip_base_cost
        self.skip_per_note = skip_per_note
        self.timing_scale = timing_scale
        self.idle_cost_per_sec = idle_cost_per_sec
        self.tempo_smoothing = tempo_smoothing
        self.max_tempo_ratio = max_tempo_ratio
        self.lookahead_notes = lookahead_notes
        self.repeat_jump_cost = repeat_jump_cost
        self.local_budget = local_budget
        self.far_jump_budget = far_jump_budget
        self.far_jump_proposers = far_jump_proposers
        self.adaptive_window = adaptive_window
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.entropy_threshold = entropy_threshold
        self.profile = profile
        self.initial_tempo = score.initial_tempo_bpm / 60.0
        # repeat-region opening notes, sealed once: the per-event scan of the
        # whole solo staff would otherwise repeat for every hypothesis
        self._repeat_openings: dict[float, tuple[int, ...]] = {
            first: tuple(
                j for j, n in enumerate(score.solo)
                if n.start >= first - 1e-9 and n.start <= first + 3.0
            )
            for first in score.repeat_targets()
        }
        self.hypotheses: list[Hypothesis] = []
        self.event_count = 0
        self.last_estimate: Estimate | None = None
        # cost-breakdown counters for the candidate-update path; always-on
        # integer increments, phase timing only when ``profile`` is set
        self.stats: dict[str, int] = {
            "on_events": 0,
            "off_events": 0,
            "cand_insertion": 0,
            "cand_forward": 0,
            "cand_repeat": 0,
            "merged_in": 0,
            "merged_out": 0,
        }
        self.phase_seconds: dict[str, float] = {"expand": 0.0, "merge": 0.0}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def handle_event(self, event: NoteEvent) -> Estimate:
        """Feed exactly one observed event; no future data is consulted."""
        if event.kind == "on":
            self._handle_on(event)
            is_onset = True
        else:
            self._handle_off(event)
            is_onset = False
        self.event_count += 1
        self.last_estimate = self._make_estimate(event.time, is_onset)
        return self.last_estimate

    def tick(self, now: float) -> Estimate:
        """Clock update with no new event: gently erode stale hypotheses."""
        if self.hypotheses:
            latest = max(h.anchor_time for h in self.hypotheses)
            dt = max(0.0, now - latest)
            if dt > 0:
                for h in self.hypotheses:
                    h.log_weight -= self.idle_cost_per_sec * dt
                self._prune()
        self.last_estimate = self._make_estimate(now, False)
        return self.last_estimate

    def beat_to_time(self, beat: float, now: float) -> float:
        """Project a future score beat onto the clock using the best reading."""
        best = self._best()
        if best is None:
            return now + (beat - self.score.length_beats) / max(self.initial_tempo, 1e-6)
        delta_beats = beat - best.anchor_beat
        return best.anchor_time + delta_beats / max(best.tempo, 1e-6)

    # ------------------------------------------------------------------ #
    # Note-on handling
    # ------------------------------------------------------------------ #
    def _handle_on(self, event: NoteEvent) -> None:
        if not self.hypotheses:
            self.hypotheses = self._seed(event)
            return

        self.stats["on_events"] += 1
        t0 = perf_counter() if self.profile else 0.0
        # Computation order of one candidate update:
        # 1. Seal per-event shared tables. Every hypothesis window overlaps
        #    its neighbours', so the same target note would otherwise be
        #    pitch-scored once per hypothesis; the memo scores each distinct
        #    target pitch exactly once per event.
        # 2. Local continuation tier. Hypotheses are stored weight-sorted
        #    (every normalize/prune site keeps that invariant), so spending
        #    the local budget in storage order spends it on the strongest
        #    readings first; the budget caps total forward scoring per event.
        # 3. Far-jump tier. A reserved budget lets the strongest hypotheses
        #    propose repeat-region openings, so local pressure can never
        #    starve a legitimate jump interpretation.
        # 4. Merge duplicate beliefs, truncate to the beam, normalize.
        pitch_memo: dict[int, float] = {}
        lookahead = self._event_lookahead()
        upper = min(
            len(self.score.solo),
            max(h.index for h in self.hypotheses) + lookahead,
        )
        candidates: list[Hypothesis] = []
        local_spent = 0
        for h in self.hypotheses:
            # Interpretation 1: insertion (repeated press / ornament / noise).
            # Re-pressing a held key is common and must not kill the reading.
            held = set(h.held) | {event.pitch}
            ins_cost = self.reinsertion_cost if event.pitch in h.held else self.insertion_cost
            candidates.append(
                Hypothesis(
                    h.index, h.anchor_beat, h.anchor_time, h.tempo,
                    h.log_weight - ins_cost, frozenset(held),
                )
            )
            self.stats["cand_insertion"] += 1
            # Interpretation 2: match a score note at or ahead of index.
            for j in range(h.index, min(upper, h.index + lookahead)):
                if self.local_budget is not None and local_spent >= self.local_budget:
                    break
                candidates.append(self._match(h, j, event, pitch_memo))
                local_spent += 1
                self.stats["cand_forward"] += 1
        # Interpretation 3: a notated repeat sends the reading backwards
        # to the region start. This competes with the forward reading, so
        # the two score positions coexist until later evidence decides.
        proposers = (
            self.hypotheses
            if self.far_jump_proposers is None
            else self.hypotheses[: self.far_jump_proposers]
        )
        far_spent = 0
        for h in proposers:
            for first_beat in self.score.repeat_targets():
                new = self._repeat_matches(h, first_beat, event, pitch_memo)
                if self.far_jump_budget is not None:
                    new = new[: max(0, self.far_jump_budget - far_spent)]
                far_spent += len(new)
                self.stats["cand_repeat"] += len(new)
                candidates.extend(new)

        t1 = perf_counter() if self.profile else 0.0
        self.stats["merged_in"] += len(candidates)
        merged = self._merge(candidates)
        self.stats["merged_out"] += len(merged)
        merged.sort(key=lambda x: x.log_weight, reverse=True)
        self.hypotheses = self._normalize(merged[: self.beam_width])
        if self.profile:
            t2 = perf_counter()
            self.phase_seconds["expand"] += t1 - t0
            self.phase_seconds["merge"] += t2 - t1

    def _event_lookahead(self) -> int:
        """Forward window for this event.

        Fixed ``lookahead_notes`` unless the adaptive variant is enabled;
        then a confident (low-entropy) beam gets the narrow window and the
        window only widens when the candidate entropy says the reading is
        genuinely undecided.
        """
        if not self.adaptive_window:
            return self.lookahead_notes
        weights = [h.weight for h in self.hypotheses]
        total = math.fsum(weights)
        if total <= 0.0 or len(weights) < 2:
            return self.lookahead_min
        entropy = -math.fsum(
            (w / total) * math.log(w / total) for w in weights if w > 0.0
        ) / math.log(len(weights))
        return self.lookahead_max if entropy >= self.entropy_threshold else self.lookahead_min

    def _match(
        self, h: Hypothesis, j: int, event: NoteEvent, pitch_memo: dict[int, float]
    ) -> Hypothesis:
        target = self.score.solo[j]
        skipped = j - h.index
        cost = 0.0

        # pitch: exact note or chromatic error; one computation per distinct
        # target pitch per event, shared across all hypotheses
        pc = pitch_memo.get(target.pitch)
        if pc is None:
            pc = _pitch_cost(abs(target.pitch - event.pitch), self.pitch_scale)
            pitch_memo[target.pitch] = pc
        cost += pc

        # skipped score notes between the old position and this one
        # (negative values are repeat jumps, handled by their own cost)
        if skipped > 0:
            cost += self.skip_base_cost + self.skip_per_note * skipped
        # timing: local expected inter-onset interval under current tempo.
        # This stays comparable whether zero or several notes were skipped,
        # so one missed tail note leaves a competitive alternative reading.
        prev_start = (
            self.score.solo[j - 1].start if j > 0 else max(0.0, target.start - 1.0)
        )
        expected_ioi = (target.start - prev_start) / max(h.tempo, 1e-6)
        observed_ioi = event.time - h.anchor_time
        if expected_ioi > 0:
            normalized = (observed_ioi - expected_ioi) / expected_ioi
            # cap the penalty: a deliberate pause is timing ambiguity, not a
            # hard rejection, so the reading survives and recovers on resume
            cost += self.timing_scale * min(2.25, normalized * normalized)

        # adapt tempo from this observed interval (gradual change, bounded)
        observed_tempo = h.tempo
        if target.start > h.anchor_beat and event.time > h.anchor_time:
            observed_tempo = (target.start - h.anchor_beat) / (event.time - h.anchor_time)
            lo = h.tempo / self.max_tempo_ratio
            hi = h.tempo * self.max_tempo_ratio
            observed_tempo = min(hi, max(lo, observed_tempo))
        new_tempo = (1.0 - self.tempo_smoothing) * h.tempo + self.tempo_smoothing * observed_tempo
        cost += 0.12 * (math.log(new_tempo / max(h.tempo, 1e-9))) ** 2

        held = set(h.held)
        held.discard(target.pitch)
        held.add(event.pitch)
        return Hypothesis(
            index=j + 1,
            anchor_beat=target.start,
            anchor_time=event.time,
            tempo=new_tempo,
            log_weight=h.log_weight - cost,
            held=frozenset(held),
        )

    def _repeat_matches(
        self, h: Hypothesis, first_beat: float, event: NoteEvent,
        pitch_memo: dict[int, float],
    ) -> list[Hypothesis]:
        """Match this event near the start of a notated repeat region.

        A backwards jump is only offered once the reading has reached the
        region's tail (last note), so ordinary in-phrase notes cannot loop.
        Only the opening notes of the region are considered to bound the beam.
        Timing uses the local inter-onset interval like forward matches, and
        the jump itself carries a fixed structural cost.
        """
        region_end = self._repeat_end(first_beat)
        if region_end is None:
            return []
        current_beat = (
            self.score.solo[h.index - 1].start if h.index > 0 else -1.0
        )
        # require the reading to have reached the region tail; tolerate the
        # tail note itself being skipped (one beat short of the boundary)
        if current_beat < region_end - 1.0 - 1e-9:
            return []
        # Temporal signature of a repeat: this onset matches the timing of a
        # note near the region opening rather than the next forward note.
        # Compare its elapsed time since the region opening against what a
        # forward continuation would require, and accept when the forward
        # reading is even slightly late relative to the repeat reading.
        forward_index = min(h.index, len(self.score.solo) - 1)
        forward_note = self.score.solo[forward_index]
        expected_forward_time = h.anchor_time + (
            forward_note.start - h.anchor_beat
        ) / max(h.tempo, 1e-6)
        early_by = expected_forward_time - event.time
        typical_ioi = 1.0 / max(h.tempo, 1e-6)
        if early_by < 0.5 * typical_ioi:
            return []
        out: list[Hypothesis] = []
        for j in self._repeat_openings[first_beat]:
            base = self._match(h, j, event, pitch_memo)
            base.log_weight -= self.repeat_jump_cost
            out.append(base)
        return out

    def _repeat_end(self, first_beat: float) -> float | None:
        for first, last in self.score.repeats:
            if abs(first - first_beat) < 1e-9:
                return last
        return None

    def _seed(self, event: NoteEvent) -> list[Hypothesis]:
        """First key: it may be the first note or any nearby note after skips."""
        out: list[Hypothesis] = []
        for j, note in enumerate(self.score.solo[: self.lookahead_notes]):
            dist = abs(note.pitch - event.pitch)
            cost = _pitch_cost(dist, self.pitch_scale)
            if j > 0:
                cost += self.skip_base_cost + self.skip_per_note * j
            out.append(
                Hypothesis(
                    index=j + 1,
                    anchor_beat=note.start,
                    anchor_time=event.time,
                    tempo=self.initial_tempo,
                    log_weight=-cost,
                    held=frozenset({event.pitch}),
                )
            )
        out.sort(key=lambda h: h.log_weight, reverse=True)
        return self._normalize(out[: self.beam_width])

    def _handle_off(self, event: NoteEvent) -> None:
        self.stats["off_events"] += 1
        for h in self.hypotheses:
            if event.pitch in h.held:
                h.held = frozenset(p for p in h.held if p != event.pitch)
        # an off with no matching held key is ignored: harmless, no position change

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(hyps: list[Hypothesis]) -> list[Hypothesis]:
        # same (index, held set) beliefs collapse by log-sum-exp weight
        best: dict[tuple[int, frozenset[int]], Hypothesis] = {}
        for h in hyps:
            key = (h.index, h.held)
            old = best.get(key)
            if old is None:
                best[key] = h
            else:
                m = max(old.log_weight, h.log_weight)
                merged_weight = m + math.log(
                    math.exp(old.log_weight - m) + math.exp(h.log_weight - m)
                )
                keep = old if old.log_weight >= h.log_weight else h
                keep.log_weight = merged_weight
                best[key] = keep
        return list(best.values())

    @staticmethod
    def _normalize(hyps: list[Hypothesis]) -> list[Hypothesis]:
        if not hyps:
            return hyps
        m = max(h.log_weight for h in hyps)
        total = math.fsum(math.exp(h.log_weight - m) for h in hyps)
        norm = m + math.log(total)
        for h in hyps:
            h.log_weight -= norm
        return hyps

    def _prune(self) -> None:
        self._normalize(self.hypotheses)
        self.hypotheses.sort(key=lambda h: h.log_weight, reverse=True)
        keep = []
        for h in self.hypotheses:
            if h.weight >= 1e-4 or len(keep) < 4:
                keep.append(h)
        self.hypotheses = self._normalize(keep[: self.beam_width])

    def _best(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.log_weight)

    def _make_estimate(self, now: float, is_onset: bool) -> Estimate:
        if not self.hypotheses:
            return Estimate(
                time=now, positions=(), best_beat=0.0, best_index=0,
                tempo=self.initial_tempo, confidence=0.0, margin=0.0,
                expected_index=0, event_count=self.event_count, is_onset=is_onset,
            )
        ranked = sorted(self.hypotheses, key=lambda h: h.log_weight, reverse=True)
        weights = [h.weight for h in ranked]
        total = math.fsum(weights)
        positions = tuple(
            (self.score.solo[h.index - 1].start if h.index > 0 else 0.0, w / total)
            for h, w in zip(ranked, weights)
        )
        best = ranked[0]
        second = ranked[1].weight / total if len(ranked) > 1 else 0.0
        best_w = weights[0] / total
        margin = best_w - second
        # confidence: weight dominance, damped when the best reading is idle
        best_beat = (
            self.score.solo[best.index - 1].start if best.index > 0 else 0.0
        )
        idle = now - best.anchor_time
        # silence dominates the confidence regardless of how many small
        # hypotheses survived pruning: stale certainty is still uncertainty
        idle_damp = math.exp(-0.45 * max(0.0, idle - 1.0))
        ambiguity_floor = min(0.25, 0.05 * max(0.0, idle - 1.5))
        margin = max(0.0, margin - ambiguity_floor)
        confidence = min(1.0, (0.45 + 0.55 * margin) * idle_damp)
        return Estimate(
            time=now,
            positions=positions,
            best_beat=best_beat,
            best_index=max(0, best.index - 1),
            tempo=best.tempo,
            confidence=confidence,
            margin=margin,
            expected_index=best.index,
            event_count=self.event_count,
            is_onset=is_onset,
        )

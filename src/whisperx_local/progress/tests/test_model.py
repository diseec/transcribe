"""Progress model: weighting, monotonicity, interpolation, ETA.

These are the rules that made the old bar unpredictable, so each one is pinned by a
test: it filled and reset per stage, it froze whenever nothing reported, and its jump
to ~60% during speaker separation was never explained.
"""

import unittest

from whisperx_local.progress.model import TIME_CEILING, ProgressModel, Stage


class FakeClock:
    """A monotonic clock the test drives by hand, so no test ever sleeps."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BandsTest(unittest.TestCase):
    def test_bands_are_contiguous_and_cover_the_whole_bar(self):
        model = ProgressModel(clock=FakeClock())
        bands = [model.band(stage.key) for stage in model.stages]
        self.assertAlmostEqual(bands[0][0], 0.0)
        self.assertAlmostEqual(bands[-1][1], 100.0)
        for (_, high), (low, _) in zip(bands, bands[1:]):
            self.assertAlmostEqual(high, low)

    def test_weights_are_normalised(self):
        model = ProgressModel(
            (Stage("a", "A", 1.0), Stage("b", "B", 3.0)), clock=FakeClock()
        )
        self.assertAlmostEqual(model.band("a")[1], 25.0)
        self.assertAlmostEqual(model.band("b")[1], 100.0)

    def test_equal_weights_need_not_sum_to_one(self):
        model = ProgressModel(
            (Stage("a", "A", 2.0), Stage("b", "B", 2.0)), clock=FakeClock()
        )
        self.assertAlmostEqual(model.band("b")[0], 50.0)

    def test_no_stages_is_rejected(self):
        with self.assertRaises(ValueError):
            ProgressModel((), clock=FakeClock())

    def test_unknown_stages_are_rejected(self):
        model = ProgressModel(clock=FakeClock())
        with self.assertRaises(KeyError):
            model.begin("nope")
        with self.assertRaises(KeyError):
            model.skip("nope")


class MonotonicityTest(unittest.TestCase):
    def test_reporting_backwards_is_ignored(self):
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.report(80.0)
        reached = model.percent()
        model.report(20.0)
        self.assertGreaterEqual(model.percent(), reached)

    def test_moving_to_the_next_stage_does_not_reset_the_bar(self):
        # The old dashboard restarted every stage from 0%, which is what made the
        # bar sweep up and snap back six times in one run.
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.report(100.0)
        inside = model.percent()
        model.complete()
        model.begin("align")
        self.assertGreaterEqual(model.percent(), inside)

    def test_a_full_run_never_goes_backwards_and_ends_at_100(self):
        clock = FakeClock()
        model = ProgressModel(
            expected={
                "prepare": 10.0,
                "speech": 5.0,
                "transcribe": 100.0,
                "align": 30.0,
                "speakers": 20.0,
                "write": 1.0,
            },
            clock=clock,
        )
        plan = [
            ("prepare", 10.0, [50.0, 100.0]),
            ("speech", 5.0, [100.0]),
            ("transcribe", 100.0, [10.0, 50.0, 90.0, 100.0]),
            ("align", 30.0, [100.0]),
            ("speakers", 20.0, [50.0, 100.0]),
            ("write", 1.0, [100.0]),
        ]
        observed = []
        for stage, seconds, reports in plan:
            model.begin(stage)
            for portion in reports:
                clock.advance(seconds / len(reports))
                model.report(portion)
                observed.append(model.percent())
            model.complete()
            observed.append(model.percent())

        self.assertEqual(observed, sorted(observed))
        self.assertLessEqual(max(observed), 100.0)
        self.assertAlmostEqual(observed[-1], 100.0)

    def test_completing_every_stage_reaches_exactly_100(self):
        model = ProgressModel(clock=FakeClock())
        for stage in model.stages:
            model.begin(stage.key)
            model.report(100.0)
            model.complete()
        self.assertAlmostEqual(model.percent(), 100.0)


class InterpolationTest(unittest.TestCase):
    def test_a_silent_stage_still_moves(self):
        # Model loading and native alignment report nothing for minutes; the bar has
        # to keep advancing on time alone or it looks hung.
        clock = FakeClock()
        model = ProgressModel(expected={"transcribe": 100.0}, clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        start = model.percent()
        clock.advance(50.0)
        self.assertGreater(model.percent(), start)

    def test_time_alone_never_claims_a_stage_is_finished(self):
        clock = FakeClock()
        model = ProgressModel(expected={"transcribe": 100.0}, clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        clock.advance(100_000.0)
        low, high = model.band("transcribe")
        self.assertLess(model.percent(), high)
        self.assertAlmostEqual(model.percent(), low + (high - low) * TIME_CEILING)

    def test_reports_win_once_they_pass_the_time_estimate(self):
        clock = FakeClock()
        model = ProgressModel(expected={"transcribe": 100.0}, clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        clock.advance(5.0)  # time predicts 5%, the report says 80%
        model.report(80.0)
        low, high = model.band("transcribe")
        self.assertAlmostEqual(model.percent(), low + (high - low) * 0.80)

    def test_a_completed_unit_fills_its_share_even_if_slow(self):
        # Without reports, time still only fills the unit's own share of the stage.
        clock = FakeClock()
        model = ProgressModel(expected={"transcribe": 100.0}, clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        model.set_units(1, 4)
        clock.advance(1000.0)
        low, high = model.band("transcribe")
        self.assertLessEqual(model.percent(), low + (high - low) * TIME_CEILING)


class UnitScalingTest(unittest.TestCase):
    def test_progress_is_scaled_by_the_unit_position(self):
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.set_units(2, 4)
        model.report(50.0)
        low, high = model.band("transcribe")
        self.assertAlmostEqual((model.percent() - low) / (high - low), (1 + 0.5) / 4)

    def test_starting_the_next_unit_does_not_rewind(self):
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.set_units(1, 4)
        model.report(100.0)
        reached = model.percent()
        model.set_units(2, 4)
        self.assertGreaterEqual(model.percent(), reached - 1e-9)

    def test_single_unit_uses_the_whole_stage(self):
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.set_units(1, 1)
        model.report(50.0)
        low, high = model.band("transcribe")
        self.assertAlmostEqual((model.percent() - low) / (high - low), 0.5)


class BandWeightingTest(unittest.TestCase):
    def test_predictions_size_the_bands(self):
        # Equal predicted durations must give equal bands, whatever the static
        # weights say, because the bar is meant to track real time.
        model = ProgressModel(
            (Stage("a", "A", 0.70), Stage("b", "B", 0.30)),
            expected={"a": 60.0, "b": 60.0},
            clock=FakeClock(),
        )
        self.assertAlmostEqual(model.band("a")[1], 50.0)
        self.assertAlmostEqual(model.band("b")[1], 100.0)

    def test_a_stage_with_no_history_keeps_its_relative_standing(self):
        model = ProgressModel(
            (Stage("a", "A", 1.0), Stage("b", "B", 1.0)),
            expected={"a": 90.0},
            clock=FakeClock(),
        )
        self.assertAlmostEqual(model.band("a")[1], 50.0)

    def test_without_predictions_the_static_weights_apply(self):
        model = ProgressModel(
            (Stage("a", "A", 3.0), Stage("b", "B", 1.0)), clock=FakeClock()
        )
        self.assertAlmostEqual(model.band("a")[1], 75.0)

    def test_an_expensively_predicted_stage_gets_a_large_band(self):
        # Measured: speaker separation costs about a third of this pipeline while the
        # built-in weight gives it 9%, which parked the bar at its band end for a
        # minute. The prediction has to be able to correct that.
        model = ProgressModel(
            (
                Stage("transcribe", "Transcribing", 0.70),
                Stage("speakers", "Identifying speakers", 0.09),
            ),
            expected={"transcribe": 73.0, "speakers": 52.0},
            clock=FakeClock(),
        )
        self.assertAlmostEqual(model.band("speakers")[1] - model.band("speakers")[0], 41.6, delta=0.5)


class StageCompletionTest(unittest.TestCase):
    def test_a_source_reporting_100_does_not_finish_the_stage(self):
        # pyannote reports up to 100% early and then keeps working, so trusting that
        # number is what parked the bar.
        model = ProgressModel(clock=FakeClock())
        model.begin("speakers")
        model.report(100.0)
        low, high = model.band("speakers")
        self.assertLess(model.percent(), high)

    def test_completing_the_stage_does_finish_it(self):
        model = ProgressModel(clock=FakeClock())
        model.begin("speakers")
        model.report(100.0)
        model.complete()
        self.assertGreaterEqual(model.percent(), model.band("speakers")[1] - 1e-6)


class SkipTest(unittest.TestCase):
    def test_skipping_an_optional_stage_hands_its_share_over(self):
        # Not everyone wants diarization; the bar must not wait for a stage that
        # was never going to run.
        model = ProgressModel(clock=FakeClock())
        model.begin("transcribe")
        model.report(100.0)
        model.complete()
        model.skip("align")
        model.begin("speakers")
        before = model.percent()
        self.assertGreaterEqual(before, model.band("align")[1] - 1e-6)
        model.report(50.0)
        self.assertGreater(model.percent(), before)

    def test_skipped_stages_are_not_recorded_as_durations(self):
        # A skipped stage costs nothing, and teaching the predictor it takes zero
        # seconds would make every later prediction too small.
        clock = FakeClock()
        model = ProgressModel(clock=clock)
        model.begin("prepare")
        clock.advance(4.0)
        model.complete()
        model.skip("speech")
        self.assertEqual(model.measured(), {"prepare": 4.0})


class EtaTest(unittest.TestCase):
    """The estimate has to be stable, not merely arithmetically correct.

    Chunked work advances in bursts, so an estimate taken from a single instant swung
    between minutes and seconds. These pin the windowed behaviour that replaced it.
    """

    @staticmethod
    def report_overall(model: ProgressModel, target: float) -> None:
        """Drive the model to a target overall percentage, wherever its bands sit."""
        low, high = model.band("transcribe")
        model.report((target - low) / (high - low) * 100.0)

    def running_model(self, clock: FakeClock, expected=None) -> ProgressModel:
        model = ProgressModel(expected=expected, clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        return model

    def test_eta_uses_the_measured_rate(self):
        clock = FakeClock()
        model = self.running_model(clock)
        for target in (10.0, 20.0, 30.0, 40.0, 50.0):
            clock.advance(10.0)
            self.report_overall(model, target)
            model.percent()  # as a repaint would, taking a sample

        self.assertAlmostEqual(model.rate(), 1.0, delta=0.05)
        self.assertAlmostEqual(model.eta(), 50.0, delta=6.0)

    def test_a_stalled_window_yields_no_rate(self):
        # Nothing moved for a minute, which says nothing about the work left.
        clock = FakeClock()
        model = self.running_model(clock, expected={"transcribe": 100.0})
        self.report_overall(model, 50.0)
        for _ in range(6):
            clock.advance(10.0)
            model.percent()
        self.assertIsNone(model.rate())

    def test_a_stalled_window_falls_back_to_the_prediction(self):
        clock = FakeClock()
        model = self.running_model(clock, expected={"transcribe": 100.0})
        self.report_overall(model, 50.0)
        for _ in range(6):
            clock.advance(10.0)
            model.percent()
        eta = model.eta()
        self.assertIsNotNone(eta)
        self.assertLess(eta, 100.0)

    def test_eta_falls_back_to_predictions_before_any_progress(self):
        model = ProgressModel(
            expected={"prepare": 10.0, "speech": 5.0}, clock=FakeClock()
        )
        self.assertAlmostEqual(model.eta(), 15.0)

    def test_eta_is_unknown_without_predictions_or_progress(self):
        self.assertIsNone(ProgressModel(clock=FakeClock()).eta())

    def test_eta_never_goes_negative_at_the_end(self):
        model = ProgressModel(clock=FakeClock())
        for stage in model.stages:
            model.begin(stage.key)
            model.report(100.0)
            model.complete()
        self.assertGreaterEqual(model.eta() or 0.0, 0.0)

    def test_predicted_total_sums_the_known_stages(self):
        model = ProgressModel(
            expected={"transcribe": 100.0, "align": 30.0}, clock=FakeClock()
        )
        self.assertAlmostEqual(model.predicted_total(), 130.0)

    def test_predicted_total_is_unknown_without_predictions(self):
        self.assertIsNone(ProgressModel(clock=FakeClock()).predicted_total())


class RateTest(unittest.TestCase):
    def test_a_single_sample_is_not_a_rate(self):
        clock = FakeClock()
        model = ProgressModel(clock=clock)
        model.percent()
        self.assertIsNone(model.rate())

    def test_repainting_faster_than_the_interval_records_one_sample(self):
        # The dashboard repaints ten times a second; the history must not grow at
        # that rate or the window would span a fraction of a second.
        clock = FakeClock()
        model = ProgressModel(clock=clock)
        for _ in range(5):
            clock.advance(0.1)
            model.percent()
        self.assertIsNone(model.rate())

    def test_a_gap_in_the_history_leaves_no_rate(self):
        clock = FakeClock()
        model = ProgressModel(clock=clock)
        model.begin("transcribe")
        model.skip("prepare", "speech")
        EtaTest.report_overall(model, 10.0)
        clock.advance(5.0)
        model.percent()
        EtaTest.report_overall(model, 20.0)
        clock.advance(5.0)
        model.percent()
        clock.advance(100.0)  # those samples fall out of the window
        EtaTest.report_overall(model, 30.0)
        model.percent()
        self.assertIsNone(model.rate())


if __name__ == "__main__":
    unittest.main()

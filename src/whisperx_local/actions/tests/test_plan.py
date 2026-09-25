"""Planning: turning a request into the smallest set of work that satisfies it.

This is the behaviour that makes subsets real, so every interesting combination is
pinned. The two that matter most: asking for one stage when its inputs already exist
must not run the stages before it, and asking for a stage whose inputs are missing
must pull in only the producers needed.
"""

import unittest

from whisperx_local.actions import catalog, plan as planning
from whisperx_local.actions.base import AUDIO, FILES, PLAN, REPORT, SEGMENTS, TURNS, WORDS
from whisperx_local.actions.plan import Selection, parse_only, resolve


class CatalogTest(unittest.TestCase):
    def test_every_action_is_reachable_by_key(self):
        self.assertEqual(set(catalog.keys()), set(catalog.BY_KEY))

    def test_actions_are_listed_in_dependency_order(self):
        positions = {action.key: index for index, action in enumerate(catalog.ACTIONS)}
        for action in catalog.ACTIONS:
            for need in action.requires:
                producers = [
                    other for other in catalog.ACTIONS if need in other.provides
                ]
                for producer in producers:
                    with self.subTest(action=action.key, need=need):
                        self.assertLess(positions[producer.key], positions[action.key])

    def test_the_default_set_is_a_subset_of_the_known_actions(self):
        self.assertTrue(set(catalog.DEFAULT_ACTIONS) <= set(catalog.keys()))

    def test_an_unknown_action_explains_the_choices(self):
        with self.assertRaises(KeyError) as caught:
            catalog.get("summarise")
        self.assertIn("summarise", str(caught.exception))
        self.assertIn("transcribe", str(caught.exception))

    def test_options_are_offered_for_a_selection_without_duplicates(self):
        options = catalog.options_for(("transcribe", "align"))
        names = [option.name for option in options]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("language", names)
        self.assertIn("align_retries", names)

    def test_options_belong_to_the_actions_that_use_them(self):
        names = [option.name for option in catalog.options_for(("align",))]
        self.assertNotIn("language", names)
        self.assertIn("align_retries", names)

    def test_every_option_describes_itself(self):
        for action in catalog.ACTIONS:
            for option in action.options:
                with self.subTest(option=option.name):
                    self.assertTrue(option.help)
                    self.assertTrue(option.describe())

    def test_a_choice_option_lists_its_values_as_the_hint(self):
        language = catalog.get("transcribe")
        option = next(entry for entry in language.options if entry.name == "language")
        self.assertEqual(option.describe(), "fa|en")


class ParseOnlyTest(unittest.TestCase):
    def test_a_comma_list(self):
        self.assertEqual(parse_only("transcribe,align"), ("transcribe", "align"))

    def test_a_plus_list(self):
        self.assertEqual(parse_only("transcribe+align"), ("transcribe", "align"))

    def test_spaces_are_tolerated(self):
        self.assertEqual(parse_only(" transcribe , align "), ("transcribe", "align"))

    def test_a_single_name(self):
        self.assertEqual(parse_only("export"), ("export",))

    def test_an_empty_string_selects_nothing(self):
        self.assertEqual(parse_only(""), ())


class ResolveDefaultsTest(unittest.TestCase):
    def test_the_default_set_needs_no_additions(self):
        result = resolve(Selection(), available=())
        self.assertEqual(result.keys(), catalog.DEFAULT_ACTIONS)
        self.assertEqual(result.added, ())
        self.assertEqual(result.blocked, ())

    def test_removing_a_stage_drops_only_that_stage(self):
        result = resolve(Selection(without=("diarize",)), available=())
        self.assertNotIn("diarize", result.keys())
        self.assertIn("export", result.keys())

    def test_removing_everything_is_a_valid_but_empty_plan(self):
        result = resolve(Selection(without=catalog.DEFAULT_ACTIONS), available=())
        self.assertEqual(result.keys(), ())
        self.assertFalse(result)


class ResolveSubsetTest(unittest.TestCase):
    def test_a_stage_with_its_inputs_present_runs_alone(self):
        # The whole point: re-rendering subtitles must not re-transcribe an hour.
        result = resolve(Selection(only=("export",)), available={SEGMENTS})
        self.assertEqual(result.keys(), ("export",))
        self.assertEqual(result.added, ())

    def test_speakers_can_be_added_to_an_existing_transcript(self):
        result = resolve(Selection(only=("diarize",)), available={SEGMENTS, AUDIO})
        self.assertEqual(result.keys(), ("diarize",))

    def test_missing_inputs_pull_in_only_their_producers(self):
        result = resolve(Selection(only=("export",)), available=())
        self.assertEqual(result.keys(), ("prepare", "boundaries", "transcribe", "export"))
        self.assertEqual(set(result.added), {"prepare", "boundaries", "transcribe"})

    def test_a_producer_with_its_own_needs_pulls_those_in_too(self):
        # Alignment needs segments, which needs audio and a boundary plan.
        result = resolve(Selection(only=("align",)), available=())
        self.assertEqual(
            result.keys(), ("prepare", "boundaries", "transcribe", "align")
        )

    def test_an_added_producer_is_not_added_twice(self):
        result = resolve(Selection(only=("export", "align")), available=())
        self.assertEqual(result.keys().count("transcribe"), 1)

    def test_analysis_needs_the_text(self):
        result = resolve(Selection(only=("analyze",)), available={SEGMENTS})
        self.assertEqual(result.keys(), ("analyze",))

    def test_an_excluded_dependency_blocks_its_dependent(self):
        # Refusing to transcribe and asking to export cannot both be honoured, so the
        # plan reports the conflict instead of failing halfway through.
        result = resolve(
            Selection(only=("export",), without=("transcribe",)), available=()
        )
        self.assertNotIn("export", result.keys())
        self.assertIn("export", result.blocked)

    def test_an_unknown_action_is_rejected(self):
        with self.assertRaises(KeyError):
            resolve(Selection(only=("summarise",)), available=())


class ResolveArtifactsTest(unittest.TestCase):
    def test_the_plan_reports_what_will_exist_afterwards(self):
        result = resolve(Selection(only=("transcribe",)), available=())
        self.assertTrue({AUDIO, PLAN, SEGMENTS} <= set(result.available))

    def test_available_artifacts_are_carried_through(self):
        result = resolve(Selection(only=("export",)), available={SEGMENTS, TURNS, WORDS})
        self.assertTrue({SEGMENTS, TURNS, WORDS, FILES} <= set(result.available))

    def test_analysis_reports_a_report(self):
        result = resolve(Selection(only=("analyze",)), available={SEGMENTS})
        self.assertIn(REPORT, result.available)


class CanonicalTest(unittest.TestCase):
    def test_only_wins_over_the_default_set(self):
        self.assertEqual(planning.canonical(Selection(only=("export",))), ("export",))

    def test_extra_is_appended_to_the_default_set(self):
        result = planning.canonical(Selection(extra=("analyze",)))
        self.assertIn("analyze", result)
        self.assertIn("transcribe", result)

    def test_extra_already_present_is_not_duplicated(self):
        result = planning.canonical(Selection(extra=("transcribe",)))
        self.assertEqual(result.count("transcribe"), 1)

    def test_without_is_applied_last(self):
        result = planning.canonical(Selection(extra=("analyze",), without=("analyze",)))
        self.assertNotIn("analyze", result)


if __name__ == "__main__":
    unittest.main()

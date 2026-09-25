"""The menu: its layout, its key handling, and its behaviour without a terminal.

Two properties here are worth more than the rest, because both fail invisibly or
catastrophically:

* It must work with no terminal at all. A menu that reads a keypress from a pipe is a
  menu that hangs, and a redraw written into a pipe is a pile of escape sequences.
* A list taller than the terminal must be windowed. Drawing past the bottom makes the
  terminal scroll, and moving the cursor up by the number of rows written then lands
  somewhere else entirely, leaving a trail of half-drawn frames down the screen. The
  settings screen is longer than a typical window, so this is not hypothetical.
"""

import os
import unittest
from unittest import mock

from whisperx_local.ui.menu import Choice, Menu


class Sink:
    """Swallows writes and reports itself as not a terminal."""

    def __init__(self, is_tty: bool = False):
        self.is_tty = is_tty
        self.written = ""

    def write(self, text):
        self.written += text
        return len(text)

    def flush(self):
        return None

    def isatty(self):
        return self.is_tty


def menu(**kwargs) -> Menu:
    return Menu(out=Sink(), **kwargs)


class RowLayoutTest(unittest.TestCase):
    def test_rows_are_numbered_from_one(self):
        rows = Menu.rows([Choice("First"), Choice("Second")], 0)
        self.assertIn(" 1.", rows[0])
        self.assertIn(" 2.", rows[1])

    def test_the_cursor_marks_exactly_one_row(self):
        rows = Menu.rows([Choice("First"), Choice("Second"), Choice("Third")], 1)
        marked = [row for row in rows if "❯" in row]
        self.assertEqual(len(marked), 1)
        self.assertIn("Second", marked[0])

    def test_a_detail_is_shown_when_there_is_one(self):
        self.assertIn("explanation", Menu.rows([Choice("Label", "explanation")], 0)[0])

    def test_multi_select_shows_which_are_marked(self):
        rows = Menu.rows([Choice("One"), Choice("Two")], 0, chosen=(1,), multi=True)
        self.assertIn("[ ]", rows[0])
        self.assertIn("[x]", rows[1])

    def test_single_select_has_no_checkbox(self):
        self.assertNotIn("[ ]", Menu.rows([Choice("One")], 0)[0])

    def test_every_choice_gets_a_row(self):
        choices = [Choice(f"item {index}") for index in range(6)]
        self.assertEqual(len(Menu.rows(choices, 0)), 6)

    def test_an_offset_keeps_the_numbering_global(self):
        # A windowed list still has to number rows as the reader sees them.
        rows = Menu.rows([Choice("hidden"), Choice("shown")], 0, offset=5)
        self.assertIn(" 6.", rows[0])
        self.assertIn(" 7.", rows[1])


class ViewportTest(unittest.TestCase):
    """A list taller than the window must not break the redraw."""

    def long_choices(self, count=40) -> list[Choice]:
        return [Choice(f"setting {index}") for index in range(count)]

    def test_a_short_list_is_shown_whole(self):
        choices = self.long_choices(3)
        self.assertEqual(len(menu().frame(choices, 0)), 3)

    def test_a_long_list_is_windowed(self):
        entry = menu()
        frame = entry.frame(self.long_choices(40), 0)
        self.assertLessEqual(len(frame), entry._viewport_limit() + 2)

    def test_what_is_scrolled_off_is_counted(self):
        frame = menu().frame(self.long_choices(40), 0)
        self.assertTrue(any("below" in row for row in frame))

    def test_scrolling_down_reveals_the_later_rows(self):
        entry = menu()
        frame = entry.frame(self.long_choices(40), 39)
        self.assertTrue(any("above" in row for row in frame))
        self.assertTrue(any("setting 39" in row for row in frame))

    def test_the_cursor_is_always_visible(self):
        entry = menu()
        choices = self.long_choices(40)
        for cursor in (0, 1, 19, 20, 38, 39):
            with self.subTest(cursor=cursor):
                frame = entry.frame(choices, cursor)
                self.assertTrue(any("❯" in row for row in frame))
                self.assertTrue(any(f"setting {cursor}" in row for row in frame))

    def test_the_drawn_height_never_grows_between_frames(self):
        # The redraw moves up by the previous height, so it has to be stable.
        entry = menu()
        choices = self.long_choices(40)
        entry._paint(entry.frame(choices, 0) + ["hint"])
        first = entry._height
        entry._paint(entry.frame(choices, 39) + ["hint"])
        self.assertEqual(entry._height, first)

    def test_a_windowed_frame_does_not_exceed_the_terminal(self):
        entry = menu()
        with mock.patch(
            "shutil.get_terminal_size", return_value=os.terminal_size((100, 24))
        ):
            frame = entry.frame(self.long_choices(40), 10)
        self.assertLess(len(frame) + 1, 24)


class NonTerminalTest(unittest.TestCase):
    """Without a terminal the menu still works, by number instead of by arrow key."""

    def answer(self, text, **kwargs):
        entry = menu()
        with mock.patch("builtins.input", return_value=text):
            return entry.choose("Pick", [Choice("a"), Choice("b"), Choice("c")], **kwargs)

    def test_a_number_selects_that_row(self):
        self.assertEqual(self.answer("2"), 1)

    def test_an_empty_line_accepts_the_first_row(self):
        self.assertEqual(self.answer(""), 0)

    def test_the_last_number_among_several_wins(self):
        self.assertEqual(self.answer("1 3"), 2)

    def test_an_out_of_range_number_is_ignored_until_a_valid_one_arrives(self):
        entry = menu()
        with mock.patch("builtins.input", side_effect=["9", "2"]):
            self.assertEqual(entry.choose("Pick", [Choice("a"), Choice("b")]), 1)

    def test_multi_select_returns_a_sorted_tuple(self):
        self.assertEqual(self.answer("3 1", multi=True), (0, 2))

    def test_multi_select_can_accept_the_current_answer(self):
        self.assertEqual(self.answer("", multi=True, chosen=(1,)), (1,))

    def test_cancelling_returns_nothing(self):
        entry = menu()
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(entry.choose("Pick", [Choice("a")]))

    def test_no_choices_is_not_an_error(self):
        self.assertIsNone(menu().choose("Pick", []))

    def test_multi_select_of_nothing_is_an_empty_choice(self):
        self.assertEqual(menu().choose("Pick", [], multi=True), ())


class TerminalDetectionTest(unittest.TestCase):
    def test_a_pipe_is_not_a_terminal(self):
        self.assertFalse(Menu(out=Sink(is_tty=False)).interactive)

    def test_output_to_a_terminal_still_needs_input_from_one(self):
        # Both ends matter: redrawing into a pipe is worse than not redrawing.
        entry = Menu(out=Sink(is_tty=True))
        self.assertEqual(entry.interactive, entry.keys.available)

    def test_a_screen_gets_a_banner(self):
        out = Sink()
        Menu(out=out).banner("Title", "Subtitle")
        self.assertIn("Title", out.written)
        self.assertIn("Subtitle", out.written)


if __name__ == "__main__":
    unittest.main()

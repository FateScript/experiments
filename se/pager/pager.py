#!/usr/bin/env python3

import bisect
import functools
import itertools
import math
import sys
import curses

from typing import Tuple

from console import get_press_key, termainal_size

__all__ = ["Pager"]


HELP_TEXT = """
Simple pager commands:

j, +, <DOWN-ARROW>
    Forward one line.
k, -, <UP-ARROW>
    Back one line.
q, Q, ^C, ^D, ^Z
    Quit return to the caller.

Hit any key to continue:"""


def clear_screen():
    print("\033[H\033[J")


class Pager:

    def __init__(self, content):
        # TODO: percent bar
        self._content = content
        self.content_lines = content.splitlines()
        self.content_length = len(self.content_lines)

        self.prev_begin, self.prev_end = 0, self.content_length

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()

    def render(self):
        """
        Simple pager commands, render in terminal.
        Press q/esc/ctrl+d to quit.
        j/down arrow/+ to forward one line.
        k/up arrow/- to back one line.
        """
        _, row = self.start_end_lines_number(0, as_end=False)
        self.prev_end = row
        text = "\n".join(self.content_lines[0:row]) + "\n"
        self.write(text)

        while True:
            key = get_press_key()
            if key in ("q", "Q", "<ESCAPE>", "<EOF>"):
                break
            elif key in ("j", "J", "+", "<DOWN-ARROW>"):
                end = self.prev_end + 1
                if end >= self.content_length:  # reach the end of the content
                    end = self.content_length - 1
                else:
                    self.write("\n".join(self.content_lines[self.prev_end:end]) + "\n")
                    self.prev_begin = self.prev_begin + 1
                self.prev_end = end

            elif key in ("k", "K", "-", "<UP-ARROW>"):
                begin = self.prev_begin - 1
                if begin < 0:
                    begin = 0
                else:
                    clear_screen()
                    self.prev_end = self.prev_end - 1
                    begin, _ = self.start_end_lines_number(self.prev_end, as_end=True)
                    self.write("\n".join(self.content_lines[begin:self.prev_end]) + "\n")
                self.prev_begin = begin

    @functools.lru_cache
    def line_prefix_sum(self, columns: int):
        line_per_text = [math.ceil(len(line) / columns) for line in self.content_lines]
        prefix_sum = itertools.accumulate(line_per_text, func=lambda x, y: x + y)
        return list(prefix_sum)

    def start_end_lines_number(self, idx: int, as_end: bool = True) -> Tuple[int, int]:
        """Start and end lines number to display the content.

        Args:
            idx (int): The current line number.
            as_end (bool): If True, idx is the end line number.
                Otherwise, idx is the start line number. Default to True.

        Returns:
            Tuple[int, int]: The start and end line number, inclusive format.
        """
        col, row = termainal_size()
        prefix_sum = self.line_prefix_sum(col)
        valid_row = row - 1
        if as_end:  # end first
            line_number = prefix_sum[idx]
            start = bisect.bisect_right(prefix_sum, line_number - valid_row) + 1
            return (start, idx)
        else:  # start first
            line_number = prefix_sum[idx]
            end = bisect.bisect_left(prefix_sum, line_number + valid_row) - 1
            return (idx, end)

    def curse_render(self):
        scr = curses.initscr()
        try:
            self.curse_view(scr)
        finally:
            curses.endwin()

    def curse_view(self, screen: curses.window):
        screen.clear()
        _, end = self.start_end_lines_number(0, as_end=False)
        self.prev_end = end
        for idx in range(end + 1):
            try:
                screen.addstr(self.content_lines[idx] + "\n")
            except curses.error:  # exceed the screen size
                self.prev_end = idx - 1
        screen.refresh()

        while True:
            key = get_press_key()
            if key in ("q", "Q", "<ESCAPE>", "<EOF>"):
                break
            elif key in ("j", "J", "+", "<DOWN-ARROW>"):
                end = self.prev_end + 1
                if end >= self.content_length:  # reach the end of the content
                    end = self.content_length - 1
                begin, end = self.start_end_lines_number(end, as_end=True)

            elif key in ("k", "K", "-", "<UP-ARROW>"):
                begin = self.prev_begin - 1
                if begin < 0:
                    begin = 0
                begin, end = self.start_end_lines_number(begin, as_end=False)

            elif key in ("h", "H", "?"):
                screen.clear()
                screen.addstr(HELP_TEXT)
                screen.refresh()
                get_press_key()

            self.prev_begin, self.prev_end = begin, end
            screen.clear()
            for idx in range(begin, end + 1):
                try:
                    screen.addstr(self.content_lines[idx] + "\n")
                except curses.error:  # exceed the screen size
                    self.prev_end = idx - 1
                    pass
            screen.refresh()

    def help(self):
        print(HELP_TEXT)
        get_press_key()


if __name__ == "__main__":
    import string
    import random
    random.seed(42)

    values = [
        "".join([random.choice(string.ascii_letters) for _ in range(random.randint(20, 1000))])
        for _ in range(100)
    ]

    x = Pager("\n".join(values))
    # x.render()
    x.curse_render()

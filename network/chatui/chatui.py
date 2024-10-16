#!/usr/bin/env python3

import sys
import os


def init_windows():
    print_now(clear_screen())


def read_command(prompt="> "):
    lines = get_terminal_lines()

    buf = position_cursor(lines - 1)
    buf += clear_line()
    buf += prompt

    print_now(buf)

    s = sys.stdin.readline().strip()

    return s


def print_message(s):
    lines = get_terminal_lines()
    line = lines - 3

    buf = save_cursor_position()
    buf += set_scrolling_region(line)
    buf += position_cursor(line)
    buf += '\n' + s
    buf += set_scrolling_region()
    buf += restore_cursor_position()

    print_now(buf)


def end_windows():
    pass


def print_now(s):
    print(s, end="", flush=True)


def get_terminal_lines():
    _, lines = os.get_terminal_size()
    return lines


def clear_line():
    return "\x1b[2K"


def clear_screen():
    return "\x1b[2J"


def save_cursor_position():
    # return "\x1b7\x1b[s"
    # return "\x1b[s"
    return "\x1b7"


def restore_cursor_position():
    # return "\x1b[u\x1b8"
    # return "\x1b[u"
    return "\x1b8"


def position_cursor(row, col=1):
    return f"\x1b[{row};{col}f"


def set_scrolling_region(line0=None, line1=None):
    if line0 is None:
        return "\x1b[r"

    if line1 is None:
        line1 = line0
        line0 = 1

    return f"\x1b[{line0};{line1}r"

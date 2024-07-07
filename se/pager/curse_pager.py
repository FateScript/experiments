#!/usr/bin/env python3

import curses
import os
import sys


def curse_pager(stdscr: curses.window, text: str):
    curses.curs_set(0)  # hide the cursor
    stdscr.clear()

    lines = text.splitlines()
    max_y, max_x = stdscr.getmaxyx()

    offset = 0
    while True:
        stdscr.clear()
        for i, line in enumerate(lines[offset:offset + max_y - 1]):
            stdscr.addstr(i, 0, line[:max_x - 1])
        stdscr.refresh()

        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key in (curses.KEY_DOWN, ord("j")) and offset < len(lines) - max_y:
            offset += 1
        elif key in (curses.KEY_UP, ord("k")) and offset > 0:
            offset -= 1
        elif key == curses.KEY_NPAGE and offset < len(lines) - max_y:
            offset = min(offset + max_y - 1, len(lines) - max_y)
        elif key == curses.KEY_PPAGE and offset > 0:
            offset = max(offset - max_y + 1, 0)


def main(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'r') as f:
        text = f.read()

    curses.wrapper(curse_pager, text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <filename>")
        sys.exit(1)

    main(sys.argv[1])

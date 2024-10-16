#!/usr/bin/env python3

# python3 -m pip install uni-curses
from unicurses import (
    cbreak,
    clear,
    clrtoeol,
    endwin,
    getmaxyx,
    getstr,
    getyx,
    initscr,
    move,
    newwin,
    nocbreak,
    noraw,
    refresh,
    scrollok,
    waddstr,
    wmove,
    wrefresh,
    mvaddstr,
)


upper_window = None


def init_windows():
    global stdscr
    global upper_window

    stdscr = initscr()

    # cbreak() is ignored by getstr()
    # My eyes ze goggles do nothing
    cbreak()

    clear()

    lines, cols = getmaxyx(stdscr)
    upper_window = newwin(lines - 3, cols, 0, 0)
    wmove(upper_window, 0, 0)
    scrollok(upper_window, True)


def read_command(prompt="> "):
    lines, _ = getmaxyx(stdscr)
    line = lines - 2

    move(line, 0)
    clrtoeol()
    mvaddstr(line, 0, prompt)

    refresh()

    s = getstr()

    # Search for CTRL-C
    #
    # This is so hackish. cbreak mode doesn't seem to affect getstr()
    # despite the documentation. And I don't want to write my own
    # cbreak-aware getstr() with getch().

    for c in s:
        if c == '\x03':
            raise KeyboardInterrupt

    return s


def print_message(s):
    global upper_window
    global stdscr

    pos = getyx(stdscr)

    waddstr(upper_window, "\n" + s)
    wrefresh(upper_window)

    move(*pos)

    refresh()


def end_windows():
    clear()
    refresh()
    nocbreak()
    noraw()
    endwin()

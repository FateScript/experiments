#!/usr/bin/env python3

import os
import sys
import tty
import termios
import select

ESC = "\x1b"  # ctrl + 3 also
EOF = "\x04"  # unix (^D), for windows it's "\x1a" (^Z)

KEY_SUFFIX = {
    "A": "<UP-ARROW>",
    "B": "<DOWN-ARROW>",
    "D": "<LEFT-ARROW>",
    "C": "<RIGHT-ARROW>",
    "5": "<PAGE-UP>",
    "6": "<PAGE-DOWN>",
    "H": "<HOME>",
    "F": "<END>",
    "M": "<DOWN-ARROW>",
    "S": "<PAGE-UP>",
    "T": "<PAGE-DOWN>",
}


def getchar(num_bytes: int = 1) -> str:
    """Press key + enter and get the char in stdin."""
    sys.stdout.flush()
    stdin_fd = sys.stdin.fileno()
    char: bytes = os.read(stdin_fd, num_bytes)
    return char.decode()


def get_press_key() -> str:
    """Press key and get the key in string.
    If the key is an escape sequence, return the key name inside '<>'.
    """
    sys.stdout.flush()
    stdin_fd = sys.stdin.fileno()

    def _get_key(num_bytes: int = 1) -> str:
        return os.read(stdin_fd, num_bytes).decode()

    def _get_key_timeout() -> str:
        f = select.select([sys.stdin], [], [], 0.1)[0]
        if not f:
            return None
        cur_io = f[0]

        cur_io.seek(os.SEEK_END)
        length = cur_io.tell()
        return cur_io.read(length)

    old_settings = termios.tcgetattr(stdin_fd)

    try:
        tty.setraw(stdin_fd)
        char = _get_key()
        if char == ESC:
            next_char = _get_key_timeout()
            if next_char is None:
                return "<ESCAPE>"
            elif next_char.startswith("["):
                next_char = next_char[1:]
                if next_char in KEY_SUFFIX:
                    return KEY_SUFFIX[next_char]
                else:
                    return f"<UNKNOWN: {next_char}>"
            else:
                return f"<UNKNOWN: {next_char}>"
        elif char == EOF:
            return "<EOF>"

        return char
    except Exception as e:
        print(e)
        return "<ERROR>"
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)


def fd_size(fd: int):
    """Returns the terminal (x,y) size for fd.

    Args:
        fd: The terminal file descriptor.
    """
    import struct
    import fcntl
    try:
        # This magic incantation converts a struct from ioctl(2) containing two
        # binary shorts to a (rows, columns) int tuple.
        rc = struct.unpack(b"hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "junk"))
        return (rc[1], rc[0]) if rc else None
    except Exception:
        return None


def termianl_size_backup():
    in_fd, out_fd, err_fd = sys.stdin.fileno(), sys.stdout.fileno(), sys.stderr.fileno()
    xy = fd_size(in_fd) or fd_size(out_fd) or fd_size(err_fd)

    if not xy:  # use /dev/tty as a fallback
        fd = None
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            xy = fd_size(fd)
        except Exception:
            xy = None
        finally:
            if fd is not None:
                os.close(fd)

    if not xy:  # fallback to environment variables
        xy = (int(os.environ["COLUMNS"]), int(os.environ["LINES"]))

    return xy


def termainal_size():
    """Get the terminal size."""
    size = os.get_terminal_size()
    return size.columns, size.lines

#!/usr/bin/env python3

# ackermann function grows very fast, so be careful with large values of m and n.
# For example, ackermann(3, 4) is already a very large number.
# ackermann(4, 1) will crash by default, but its value is 65533.
# Ackermann function shows the difference between `computable` and `primitive recursive`.


def ackermann(m: int, n: int, indent: int = 0, verbose: bool = False) -> int:
    if verbose:
        print("  " * indent + f'ackermann({m}, {n})')
    indent += 1

    assert m >= 0 and n >= 0
    if m == 0:
        return n + 1
    # m > 0
    elif n == 0:
        return ackermann(m - 1, 1, indent, verbose)
    else:
        return ackermann(m - 1, ackermann(m, n - 1, indent, verbose), indent, verbose)


CACHE = {}


def cached_ackermann(m: int, n: int) -> int:
    global CACHE

    if (m, n) in CACHE:
        return CACHE[(m, n)]

    if m == 0:
        result = n + 1
    # m > 0
    elif n == 0:
        result = cached_ackermann(m - 1, 1)
    else:
        result = cached_ackermann(m - 1, cached_ackermann(m, n - 1))

    CACHE[(m, n)] = result
    return result


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(50000)
    import fire
    fire.Fire({
        '_': ackermann,  # default command
        'cached': cached_ackermann,
    })

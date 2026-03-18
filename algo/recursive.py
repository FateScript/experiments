# sovle 17:10 problem in https://www.bilibili.com/video/BV1KNPHzPEra
import pytest


def f(n):
    print(f"f({n})")
    if n <= 1:
        return 1
    else:
        return f(n-1) + g(n-2)


def g(n):
    print(f"g({n})")
    if n <= 1:
        return 1
    else:
        return f(n+1) + g(n-2)


def non_recursive_call(n):
    stack = []
    stack.append(('f', n))
    while len(stack) > 0:
        func, value = stack.pop()
        print(f"{func}({value})")
        if func == "f":
            if value <= 1:
                continue
            else:
                stack.append(('g', value - 2))
                stack.append(('f', value - 1))
        elif func == "g":
            if value <= 1:
                continue
            else:
                stack.append(('g', value - 2))
                stack.append(('f', value + 1))
            pass
        else:
            raise ValueError("Unknow fucntion")


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_non_recursive_call_same_as_recursive(capsys, n):
    f(n)
    captured_f = capsys.readouterr()

    non_recursive_call(n)
    captured_non_recursive = capsys.readouterr()
    assert captured_f.out == captured_non_recursive.out, f"n={n} missmatch"


if __name__ == "__main__":
    f(5)
    print("-----")
    non_recursive_call(5)

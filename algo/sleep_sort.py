#!/usr/bin/env python3

import ast
import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from typing import List


def sleep_sort(array: List) -> List:
    workers = len(array)

    def f(x):
        time.sleep(x)
        print(x)

    buf = io.StringIO()
    with redirect_stdout(buf):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(f, array)

    sorted_text = buf.getvalue()
    sorted_data = [ast.literal_eval(x) for x in sorted_text.split()]
    return sorted_data


def test_sleep_sort():
    data_to_test = [
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
        [3.1, 1.2, 4.3, 1.1, 5.5, 7, 2, 6.0, 5.9, 3.2, 5.95],
    ]
    for data in data_to_test:
        sorted_data = sleep_sort(data)
        assert sorted_data == sorted(data)
    print("sleep_sort passed")


if __name__ == "__main__":
    test_sleep_sort()

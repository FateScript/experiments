#!/usr/bin/env python3

# Bitonic sort is a algorithm used for parallel-friendly device such as GPU.
# Number of pass: T(n) = 1 + 2 + ... + log(n) = O(log(n) ** 2)
# Computation complexity: O( n*log^2(n) )
# Reference
# https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting

import math
import random
from enum import Enum
from typing import List


class DIR(Enum):
    UP = True
    DOWN = False


def swap_by_direction(array: List, idx1: int, idx2: int, direction: DIR):
    # for clarity, not effiency
    max_val, min_val = max(array[idx1], array[idx2]), min(array[idx1], array[idx2])
    if direction == DIR.DOWN:
        array[idx1], array[idx2] = max_val, min_val
    else:
        array[idx1], array[idx2] = min_val, max_val


def bitonic_subarray(array: List, start_idx: int, end_idx: int, step: int, direction: DIR):
    """Index here is a closeset, which means [start_idx, end_idx]"""
    if step == 0:
        return
    for i in range(start_idx, end_idx - step + 1):
        swap_by_direction(array, i, i + step, direction)

    step //= 2
    mid_idx = (start_idx + end_idx) // 2
    bitonic_subarray(array, start_idx, mid_idx, step, direction)
    bitonic_subarray(array, mid_idx + 1, end_idx, step, direction)


def bitonic_sort(array: List, descending: bool = False):
    """inplace sort the input array"""
    length = len(array)
    stages = math.ceil(math.log2(length))
    assert length == 2 ** stages, f"only power of 2 length is supported, get {length}"
    for stage in range(stages):
        step = 2 ** stage
        idx_range = 2 * step
        dir_ = DIR.DOWN if descending else DIR.UP
        for start_idx in range(0, length, idx_range):
            bitonic_subarray(array, start_idx, start_idx + idx_range - 1, step, dir_)
            dir_ = DIR(not dir_.value)


def bitonic_merge_array(array: List, start: int, end: int, direction: DIR):
    length = end - start + 1
    if length == 1:
        return

    step = length // 2
    mid = start + step - 1  # mid is also (start + end) // 2, like in `bitonic_sort_array`
    for i in range(start, start + step):
        swap_by_direction(array, i, i + step, direction)
    bitonic_merge_array(array, start, mid, direction)
    bitonic_merge_array(array, mid + 1, end, direction)


def bitonic_sort_array(array: List, start: int, end: int, direction: DIR):
    length = end - start + 1
    if length == 1:
        return

    mid = (start + end) // 2
    bitonic_sort_array(array, start, mid, direction)
    bitonic_sort_array(array, mid + 1, end, DIR(not direction.value))
    bitonic_merge_array(array, start, end, direction)


def bitonic_sort_algo(array: List, descending: bool = False):
    # most used implementment of bitonic sort
    length = len(array)
    stages = math.ceil(math.log2(length))
    assert length == 2 ** stages, f"only power of 2 length is supported, get {length}"
    dir_ = DIR.DOWN if descending else DIR.UP
    bitonic_sort_array(array, 0, length - 1, dir_)


def test_bitonic_sort(array_length=32, test_loop=10, algo=bitonic_sort):
    array = list(range(array_length))
    for _ in range(test_loop):
        copy_array = array.copy()
        random.shuffle(copy_array)
        prev_val = copy_array.copy()
        algo(copy_array)
        assert copy_array == array, f"bitonic_sort failed at {prev_val}"
    print(f"{algo.__name__} passed.")


if __name__ == "__main__":
    test_bitonic_sort(test_loop=100)
    test_bitonic_sort(test_loop=100, algo=bitonic_sort_algo)

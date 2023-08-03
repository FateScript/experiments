#!/usr/bin/env python3

# Bitonic sort is a algorithm used for parallel-friendly device such as GPU.
# Number of pass: T(n) = 1 + 2 + ... + log(n) = O(log(n) ** 2)
# Computation complexity: O(n * log(n) ** 2)
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


def test_bitonic_sort(array_length=32, test_loop=10):
    array = list(range(array_length))
    for _ in range(test_loop):
        copy_array = array.copy()
        random.shuffle(copy_array)
        prev_val = copy_array.copy()
        bitonic_sort(copy_array)
        assert copy_array == array, f"bitonic_sort failed at {prev_val}"
    print("bitonic_sort passed.")


if __name__ == "__main__":
    test_bitonic_sort(test_loop=100)

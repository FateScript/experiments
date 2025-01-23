#!/usr/bin/env python3

# Online softmax algorithm
# refernece paper:
# https://arxiv.org/abs/1805.02867
# https://arxiv.org/abs/2001.04438
# Flash attn also uses this algorithm


import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def online_softmax(x):
    # 1st pass: compute the maximum and exp_sum
    row, col = x.shape
    row_max, exp_sum = np.ones((row, 1)) * float("-inf"), np.zeros((row, 1))
    split_x = np.split(x, col, axis=-1)
    for col_x in split_x:
        real_max = np.maximum(row_max, col_x)
        exp_sum = np.exp(col_x - real_max) + np.exp(row_max - real_max) * exp_sum
        row_max = real_max

    # 2nd pass: compute the softmax
    return np.exp(x - row_max) / exp_sum


def test_online_softmax():
    x = np.random.randn(32, 128)
    y1 = softmax(x)
    y2 = online_softmax(x)
    assert np.allclose(y1, y2)
    print("Passed online softmax test.")


if __name__ == "__main__":
    test_online_softmax()

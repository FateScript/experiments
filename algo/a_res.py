#!/usr/bin/env python3

# see prove of A-res algorithm
# https://claude.ai/chat/f0a41620-bf87-4ec2-a7f5-8146be42234c

import random


def a_res(weights: list[float | int], num_sampels: int = 10) -> list[int]:
    """
    A-res algorithm for sampling from a list of weights.
    """
    samples = []
    for idx, w in enumerate(weights):
        u = random.random()  # uniform random number in [0, 1)
        key = u ** (1 / w)
        samples.append((idx, key))

    sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_samples[:num_sampels]]


if __name__ == "__main__":
    weights = [0.1, 0.5, 0.2, 0.3]
    samples = a_res(weights, num_sampels=2)
    print(samples)

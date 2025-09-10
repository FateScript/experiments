#!/usr/bin/env python3

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as dist

__all__ = ["draw_distribution"]


def plot_distribution(
    data: np.ndarray,
    title: str = "plot",
    dist_type: str = "norm",
):
    plt.figure(figsize=(8, 6))
    plt.hist(
        data, bins='auto', color='blue',
        alpha=0.7, rwidth=0.85, density=True,
    )
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)

    # Generate x values for the curve
    x = np.linspace(np.min(data), np.max(data), 1000)
    fit_dist = getattr(dist, dist_type)
    # Generate y values for the normal distribution curve with mean and std
    param = fit_dist.fit(data)
    y = fit_dist.pdf(x, *param)
    plt.plot(x, y, color='red', linewidth=2)

    plt.legend([f'{dist_type} distribution', 'data distribution'])
    return plt


def draw_distribution(
    data: np.ndarray,
    title: str = "distribution",
    distributions: List[str] = ["norm", "invgauss"],
    display: bool = True
):
    if distributions is None:
        distributions = dir(dist)

    for dist_type in distributions:
        if dist_type.startswith("_"):
            continue
        fit_dist = getattr(dist, dist_type)
        if hasattr(fit_dist, "fit"):
            img = plot_distribution(data, title, dist_type)
            img.savefig(f"{dist_type}_{title}.png")
            if display:
                img.show()


if __name__ == "__main__":
    data = np.random.normal(0, 1, 10000)
    draw_distribution(data)

#!/usr/bin/env python3

# Read the paper: https://www.eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf
# to feel the power of two random choices.
# This code snippet is a simple visualization of the load balance algorithm.

from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def texts_from_data(data, ax) -> List:
    row, column = data.shape
    texts = []
    for i in range(row):
        row_data = []
        for j in range(column):
            text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
            row_data.append(text)
        texts.append(row_data)
    return texts


def random_xy(row, column) -> Tuple[int, int]:
    """Return a random x, y coordinate."""
    return np.random.randint(0, row), np.random.randint(0, column)


def random_select(data):
    """Random select a cell assign the task"""
    row, column = data.shape
    x, y = random_xy(row, column)
    data[x, y] += 1
    return data, x, y


def select_max_two_random(data):
    """Random select two celles and assign the task to the minimum one"""
    row, column = data.shape
    x1, y1 = random_xy(row, column)
    x2, y2 = random_xy(row, column)
    if data[x1, y1] < data[x2, y2]:
        data[x1, y1] += 1
        x, y = x1, y1
    else:
        data[x2, y2] += 1
        x, y = x2, y2
    return data, x, y


def visualize_load_balance(
    row: int = 4, column: int = 5,
    min_load: int = 0, max_load: int = 30,
    save_file: Optional[str] = None,
    save_dpi: int = 80,
):
    random_load = np.zeros((row, column), dtype=np.int32)
    max_two_load = np.zeros((row, column), dtype=np.int32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(random_load, cmap='viridis', vmin=min_load, vmax=max_load)
    im2 = ax2.imshow(max_two_load, cmap='viridis', vmin=min_load, vmax=max_load)

    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(column))
        ax.set_yticks(np.arange(row))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax1.set_title("Random")
    ax2.set_title("Two random choices")
    texts1 = texts_from_data(random_load, ax1)
    texts2 = texts_from_data(max_two_load, ax2)

    plt.tight_layout()

    def update_data(frame_id):
        # update the heatmap data and text annotations
        _, x1, y1 = random_select(random_load)
        _, x2, y2 = select_max_two_random(max_two_load)
        assert random_load.sum() == max_two_load.sum()

        im1.set_array(random_load)
        im2.set_array(max_two_load)

        texts1[x1][y1].set_text(random_load[x1, y1])
        texts2[x2][y2].set_text(max_two_load[x2, y2])

        return im1, im2, texts1, texts2

    ani = animation.FuncAnimation(fig, update_data, frames=500, interval=10)
    if save_file:
        ani.save(save_file, writer="pillow", dpi=save_dpi)
        print(f"Save the animation to {save_file}")
    else:
        plt.show()


if __name__ == "__main__":
    import fire
    fire.Fire(visualize_load_balance)

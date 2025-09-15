
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x, y) -> tuple[float, float]:
    x_b = np.c_[np.ones((x.shape[0], 1)), x]  # add bias term (intercept)
    theta = np.linalg.inv(x_b.T @ x_b) @ (x_b.T @ y)
    intercept, slope = theta
    return intercept, slope


def quartet(plot=True):
    data = {
        'Dataset I': {
            'x': [10.0, 8.0,  13.0, 9.0,  11.0, 14.0, 6.0,  4.0,  12.0,  7.0,  5.0],
            'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        },
        'Dataset II': {
            'x': [10.0, 8.0,  13.0, 9.0,  11.0, 14.0, 6.0,  4.0,  12.0, 7.0,  5.0],
            'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
        },
        'Dataset III': {
            'x': [10.0, 8.0,  13.0,  9.0,  11.0, 14.0, 6.0,  4.0,  12.0, 7.0,  5.0],
            'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
        },
        'Dataset IV': {
            'x': [8.0,  8.0,  8.0, 8.0,  8.0,  8.0,  8.0,  19.0,  8.0,  8.0,  8.0],
            'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
        }
    }

    _, axs = plt.subplots(2, 2, figsize=(12, 10))

    # mean, std, correlation, linear regression are the same for all datasets
    for i, (key, value) in enumerate(data.items()):
        x = np.array(value['x'])
        y = np.array(value['y'])

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        corr_xy = np.corrcoef(x, y)[0, 1]
        intercept, slope = linear_regression(x.reshape(-1, 1), y)

        print(f"\nStatistics for {key}:")
        print(f"Mean of x: {mean_x:.2f}, Mean of y: {mean_y:.2f}")
        print(f"Variance of x: {var_x:.2f}, Variance of y: {var_y:.2f}")
        print(f"Correlation coefficient: {corr_xy:.2f}")
        print(f"Linear regression: y = {slope:.2f}x + {intercept:.2f}")
        ax = axs[i//2, i % 2]
        ax.scatter(x, y, color='blue', label='Data points')
        ax.plot(x, slope * x + intercept, color='red', label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}')
        ax.set_title(key)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.set_xlim(2, 20)
        ax.set_ylim(2, 14)

    if plot:
        plt.tight_layout()
        plt.suptitle("Anscombe's Quartet", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()


if __name__ == "__main__":
    quartet(plot=False)

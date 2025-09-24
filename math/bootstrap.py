#!/usr/bin/env python3
import itertools
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

z_dict = {
    90: 1.645,
    95: 1.960,
    99: 2.576,
}


def bootstrap(
    data: np.ndarray, statistic_fn, n_bootstrap: int = 1000,
    ci: int = 95, percentile: bool = True,
):
    """
    Args:
        data (np.ndarray): 1D numpy array, the original data.
        statistic_fn (Callable): Function to compute the statistic
            (e.g., np.mean, np.median, or a custom function).
        n_bootstrap (int, optional): Number of bootstrap samples to generate. Defaults to 1000.
        ci (float, optional): Confidence interval size (percentage). Defaults to 95.
        percentile (bool, optional): If True, use the percentile method for CI.
    """
    n = len(data)

    stat_bootstrap = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        stat_bootstrap[i] = statistic_fn(sample)

    if percentile:
        alpha = (100 - ci) / 2
        low, high = np.percentile(stat_bootstrap, [alpha, 100 - alpha])
    else:
        se = np.std(stat_bootstrap, ddof=1)
        mean_stat = np.mean(stat_bootstrap)
        z_value = z_dict[ci]
        low, high = mean_stat - z_value * se, mean_stat + z_value * se

    return (float(low), float(high)), stat_bootstrap


def plot_curve(results, ci):
    import matplotlib.pyplot as plt

    bootstrap_sizes = sorted(set(r["Bootstrap Size"] for r in results))
    color_map = {ci_level: color for ci_level, color in zip(ci, plt.cm.tab10.colors)}

    marker_map = {True: 'o', False: 's'}

    for ci_level in ci:
        color = color_map[ci_level]
        for is_percentile in [True, False]:
            x = []
            y = []
            for size in bootstrap_sizes:
                for r in results:
                    if r["Bootstrap Size"] == size and r["CI Level"] == ci_level and r["Percentile"] == is_percentile:
                        x.append(size)
                        y.append(r["Coverage Probability"])
            label = f'CI {ci_level}%, Percentile={is_percentile}'
            plt.plot(x, y, marker=marker_map[is_percentile], color=color, label=label, linestyle='-')

    # Draw horizontal dashed lines for each CI level
    for idx, ci_level in enumerate(ci):
        plt.axhline(y=ci_level / 100, color=color_map[ci_level], linestyle='--', linewidth=1,
                    label=f'CI {ci_level}% Expected' if idx == 0 else None)

    plt.xlabel("Bootstrap Size")
    plt.ylabel("Coverage Probability")
    plt.title("Bootstrap CI Coverage Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    mean, std = 1, 2
    num_runs = 1000
    bootstrap_size = [5, 10, 20, 30]
    ci = [90, 95, 99]
    percentile = [True, False]

    results = []

    for size, ci_level, is_percentile in itertools.product(bootstrap_size, ci, percentile):
        match_count = 0
        for _ in tqdm(range(num_runs), leave=False):
            data = np.random.normal(mean, mean, size=size)
            # data = np.random.uniform(mean - std, mean + std, size=size)
            (low, high), _ = bootstrap(data, np.mean, ci=ci_level, percentile=is_percentile)
            if low <= mean <= high:
                match_count += 1
        coverage = match_count / num_runs
        print(f"Bootstrap size: {size}, CI level: {ci_level}, Percentile: {is_percentile}, Prob: {coverage:.2%}")
        results.append({
            "Bootstrap Size": size,
            "CI Level": ci_level,
            "Percentile": is_percentile,
            "Coverage Probability": coverage,
        })
    print(tabulate(results, headers="keys", tablefmt="github", showindex=False))
    plot_curve(results, ci)

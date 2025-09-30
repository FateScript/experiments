import numpy as np
import matplotlib.pyplot as plt


def round_half_up(n):
    if n - np.floor(n) == 0.5:
        return int(np.floor(n)) + 1
    else:
        return int(round(n))


def bankers_round(n):
    if n - np.floor(n) == 0.5:
        if int(np.floor(n)) % 2 == 0:
            return int(np.floor(n))
        else:
            return int(np.floor(n)) + 1
    else:
        return int(round(n))


def stochastic_round(x, x1, x2):
    """
    Perform stochastic rounding to x1 or x2 with probabilities proportional to the distance
    from x to each number.
    """
    assert x1 < x2, "x1 should be less than x2"
    p_up = (x2 - x) / (x2 - x1)
    # p_down = (x - x1) / (x2 - x1)

    # Randomly choose based on the probabilities
    return x1 if np.random.rand() < p_up else x2


def stochastic_rounding_method1(n):
    n1, n2 = np.floor(n), np.ceil(n)
    return n1 if np.random.rand() < 0.5 else n2


def stochastic_rounding_method2(n):
    n1, n2 = np.floor(n), np.ceil(n)
    return stochastic_round(n, n1, n2)


def generate_data_with_half(max_data: int = 10, ratio: float = 0.1, num_points: int = 1000):
    """Generate random data with a certain ratio of x.5 values.

    Args:
        max_data (int): Maximum value of the data.
        ratio (float): Ratio of x.5 values in the data.
    """
    data = np.random.uniform(0, max_data, num_points)
    # set part of data to x.5
    for i in range(len(data)):
        if np.random.rand() < ratio:
            data[i] = np.floor(data[i]) + 0.5
    return data


def round_analysis(num_points: int = 50000):
    max_data = 10
    ratios = [0, 0.5, 1.0]
    methods = [
        ("Round half up", round_half_up),
        ("Bankers round", bankers_round),
        ("Stochastic round 1", stochastic_rounding_method1),
        ("Stochastic round 2", stochastic_rounding_method2)
    ]

    fig, axs = plt.subplots(len(ratios), len(methods), figsize=(12, 8))
    bins = np.arange(0, max_data + 2, 1) - 0.5

    for i, ratio in enumerate(ratios):
        data = generate_data_with_half(max_data, ratio, num_points=num_points)
        data_mean = np.mean(data)
        print("---" * 20)
        print(f"Data with ratio {ratio} of x.5 values, mean: {data_mean}\n")
        for j, (method_name, method_func) in enumerate(methods):  # Only show Bankers and Half Up
            rounded_data = np.array([method_func(x) for x in data])
            round_mean = np.mean(rounded_data)
            shift_percent = (round_mean - data_mean) / data_mean * 100 if data_mean != 0 else 0
            print(
                f"Data with ratio {ratio} of x.5 values, "
                f"{method_name} mean: {round_mean}, shift: {shift_percent:.2f}%\n"
            )
            hist_data, _ = np.histogram(data, bins=bins)
            hist_rounded, _ = np.histogram(rounded_data, bins=bins)
            ax = axs[i, j]
            ax.bar(bins[:-1], hist_data, width=1, alpha=0.3, label="Original Data", color='blue')
            ax.bar(bins[:-1], hist_rounded, width=1, alpha=0.3, label="Rounded Data", color='green')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            if j == 0:
                ax.set_title(f'ratio={ratio}, {method_name}')
            else:
                ax.set_title(method_name)
            ax.legend()

    # Hide unused subplot (axs[0,2] and axs[1,2] if only 2 methods)
    for i in range(len(ratios)):
        for j in range(len(methods), len(methods)):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    round_analysis(num_points=50000)

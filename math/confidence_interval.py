import numpy as np

z_dict = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def interval_by_sample(data, confidence: float):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z_value = z_dict[confidence]
    interval = (mean - z_value * std / np.sqrt(n), mean + z_value * std / np.sqrt(n))
    return interval


def interval_prob_check(confidence: float = 0.90, verbose: bool = False):
    mu, sigma = 1, 5
    n = 1000  # num samples
    interval_samples = 50000

    match_count = 0
    for _ in range(interval_samples):
        data = np.random.normal(loc=mu, scale=sigma, size=n)
        ci_lower, ci_upper = interval_by_sample(data, confidence=confidence)
        if ci_lower <= mu <= ci_upper:
            match_count += 1
        elif verbose:
            print(f"CI does not contain true mean: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Confidence level: {confidence}, Prob: {match_count / interval_samples:.2%}")


if __name__ == "__main__":
    for prob in z_dict.keys():
        interval_prob_check(confidence=prob, verbose=False)

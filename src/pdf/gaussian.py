import numpy as np


class GaussianDistribution:
    def __init__(self, sample: np.ndarray[np.float64]) -> None:
        self.sample = sample
        self.mean = np.mean(sample)
        self.std = np.std(sample)
        self.n = len(sample)

    @property
    def sample_list(self) -> list[np.float64]:
        return self.sample.tolist()

    def pick_rand(self) -> np.float64:
        return np.random.choice(self.sample)

    def pick_avg(self) -> np.float64:
        return self.mean

    def pick_max(self) -> np.float64:
        return np.max(self.sample)

    def pick_min(self) -> np.float64:
        return np.min(self.sample)

    def pick_top5(self) -> np.float64:
        pt = np.percentile(self.sample, 95)
        after_five_percent = self.sample[self.sample > pt]

        return np.random.choice(after_five_percent)

    def pick_bottom5(self) -> np.float64:
        pt = np.percentile(self.sample, 5)
        before_five_percent = self.sample[self.sample < pt]

        return np.random.choice(before_five_percent)

    def confidence_range(self) -> tuple[np.float64, np.float64]:
        z = 1.96
        bias = z * self.std / np.sqrt(self.n)
        return self.mean - bias, self.mean + bias

    def replace_sample(self, sample: np.ndarray[np.float64]) -> None:
        self.sample = sample
        self.mean = np.mean(sample)
        self.std = np.std(sample)
        self.n = len(sample)

from abc import abstractmethod

import numpy as np


class CrossoverBehavior:
    """
    CrossoverBehavior class
    """

    def __init__(self, method_name: str):
        self.method_name = method_name

    @abstractmethod
    def crossover(
        self, first: np.ndarray[np.float64], second: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        raise NotImplementedError

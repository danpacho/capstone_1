from random import choice
import numpy as np

from src.ga.p4_crossover.crossover_behavior import CrossoverBehavior


class OnePointCrossover(CrossoverBehavior):
    """
    OnePoint Crossover

    Example:
    ```python
    first = [1, 2, 3, 4, 5]
    second = [6, 7, 8, 9, 10]
    one_point_crossover = OnePointCrossover()
    # pick one point randomly -> 3
    one_point_crossover.crossover(first, second)
    Output = [1, 2, 3, 9, 10]
    ```
    """

    def __init__(self):
        super().__init__("OnePoint Crossover")

    def crossover(
        self, first: np.ndarray[np.float64], second: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        choice_index = choice(range(len(first)))
        one_point_cross = np.concatenate((first[:choice_index], second[choice_index:]))
        return one_point_cross


class TwoPointCrossover(CrossoverBehavior):
    """
    TwoPoint Crossover

    Example:
    ```python
    first = [1, 2, 3, 4, 5]
    second = [6, 7, 8, 9, 10]
    two_point_crossover = TwoPointCrossover()
    # pick two points randomly -> 2, 3
    two_point_crossover.crossover(first, second)
    Output = [1, 2, 8, 9, 5]
    ```
    """

    def __init__(self):
        super().__init__("TwoPoint Crossover")

    def crossover(
        self, first: np.ndarray[np.float64], second: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        choice_index_1 = choice(range(len(first)))
        choice_index_2 = choice(range(len(first)))

        if choice_index_1 > choice_index_2:
            choice_index_1, choice_index_2 = choice_index_2, choice_index_1
        elif choice_index_1 == choice_index_2:
            choice_index_2 += 1

        two_point_cross = np.concatenate(
            (
                first[:choice_index_1],
                second[choice_index_1:choice_index_2],
                first[choice_index_2:],
            )
        )
        return two_point_cross


class UniformCrossover(CrossoverBehavior):
    """
    Uniform Crossover

    Example:
    ```python
    first = [1, 2, 3, 4, 5]
    second = [6, 7, 8, 9, 10]
    uniform_crossover = UniformCrossover()
    uniform_crossover.crossover(first, second)
    Output = [1, 7, 3, 9, 5]
    ```
    """

    def __init__(self):
        super().__init__("Uniform Crossover")

    def crossover(
        self, first: np.ndarray[np.float64], second: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        uniform_cross = np.array(
            [
                first[i] if choice([True, False]) else second[i]
                for i in range(len(first))
            ]
        )
        return uniform_cross


one_point_crossover = OnePointCrossover()
two_point_crossover = TwoPointCrossover()
uniform_crossover = UniformCrossover()

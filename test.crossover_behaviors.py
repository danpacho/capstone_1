import numpy as np

from src.ga.p4_crossover.behaviors import (
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
)

# Test case 1
first = np.array([1, 2, 3, 4, 5])
second = np.array([6, 7, 8, 9, 10])
expected_output = np.array([1, 2, 8, 9, 5])
output = two_point_crossover.crossover(first, second)
print(output)

# Test case 2
first = np.array([11, 12, 13, 14, 15])
second = np.array([16, 17, 18, 19, 20])
expected_output = np.array([11, 12, 18, 19, 15])
output = one_point_crossover.crossover(first, second)
print(output)

# Test case 3
first = np.array([21, 22, 23, 24, 25])
second = np.array([26, 27, 28, 29, 30])
expected_output = np.array([21, 22, 28, 29, 30])
output = uniform_crossover.crossover(first, second)
print(output)

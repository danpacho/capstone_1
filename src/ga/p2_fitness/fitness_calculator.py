from abc import abstractmethod
from typing import Generic, TypeVar

from src.ga.chromosome.chromosome import Chromosome


ChromosomeType = TypeVar("ChromosomeType", bound=Chromosome)


class FitnessCalculator(Generic[ChromosomeType]):
    """
    FitnessCalculator class
    """

    def __init__(
        self,
        fitness_method_name: str,
        min_criteria_value_list: list[float],
        effective_criteria_list: list[float],
        criteria_label_list: list[str],
        criteria_weight_list: list[float],
    ):
        """
        Args:
            fitness_method_name (`str`): Name of the fitness method
            min_criteria_value_list (`list[float]`): List of minimum criteria values
            effective_criteria_list (`list[float]`): List of effective criteria values
            criteria_label_list (`list[str]`): List of criteria labels
            criteria_weight_list (`list[float]`): List of criteria weights
        """
        self.fitness_method_name = fitness_method_name
        self.criteria_label_list = criteria_label_list
        self.min_criteria_value_list = min_criteria_value_list
        self.effective_criteria_list = effective_criteria_list
        self.criteria_weight_list = criteria_weight_list

    @abstractmethod
    def calculate(self, chromosome: ChromosomeType) -> list[float]:
        """
        Calculate the fitness value of the chromosome

        Args:
            chromosome (`ChromosomeType`): Chromosome to calculate the fitness value
        """
        raise NotImplementedError

    def judge_fitness(self, chromosome: ChromosomeType) -> tuple[bool, float]:
        """
        Judge the fitness of the chromosome

        Args:
            chromosome (`ChromosomeType`): Chromosome to judge the criteria values
        Returns:
            `tuple[bool, float, float]`: (Fitness success flag and Fitness score and Biased fitness score)
        Example:
        ```python
        success, fitness, biased_fitness = self.judge_criteria(chromosome)

        if success:
            # Chromosome is valid
            # we can use the fitness to select the chromosome
        else:
            # Chromosome is invalid
        ```
        """
        success = True
        fitness = 0.0
        biased_fitness = 0.0

        criteria_values = self.calculate(chromosome)
        for i, criteria_value in enumerate(criteria_values):
            if criteria_value < self.min_criteria_value_list[i]:
                success = False

            biased_fitness += self.criteria_weight_list[i] * criteria_value
            fitness += criteria_value

        return (success, fitness, biased_fitness)

    @property
    def criteria_table(self):
        """
        Returns a table containing the criteria labels, minimum, effective, weights.

        Example:
        ```python
        {
            "mass_flow_rate": {
                "min_value": 0.5,
                "effective_value": 1.0,
                "weight": 0.5,
            },
            "drag": {
                "min_value": 0.75,
                "effective_value": 1.0,
                "weight": 0.25,
            },
            ...
        }
        ```
        """
        table: dict[str, dict[str, float]] = {}
        for i, label in enumerate(self.criteria_label_list):
            table[label] = {
                "min_value": self.min_criteria_value_list[i],
                "effective_value": self.effective_criteria_list[i],
                "weight": self.criteria_weight_list[i],
            }

        return table

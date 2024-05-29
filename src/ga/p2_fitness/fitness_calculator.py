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
        criteria_label_list: list[str],
        criteria_weight_list: list[float],
    ):
        """
        Args:
            fitness_method_name (`str`): Name of the fitness method
            criteria_label_list (`list[str]`): List of criteria labels
            criteria_weight_list (`list[float]`): List of criteria weights
        """
        self.fitness_method_name = fitness_method_name
        self.criteria_label_list = criteria_label_list
        for weight in criteria_weight_list:
            if weight <= 0:
                raise ValueError("Criteria weight should be greater than or equal to 0")

        self.criteria_weight_list = criteria_weight_list

    @abstractmethod
    def calculate(self, chromosome: ChromosomeType) -> list[float]:
        """
        Calculate the fitness value of the chromosome

        Args:
            chromosome (`ChromosomeType`): Chromosome to calculate the fitness value
        """
        raise NotImplementedError

    @abstractmethod
    def judge_fitness(
        self, chromosome: ChromosomeType
    ) -> tuple[bool, float, float, tuple]:
        """
        Judge the fitness of the chromosome

        Args:
            chromosome (`ChromosomeType`): Chromosome to judge the criteria values
        Returns:
            `tuple[bool, float, float, tuple]`: (Fitness success flag, Fitness score, Biased fitness score, Criteria values tuple)
        Example:
        ```python
        success, fitness, biased_fitness, result_pure = self.judge_criteria(chromosome)

        if success:
            # Chromosome is valid
            # we can use the fitness to select the chromosome
        else:
            # Chromosome is invalid
        ```
        """
        raise NotImplementedError

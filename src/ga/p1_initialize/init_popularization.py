from abc import abstractmethod
from typing import Generic, TypeVar

from src.ga.chromosome.chromosome import Chromosome

ChromosomeType = TypeVar("ChromosomeType", bound=Chromosome)


class PopularizationInitializer(Generic[ChromosomeType]):
    """
    PopularizationInitializer class
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

    @abstractmethod
    def initialize(self) -> list[ChromosomeType]:
        """
        Initialize the population
        """
        raise NotImplementedError

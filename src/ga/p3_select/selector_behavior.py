from abc import abstractmethod
from typing import Generic, TypeVar

from src.ga.chromosome.chromosome import Chromosome
from src.storage.population_storage import PopulationStorage

ChromosomeType = TypeVar("ChromosomeType", bound=Chromosome)


class SelectionBehavior(Generic[ChromosomeType]):
    """
    SelectionBehavior class
    """

    def __init__(self, selection_strategy_name: str):
        self.selection_strategy_name = selection_strategy_name

    @abstractmethod
    def select(
        self, population: list[ChromosomeType], population_info: PopulationStorage
    ) -> list[ChromosomeType]:
        """
        Select the chromosomes from the population information.
        """
        raise NotImplementedError

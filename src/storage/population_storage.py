from typing import Literal, Union

import numpy as np

from src.ga.chromosome.chromosome import Chromosome
from src.storage.stochastic_storage import StochasticStorage
from src.storage.storage_unit import Storage


class PopulationStorage:
    """
    PopulationStorage class

    Type1. population storage:
    ```__population_storage.json
    {
        "Chromosome1": [0.1, 0.2],
        "Chromosome2": [0.1, 0.2],
        ...
    }
    ```
    Type2. population fitness storage:
    ```__population_fitness_storage.json
    {
        "fitness": [0.1, 0.2, 0.3, ...],
        "biased_fitness": [0.1, 0.2, 0.3, ...],
        ...
    }
    ```
    """

    def __init__(self, root_filename_tuple: tuple[str, str]):
        """
        Args:
            root_filename: root filenames for (`population`, `population fitness`) storage.
        """
        self.root_filename_tuple = root_filename_tuple
        self.population_storage: Storage[list[float]] = Storage(
            root_filename_tuple[0], "PopulationStorage"
        )

        self.population_fitness_storage = StochasticStorage(root_filename_tuple[1])
        self.population_fitness_storage.label = "PopulationFitnessStorage"

    def reset(self) -> None:
        """
        Reset the population storage.

        When next generation starts, the population storage should be reset.
        """
        self.population_storage.reset()
        self.population_fitness_storage.reset()

    def save(self) -> None:
        """
        Save the population storage.

        When the fitness of the population is updated, the storage should be saved.
        """
        self.population_storage.save()
        self.population_fitness_storage.save()

    def insert_chromosome(
        self,
        chromosome: Chromosome,
        fitness: tuple[float, float],
    ) -> None:
        """
        Insert a single chromosome to the population storages
        (PopulationStorage & FitnessStorage).

        Args:
            generation (`int`): Generation number
            chromosome (`Chromosome`): Chromosome to insert
            fitness (`tuple[float, float]`): Fitness and Biased fitness of the chromosome
        """
        self.population_storage.insert_field(chromosome.label, list(fitness))
        # fitness: 0, biased_fitness: 1
        self.population_fitness_storage.insert_single(
            self.fitness_storage_keys[0], fitness[0]
        )
        self.population_fitness_storage.insert_single(
            self.fitness_storage_keys[1], fitness[1]
        )

    def inquire_chromosome_fitness(
        self, chromosome: Chromosome
    ) -> Union[tuple[float, float], None]:
        """
        Inquire the fitness of the chromosome.

        Args:
            chromosome (`Chromosome`): Chromosome to inquire the fitness

        Returns:
            `tuple[float, float]`: Fitness and Biased fitness of the chromosome
        """
        fitness = self.population_storage.inquire(chromosome.label)
        if fitness is None:
            return None

        return fitness[0], fitness[1]

    @property
    def population_size(self) -> int:
        """
        Return the population size
        """
        return self.population_storage.size

    @property
    def population_labels(self) -> list[str]:
        """
        Return the population labels
        """
        return self.population_storage.keys

    @property
    def fitness_storage_keys(
        self,
    ) -> tuple[Literal["fitness"], Literal["biased_fitness"]]:
        """
        Return the keys of the fitness storage
        """
        return "fitness", "biased_fitness"

    @property
    def fitness_distribution(self) -> np.ndarray[np.float64]:
        """
        Return the fitness distribution of the population
        """
        inquired = self.population_fitness_storage.inquire("fitness")
        return np.array(inquired) if inquired is not None else np.array([])

    @property
    def biased_fitness_distribution(self) -> np.ndarray[np.float64]:
        """
        Return the biased fitness distribution of the population
        """
        inquired = self.population_fitness_storage.inquire("biased_fitness")
        return np.array(inquired) if inquired is not None else np.array([])

    @staticmethod
    def _search_indexof(arr: np.ndarray[np.float64], target: float) -> int:
        index = arr.searchsorted(target)
        if index < len(arr) and arr[index] == target:
            return index
        return -1

    def get_fitness_percentile(
        self,
        fitness: float,
        search_type: Union[Literal["fitness", "biased_fitness"]] = "fitness",
    ) -> float:
        """
        Returns the percentile of the given fitness value in the population.

        Args:
            fitness (`float`): Fitness value to search
            search_type (`str`): Type of the fitness to search
        Returns:
            `tuple[int, bool]`: (Percentile of the fitness, Whether the fitness is in the population)
        """
        searched = PopulationStorage._search_indexof(
            (
                self.fitness_distribution
                if search_type == "fitness"
                else self.biased_fitness_distribution
            ),
            fitness,
        )

        if searched == -1:
            return 0.0

        return searched / self.population_size * 100

from typing import Literal, Union

import numpy as np

from src.ga.chromosome.chromosome import Chromosome
from src.ga.p3_select.selector_behavior import SelectionBehavior
from src.storage.population_storage import PopulationStorage


class ElitismSelectionFilter(SelectionBehavior[Chromosome]):
    """
    Elitism selection filter

    Args:
        elitism_criterion (`float`): Elitism criterion to select the chromosomes.

        `0 ~ 1`, if closer to `1`, more elitism
    """

    def __init__(
        self,
        elitism_criterion: float = 0.75,
    ):
        super().__init__("ElitismSelection")
        self.elitism_criterion = elitism_criterion

    def select(self, population, population_info):
        selected_population: list[Chromosome] = []
        for chromosome in population:
            fitness, biased_fitness = population_info.inquire_chromosome(chromosome)
            if fitness is None:
                continue

            fitness_percentile = population_info.get_fitness_percentile(
                fitness, "fitness"
            )
            if fitness_percentile >= self.elitism_criterion:
                selected_population.append(chromosome)
                continue

            biased_fitness_percentile = population_info.get_fitness_percentile(
                biased_fitness, "biased_fitness"
            )
            if biased_fitness_percentile >= self.elitism_criterion:
                selected_population.append(chromosome)

        return selected_population


class RouletteWheelSelectionFilter(SelectionBehavior[Chromosome]):
    """
    Roulette wheel selection filter

    Args:
        pointer_count (`int`): Number of pointers to select the chromosomes
        roulette_type (`Literal["random", "universal_stochastic"]`): Type of roulette wheel selection
        running_count (`int`): Number of roulette selections to run
    """

    def __init__(
        self,
        roulette_pointer_count: int = 1,
        roulette_type: Union[
            Literal["random", "universal_stochastic"]
        ] = "universal_stochastic",
        running_count: Union[int, None] = None,
    ):
        super().__init__("RouletteWheelSelection")
        self.roulette_type = roulette_type
        self.pointer_count = roulette_pointer_count
        self.roulette_count = running_count

        self._is_roulette_initialized = False
        self._fitness_roulette_wheel: np.ndarray[np.float64] = np.array([])

    def generate_roulette_wheel(self, population_info: PopulationStorage):
        """
        Generate the roulette wheel for the fitness and biased fitness distribution
        """
        fitness_sum = np.sum(population_info.fitness_distribution)
        cumulative_fitness = 0.0
        for fitness in population_info.fitness_distribution:
            cumulative_fitness += fitness / fitness_sum
            self._fitness_roulette_wheel = np.append(
                self._fitness_roulette_wheel, cumulative_fitness
            )

        self._is_roulette_initialized = True

    def select(self, population, population_info):
        if not self._is_roulette_initialized:
            self.generate_roulette_wheel(population_info)

        selected_population: list[Chromosome] = []

        count = (
            self.roulette_count
            if self.roulette_count is not None
            else population_info.population_size // self.pointer_count
        )
        for _ in range(count):
            total_fitness = np.sum(population_info.fitness_distribution)

            if self.roulette_type == "random":
                for _ in range(self.pointer_count):
                    pointer = np.random.uniform(0, total_fitness)
                    index = self._select_chromosome(
                        pointer, self._fitness_roulette_wheel
                    )
                    selected_population.append(population[index])

            else:
                pointer_distance: float = total_fitness / self.pointer_count
                start_point = np.random.uniform(0, pointer_distance)
                pointers: list[float] = [
                    start_point + i * pointer_distance
                    for i in range(self.pointer_count)
                ]
                for pointer in pointers:
                    index = self._select_chromosome(
                        pointer, self._fitness_roulette_wheel
                    )
                    selected_population.append(population[index])

        return selected_population

    def _select_chromosome(
        self, pointer: float, roulette_wheel: np.ndarray[np.float64]
    ) -> int:
        differences = np.abs(roulette_wheel - pointer)
        index = np.argmin(differences)
        return int(index)


class TournamentSelectionFilter(SelectionBehavior[Chromosome]):
    """
    Tournament selection filter

    Args:
        tournament_size (`int`): Size of the tournament
        running_count (`int`): Number of tournaments to run
    """

    def __init__(
        self, tournament_size: int = 2, running_count: Union[int, None] = None
    ):
        super().__init__("TournamentSelection")
        self.tournament_size = tournament_size
        self.tournament_count = running_count

    def select(self, population, population_info):
        selected_population: list[Chromosome] = []
        count = (
            self.tournament_count
            if self.tournament_count is not None
            else population_info.population_size
        )

        for _ in range(count):
            tournament_chromosomes: list[Chromosome] = []
            for _ in range(self.tournament_size):
                rand_index = np.random.randint(0, population_info.population_size)
                tournament_chromosomes.append(population[rand_index])

            best_chromosome = self._select_best_chromosome(
                tournament_chromosomes, population_info
            )
            selected_population.append(best_chromosome)

        return selected_population

    def _select_best_chromosome(
        self, tournament_chromosomes: list[Chromosome], population_info
    ) -> Chromosome:
        best_chromosome = None
        best_fitness = -np.inf
        for chromosome in tournament_chromosomes:
            fitness, _ = population_info.inquire_chromosome(chromosome)
            if fitness is None:
                continue

            if fitness > best_fitness:
                best_chromosome = chromosome
                best_fitness = fitness

        return best_chromosome

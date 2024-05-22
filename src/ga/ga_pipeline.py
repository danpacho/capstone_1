from random import random, shuffle
from typing import Callable, Generic, TypeVar

from src.ga.gene.gene import Gene
from src.ga.chromosome.chromosome import Chromosome
from src.ga.p1_initialize.init_popularization import PopularizationInitializer
from src.ga.p2_fitness.fitness_calculator import FitnessCalculator
from src.ga.p3_select.selector_behavior import SelectionBehavior
from src.ga.p4_crossover.crossover_behavior import CrossoverBehavior

from src.storage.storage_unit import Storage
from src.storage.population_storage import PopulationStorage
from src.storage.evolution_storage import EvolutionStorage

ChromosomeType = TypeVar("ChromosomeType", bound=Chromosome[tuple[Gene, ...]])


class GAPipeline(Generic[ChromosomeType]):
    """
    Genetic Algorithm Pipeline

    Phase 1: Initialize
    Phase 2: Fitness Calculation
    Phase 3: Selection
    Phase 4: Crossover
    Phase 5: Mutation
    """

    def __init__(
        self,
        suite_name: str,
        suite_max_count: int,
        population_initializer: PopularizationInitializer[ChromosomeType],
        fitness_calculator: FitnessCalculator[ChromosomeType],
        selector_behavior: SelectionBehavior[ChromosomeType],
        crossover_behavior: CrossoverBehavior,
        mutation_probability: float,
        immediate_exit_condition: Callable[[tuple[float, float]], bool],
    ) -> None:
        """
        Args:
            suite_name (`str`): Name of the G.A suite
            suite_max_count (`int`): Maximum count of the suite
            population_initializer: Population initializer for the genetic algorithm
            fitness_calculator: Fitness calculator for the genetic algorithm
            selector_behavior: Selector behavior for the genetic algorithm
            crossover_behavior: Crossover behavior for the genetic algorithm
            mutation_probability (`float`): Mutation probability, in the range of [0, 1], if 0.5% then 0.005
            immediate_exit_condition: Immediate exit condition, (fitness, biased_fitness) -> bool
        """
        self.suite_name = suite_name
        if suite_max_count <= 0:
            raise ValueError("The suite_max_count should be greater than 0")
        self.suite_max_count = suite_max_count
        if mutation_probability < 0 or mutation_probability > 1:
            raise ValueError(
                "The mutation_probability should be in the range of [0, 1]"
            )
        self.mutation_probability = mutation_probability

        self.population_initializer = population_initializer
        self.fitness_calculator = fitness_calculator
        self.selector_behavior = selector_behavior
        self.crossover_behavior = crossover_behavior

        self.immediate_exit_condition = immediate_exit_condition

        self.evolution_storage = EvolutionStorage(
            (
                f"__{suite_name}.fitness_evolution",
                f"__{suite_name}.biased_fitness_evolution",
            )
        )
        self.population_storage = PopulationStorage(
            (
                f"__{suite_name}.population",
                f"__{suite_name}.fitness",
            )
        )
        self.parameter_storage: Storage[list[float]] = Storage(
            f"__{suite_name}.gene_parameters"
        )
        Gene.parameter_storage = self.parameter_storage

        self._population: list[ChromosomeType] = []
        self._generation: int = 0
        self._should_stop: bool = False

    @property
    def population_count(self) -> int:
        """
        Get the population count
        """
        return len(self._population)

    @property
    def generation(self) -> int:
        """
        Get the generation count
        """
        return self._generation

    def _pre_initialize(self):
        self.evolution_storage.reset()
        self.population_storage.reset()
        self.parameter_storage.reset()

        self._generation = 0
        self._population: list[ChromosomeType] = (
            self.population_initializer.initialize()
        )

    def _post_exit(self):
        self.population_storage.reset()
        self.parameter_storage.reset()
        # Do not reset evolution storage for analysis

    def _fitness_calculation(self):
        for chromosome in self._population:
            success, fitness, biased_fitness = self.fitness_calculator.judge_fitness(
                chromosome
            )
            if success:
                if self.immediate_exit_condition((fitness, biased_fitness)):
                    self._should_stop = True
                    break
                self.population_storage.insert_chromosome(
                    chromosome, (fitness, biased_fitness)
                )

        self.population_storage.save()

    def _crossover_popularization(self):
        child_population: list[ChromosomeType] = []

        index_list = list(range(self.population_count))
        shuffle(index_list)

        for i, random_index in enumerate(index_list):
            parent1: ChromosomeType = self._population[random_index]
            parent2: ChromosomeType = self._population[
                index_list[(i + 1) % self.population_count]
            ]

            children: ChromosomeType = parent1.crossover(
                behavior=self.crossover_behavior,
                other=parent2,
            )

            child_population.append(children)

        self._population = child_population

    def _mutate_popularization(self):
        for chromosome in self._population:
            random_value = random()
            if random_value < self.mutation_probability:
                chromosome.mutate_genes()

    def _generation_runner(self):
        # Phase 2: Fitness Calculation
        self._fitness_calculation()

        if self._should_stop:
            return

        # Phase 3: Selection
        self._population = self.selector_behavior.select(
            self._population, self.population_storage
        )

        # Phase 4: Crossover
        self._crossover_popularization()

        # Phase 5: Mutation
        self._mutate_popularization()

        self._generation += 1

    def run(self):
        """
        Run the genetic algorithm pipeline
        """
        self._pre_initialize()

        while (
            self._generation <= self.suite_max_count
            or self._should_stop
            or self.population_count <= 1
        ):
            self._generation_runner()

        self._post_exit()

from random import random, shuffle
from typing import Callable, Generic, TypeVar, Union

from src.prediction.model_trainer import ModelTrainer
from src.ga.gene.gene import Gene
from src.ga.chromosome.chromosome import Chromosome
from src.ga.p1_initialize.init_popularization import PopularizationInitializer
from src.ga.p2_fitness.fitness_calculator import FitnessCalculator
from src.ga.p3_select.selector_behavior import SelectionBehavior
from src.ga.p4_crossover.crossover_behavior import CrossoverBehavior

from src.storage.population_storage import PopulationStorage
from src.storage.evolution_storage import EvolutionStorage

ChromosomeType = TypeVar("ChromosomeType", bound=Chromosome[tuple[Gene, ...]])


class GAPipeline(Generic[ChromosomeType]):
    """
    Genetic Algorithm Pipeline

    Phase 1: Initialize Population
    Phase 2: Fitness Calculation
    Phase 3: Selection
    Phase 4: Crossover
    Phase 5: Mutation
    Phase 6: Fitness Calculation for children (optional)
    """

    def __init__(
        self,
        suite_name: str,
        suite_max_count: int,
        suite_min_population: int,
        suite_min_chromosome: int,
        population_initializer: PopularizationInitializer[ChromosomeType],
        fitness_calculator: FitnessCalculator[ChromosomeType, tuple[ModelTrainer, ...]],
        selector_behavior: SelectionBehavior[ChromosomeType],
        crossover_behavior: CrossoverBehavior,
        mutation_probability: float,
        immediate_exit_condition: Callable[[tuple[float, float]], bool],
        check_children_fitness: bool = True,
        use_cache: bool = True,
    ) -> None:
        """
        Args:
            suite_name (`str`): Name of the G.A suite
            suite_max_count (`int`): Maximum count of the suite
            suite_min_population (`int`): Minimum population count
            suite_min_chromosome (`int`): Minimum chromosome count(deduplicated chromosomes)
            population_initializer: Population initializer for the genetic algorithm
            fitness_calculator: Fitness calculator for the genetic algorithm
            selector_behavior: Selector behavior for the genetic algorithm
            crossover_behavior: Crossover behavior for the genetic algorithm
            mutation_probability (`float`): Mutation probability, in the range of `[0, 1]`, if `0.5%` then `0.005`
            immediate_exit_condition: Immediate exit condition, (fitness, biased_fitness) -> bool
            check_children_fitness (`bool`): Check the children fitness
            use_cache (`bool`): Use cache for the genetic algorithm
        """
        self.suite_name = suite_name
        self.check_children_fitness = check_children_fitness
        self.use_cache = use_cache

        self.suite_min_chromosome = suite_min_chromosome

        if suite_max_count <= 0:
            raise ValueError("The suite_max_count should be greater than 0")
        self.suite_max_count = suite_max_count

        if suite_min_population <= 0:
            raise ValueError("The suite_min_population should be greater than 0")
        if suite_min_population >= population_initializer.population_size:
            raise ValueError(
                "The suite_min_population should be less than the population size"
            )
        self.suite_min_population = suite_min_population

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
                f"store/{suite_name}/fitness_evolution",
                f"store/{suite_name}/biased_fitness_evolution",
            )
        )
        self.population_storage = PopulationStorage(
            (
                f"store/{suite_name}/population",
                f"store/{suite_name}/fitness",
            )
        )

        self._population: list[ChromosomeType] = []
        self._generation: int = 0
        self._should_stop: bool = False
        self._mutation_count: int = 0
        self._gen_calc_result: dict[str, tuple[bool, float, float, tuple]] = {}

        self._reset()

    @property
    def suite_info(self) -> dict[str, str]:
        """
        Get the suite information
        """
        return {
            "suite_name": self.suite_name,
            "generation": self._generation,
            "population_count": self.population_count,
            "mutation_count": self._mutation_count,
            "immediate_exit": self._should_stop,
            "initial_population": self.population_initializer.population_size,
            "unique_population_count": self.unique_population_count,
            "suite_max_count": self.suite_max_count,
            "suite_min_population": self.suite_min_population,
            "suite_min_chromosome": self.suite_min_chromosome,
            "use_cache": str(self.use_cache),
            "check_children_fitness": str(self.check_children_fitness),
            "mutation_probability": self.mutation_probability,
            "fitness_calculator": self.fitness_calculator.fitness_method_name,
            "selector_behavior": self.selector_behavior.selection_strategy_name,
            "crossover_behavior": self.crossover_behavior.method_name,
            "population_initializer": self.population_initializer.__class__.__name__,
        }

    @property
    def population(self) -> list[ChromosomeType]:
        """
        Get the population
        """
        return self._population

    @property
    def population_count(self) -> int:
        """
        Get the population count
        """
        return len(self._population)

    @property
    def unique_population(self) -> Union[list[ChromosomeType], None]:
        """
        Get the unique population by gene label deduplication
        """
        if len(self._population) == 0:
            return None

        unique_chromosomes: list[ChromosomeType] = []
        chromosome_ids: set[str] = set()

        for chromosome in self._population:
            gene_id = ""
            for gene in chromosome.gene_tuple:
                gene_id += gene.label

            if gene_id not in chromosome_ids:
                chromosome_ids.add(gene_id)
                unique_chromosomes.append(chromosome)

        return unique_chromosomes

    @property
    def unique_population_count(self) -> int:
        """
        Get the unique population count
        """
        return len(self.unique_population) if self.unique_population is not None else 0

    @property
    def generation(self) -> int:
        """
        Get the generation count
        """
        return self._generation

    def _reset(self) -> None:
        """
        Reset the pipeline
        """
        self.evolution_storage.reset()
        self.population_storage.reset()
        Gene.parameter_storage.reset()

        self._generation = 0
        self._population = []

    def _pre_initialize(self):
        """
        Pre initialize operations
        """
        self._reset()
        self._population: list[ChromosomeType] = (
            self.population_initializer.initialize()
        )

    def _post_exit(self):
        """
        Post exit operations
        """
        final_result: list[tuple[float, float]] = []
        for chromosome in self._population:
            result = self.population_storage.inquire_chromosome_fitness(chromosome)
            if result is None:
                print(
                    f"Error: chromosome {chromosome.label} not found in the population"
                )
                continue
            final_result.append(result)

        self.population_storage.reset()

        for i, result in enumerate(final_result):
            self.population_storage.insert_chromosome(self._population[i], result)

        self.population_storage.save()

    def _fitness_calculation(self):
        fitness_success_chromosomes: list[ChromosomeType] = []
        for chromosome in self._population:
            gene_id = ""
            for gene in chromosome.gene_tuple:
                gene_id += gene.label

            success, fitness, biased_fitness, calc_result = (
                self._gen_calc_result[gene_id]
                if gene_id in self._gen_calc_result and self.use_cache
                else (self.fitness_calculator.judge_fitness(chromosome))
            )

            if gene_id not in self._gen_calc_result:
                self._gen_calc_result[gene_id] = (
                    success,
                    fitness,
                    biased_fitness,
                    calc_result,
                )
            # Update the chromosome fitness
            chromosome.update_fitness(fitness, biased_fitness, calc_result)

            if success:
                self.population_storage.insert_chromosome(
                    chromosome, (fitness, biased_fitness)
                )
                self.evolution_storage.insert_fitness_result(
                    generation=self._generation,
                    fitness_result=(fitness, biased_fitness),
                )
                fitness_success_chromosomes.append(chromosome)
                if self.immediate_exit_condition((fitness, biased_fitness)):
                    self._should_stop = True
                    print(
                        f"Extraordinary chromosome found {chromosome.label}, stopping the evolution."
                    )
                    break

        self._population = fitness_success_chromosomes

    def _crossover_popularization(self):
        child_population: list[ChromosomeType] = []

        index_list = list(range(self.population_count))
        shuffle(index_list)

        for i, random_index in enumerate(index_list):
            parent1: ChromosomeType = self._population[random_index]
            parent2: ChromosomeType = self._population[
                index_list[(i + 1) % self.population_count]
            ]

            try:
                children: ChromosomeType = parent1.crossover(
                    behavior=self.crossover_behavior,
                    other=parent2,
                )
                child_population.append(children)

            except Exception as e:
                print(f"Error at crossover chromosome is skipped: {e}")
                continue

        self._population = child_population

    def _mutate_popularization(self):
        for chromosome in self._population:
            random_value = random()
            if random_value < self.mutation_probability:
                try:
                    chromosome.mutate_genes()
                    self._mutation_count += 1
                except Exception as e:
                    print(f"Error at mutation chromosome is skipped: {e}")
                    continue

    @property
    def _terminate_condition(self) -> bool:
        """
        Check if the GA termination condition is met
        """
        return (
            (self._generation >= self.suite_max_count and self._generation != 0)
            or self._should_stop
            or self.population_count <= self.suite_min_population
            or (
                self.unique_population_count <= self.suite_min_chromosome
                and self._generation != 0
            )
        )

    def _run_generation(self):
        # Phase 1: Reset population for next generation
        self.population_storage.reset()

        # Phase 2: Fitness Calculation
        self._fitness_calculation()
        if self._should_stop:
            print(
                f"Stopping the evolution at generation {self._generation} for extraordinary chromosome."
            )
            return

        if self._terminate_condition is True:
            return

        # Phase 3: Selection
        self._population = self.selector_behavior.select(
            self._population, self.population_storage
        )

        # Phase 4: Crossover
        self._crossover_popularization()

        # Phase 5: Mutation
        self._mutate_popularization()

        # Phase 6: Fitness Calculation for children
        if self.check_children_fitness:
            self._fitness_calculation()

        # Phase 7: Increment generation
        self._generation += 1

    def _log_result(self, title: str):
        """
        Log the suite information
        """
        print("-" * 100)
        print(f">> {title}")
        print("-" * 100)

        for key, value in self.suite_info.items():
            print(f"{key.capitalize()}: {value}")
        print("-" * 100)

    def _log_generation(self):
        """
        Log the generation information
        """
        print("-" * 100)
        print(
            f">> Generation: {self._generation}, Population: {self.population_count}, Unique population: {self.unique_population_count}, Should Stop: {self._should_stop}"
        )
        print("-" * 100)

    def _record_generation(self):
        """
        Record the generation information
        """
        Gene.parameter_storage.save()
        self.evolution_storage.save()
        self.population_storage.save()

    def run(self):
        """
        Run the genetic algorithm pipeline
        """
        self._log_result("GA started")

        self._pre_initialize()

        while not self._terminate_condition:
            self._log_generation()
            self._run_generation()
            self._record_generation()

        self._post_exit()

        self._log_result("GA finished")

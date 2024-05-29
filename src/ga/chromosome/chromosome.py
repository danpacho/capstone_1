from abc import abstractmethod
from typing import Generic, TypeVar, Union

import numpy as np

from src.ga.p4_crossover.crossover_behavior import CrossoverBehavior
from src.ga.gene.gene import Gene

ChromosomeGene = TypeVar("ChromosomeGene", bound=tuple[Gene, ...])


class Chromosome(Generic[ChromosomeGene]):
    """
    Chromosome class

    Attributes:
        gene_tuple (`tuple[Gene, ...]`): The gene list of the chromosome
        chromosome_id (`str`): The ID of the chromosome
        label (`str`): The label of the chromosome
        fitness_pure_result (`Union[tuple, None]`): The fitness pure result
        fitness (`float`): The fitness value of the chromosome
        biased_fitness (`float`): The biased fitness value of the chromosome
    """

    def __init__(
        self,
        label: str,
        chromosome_id: str,
        gene_tuple: ChromosomeGene,
    ):
        """
        Args:
            label (`str`): The label of the chromosome
            chromosome_id (`str`): The ID of the chromosome
            gene_tuple (`tuple[Gene, ...]`): The gene list of the chromosome
        """
        chromosome_id = str(chromosome_id)

        self.gene_tuple = gene_tuple
        self.chromosome_id = chromosome_id
        self.label = f"{label}_{self.chromosome_id}"
        self.label += f": {[gene.label for gene in gene_tuple]}"

        self.fitness_pure_result: Union[tuple, None] = None
        self.fitness: float = 0.0
        self.biased_fitness: float = 0.0

        for gene in gene_tuple:
            gene.update_gene_id(self.chromosome_id)

    @abstractmethod
    def crossover_genes(self, other_chromosome: "Chromosome") -> "Chromosome":
        """
        Crosses over the chromosome with another chromosome

        Args:
            other_chromosome (`Chromosome`): The other chromosome to crossover with

        Returns:
            `Chromosome`: The new chromosome created from the crossover
        """
        raise NotImplementedError

    def crossover_gene_params(
        self, behavior: CrossoverBehavior, other_chromosome: "Chromosome"
    ) -> dict[int, np.ndarray[np.float64]]:
        """
        Crosses over the chromosome with another chromosome

        Returns:
            `dict[int, np.ndarray[np.float64]]`: The new chromosome created from the crossover
        Examples:
        ```python
        {
            0: [0.324, 0.123, 0.234], # Gene1
            1: [1.8994, 1.3894032]    # Gene2
        }
        ```
        """
        crossover_result: dict[int, np.ndarray[np.float64]] = {}
        for index, gene in enumerate(self.gene_tuple):
            crossover_result[index] = behavior.crossover(
                gene.parameter_list, other_chromosome.gene_tuple[index].parameter_list
            )

        return crossover_result

    def _update_gene_parameters(
        self,
        new_chromosome: "Chromosome[tuple[Gene, ...]]",
        updated_gene_parameters: dict[int, np.ndarray[np.float64]],
    ):
        for i, gene in enumerate(new_chromosome.gene_tuple):
            gene.update_gene(updated_gene_parameters[i])

    def crossover(
        self,
        behavior: CrossoverBehavior,
        other: "Chromosome[tuple[Gene, ...]]",
    ) -> "Chromosome[tuple[Gene, ...]]":
        """
        Crossover the chromosome with another chromosome
        """
        child_chromosome: "Chromosome[tuple[Gene, ...]]" = self.crossover_genes(other)

        if self.is_gene_parameter_crossover_possible(child_chromosome):
            crossover_gene_params = self.crossover_gene_params(
                behavior, child_chromosome
            )
            self._update_gene_parameters(
                child_chromosome,
                crossover_gene_params,
            )

        return child_chromosome

    @abstractmethod
    def mutate_genes(self) -> None:
        """
        Mutates the chromosome
        """
        raise NotImplementedError

    @property
    def gene_parameter_table(self) -> dict[str, np.ndarray[np.float64]]:
        """
        Returns a dictionary of gene parameters

        Example:
        ```python
        {
            "gene1": np.array([0.324, 0.123, 0.234]),
            "gene2": np.array([1.8994, 1.3894032])
        }
        ```
        """
        return {gene.label: gene.parameter_list for gene in self.gene_tuple}

    @property
    def gene_parameter_id(self) -> dict[str, set[str]]:
        """
        Returns a dictionary of gene parameter IDs

        Example:
        ```python
        {
            "gene1": {"param1", "param2", "param3"},
            "gene2": {"param4", "param5"}
        }
        """
        return {gene.label: set(gene.parameter_id_list) for gene in self.gene_tuple}

    @property
    def gene_parameter_id_set(self) -> list[set[str]]:
        """
        Returns a set of gene parameter IDs

        Example:
        ```python
        [
            {"param1", "param2", "param3"},
            {"param4", "param5"}
        ]
        """
        return [set(gene.parameter_id_list) for gene in self.gene_tuple]

    @property
    def gene_parameter(self) -> list[np.ndarray[np.float64]]:
        """
        Returns a numpy array of gene parameters

        Returns:
            `list[np.ndarray[np.float64]]`: gene parameters

        Example:
        ```python
        [
            [0.324, 0.123, 0.234],
            [1.8994, 1.3894032]
        ]
        ```
        """
        return [gene.parameter_list for gene in self.gene_tuple]

    def gene_at(self, at: int) -> Union[Gene, None]:
        """
        Returns the gene at the given index

        Args:
            at (`int`): The index of the gene

        Returns:
            `Gene`: The gene at the index
        """
        if at < 0 or at >= len(self.gene_tuple):
            return None

        return self.gene_tuple[at]

    def is_gene_parameter_crossover_possible(
        self,
        other: "Chromosome[tuple[Gene, ...]]",
    ) -> bool:
        """
        Check whether gene parameter is same or not
        """
        if len(self.gene_tuple) != len(other.gene_tuple):
            return False

        # Compare gene parameter id
        for i, param_set in enumerate(self.gene_parameter_id_set):
            if param_set != other.gene_parameter_id_set[i]:
                return False

        return True

from abc import abstractmethod
from typing import Generic, TypeVar, Union

import numpy as np

from src.ga.gene.gene import Gene

ChromosomeGene = TypeVar("ChromosomeGene", bound=tuple[Gene, ...])


class Chromosome(Generic[ChromosomeGene]):
    """
    Chromosome class
    """

    def __init__(
        self,
        label: str,
        chromosome_id: str,
        gene_tuple: ChromosomeGene,
    ):
        chromosome_id = str(chromosome_id)

        self.gene_tuple = gene_tuple
        self.chromosome_id = chromosome_id
        self.label = f"{label}_{self.chromosome_id}"
        self.label += f": {[gene.label for gene in gene_tuple]}"
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

    @abstractmethod
    def crossover_gene_params(self, other_chromosome: "Chromosome") -> "Chromosome":
        """
        Crosses over the chromosome with another chromosome

        Args:
            other_chromosome (`Chromosome`): The other chromosome to crossover with

        Returns:
            `Chromosome`: The new chromosome created from the crossover
        """
        raise NotImplementedError

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Crossover the chromosome with another chromosome
        """
        base_crossover = self.crossover_genes(other)

        if self.is_gene_parameter_crossover_possible(base_crossover):
            base_crossover = self.crossover_gene_params(base_crossover)

        return base_crossover

    @abstractmethod
    def mutate_genes(self) -> "Chromosome":
        """
        Mutates the chromosome

        Returns:
            `Chromosome`: The new chromosome after mutation
        """
        raise NotImplementedError

    @property
    def gene_parameter_table(self):
        """
        Returns a dictionary of gene parameters

        Example:
        ```python
        {
            "gene1": [0.324, 0.123, 0.234],
            "gene2": [1.8994, 1.3894032]
        }
        ```
        """
        return {gene.label: gene.parameter_list for gene in self.gene_tuple}

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

    def is_gene_parameter_crossover_possible(self, other: "Chromosome") -> bool:
        """
        Check if the gene parameters of the chromosome
        and the other chromosome are compatible for crossover

        If possible, then it can be crossover deeply, otherwise, it can be crossover shallowly
        """
        for i, curr_params in enumerate(self.gene_parameter):
            other_params = other.gene_parameter[i]
            if curr_params.shape != other_params.shape:
                return False

        return True

from typing import Literal
from random import choice

from src.geometry.pattern_unit import Pattern, PatternTransformationMatrix
from src.ga.chromosome.chromosome import Chromosome
from src.ga.gene.shape.shape_gene import ShapeGene
from src.ga.gene.pattern.pattern_gene import PatternGene


class VentHole(Chromosome[tuple[ShapeGene, PatternGene]]):
    """
    VentHole class
    """

    def __init__(
        self,
        vent_id: str,
        gene_tuple: tuple[ShapeGene, PatternGene],
        pattern_bound: tuple[tuple[float, float], tuple[float, float]],
    ):
        super().__init__(
            label="VentHole Chromosome",
            chromosome_id=vent_id,
            gene_tuple=gene_tuple,
        )

        self.pattern_bound = pattern_bound
        self.pattern = Pattern(
            PatternTransformationMatrix(
                self.gene_tuple[0].pattern_unit,
                self.gene_tuple[1].param.transformation,
                pattern_bound=pattern_bound,
            )
        )

    def crossover_genes(self, other_chromosome: "VentHole") -> "VentHole":
        selection: tuple[Literal[0], Literal[1]] = (0, 1)
        first_choice = choice(selection)
        if first_choice == 0:
            second_choice = 1
            child_chromosome = VentHole(
                gene_tuple=(
                    self.gene_tuple[first_choice],
                    other_chromosome.gene_tuple[second_choice],
                ),
                vent_id=self.chromosome_id,  # Inherit the ID from the parent
                pattern_bound=self.pattern_bound,
            )
        else:
            second_choice = 0
            child_chromosome = VentHole(
                gene_tuple=(
                    other_chromosome.gene_tuple[second_choice],
                    self.gene_tuple[first_choice],
                ),
                vent_id=self.chromosome_id,  # Inherit the ID from the parent
                pattern_bound=self.pattern_bound,
            )

        return child_chromosome

    mutation_distribution = [0] * 5 + [1] * 95
    """
    The distribution of the mutation method

    0: Mutate the whole gene at once
    1: Mutate the gene at a specific index
    """

    def mutate_genes(self):
        # 0: Mutate the whole gene at once
        # 1: Mutate the gene at a specific index
        for gene in self.gene_tuple:
            rand_choice = choice(gene.get_mutate_methods())
            method_choice = choice(self.mutation_distribution)

            if method_choice == 0:
                gene.mutate(rand_choice)
            else:
                rand_index = choice(range(gene.parameter_count))
                gene.mutate_at(rand_choice, rand_index)

    @property
    def shape_gene(self) -> ShapeGene:
        """
        Returns the shape gene
        """
        return self.gene_tuple[0]

    @property
    def pattern_gene(self) -> PatternGene:
        """
        Returns the pattern gene
        """
        return self.gene_tuple[1]

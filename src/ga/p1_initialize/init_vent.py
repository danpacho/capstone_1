from random import randint
from typing import Callable

from src.ga.p1_initialize.init_popularization import PopularizationInitializer

from src.ga.gene.gene import Gene
from src.ga.gene.pattern.pattern_gene import PatternGene, PatternGeneParameter
from src.ga.gene.shape.shape_gene import ShapeGene, ShapeGeneParameter
from src.ga.chromosome.vent_hole import VentHole


class VentInitializer(PopularizationInitializer[VentHole]):
    """
    VentInitializer class
    """

    def __init__(
        self,
        population_size: int,
        shape_gene_pool: list[Callable[[int], ShapeGeneParameter]],
        pattern_gene_pool: list[Callable[[int], PatternGeneParameter]],
        grid_scale: int,
        pattern_bound: tuple[tuple[float, float], tuple[float, float]],
    ):
        """
        Args:
            population_size (`int`): The size of the population
            shape_gene_pool (`list[Callable[[int], ShapeGeneParameter]`): The list of shape gene pool
            pattern_gene_pool (`list[Callable[[int], PatternGeneParameter]`): The list of pattern gene pool
            grid_scale (`int`): The grid scale
            pattern_bound (`tuple[tuple[float, float], tuple[float, float]]`): The pattern bound
        """
        super().__init__(population_size)

        self.shape_gene_pool = shape_gene_pool
        self.shape_gene_list: list[ShapeGene] = []

        self.pattern_gene_pool = pattern_gene_pool
        self.patter_gene_list: list[PatternGeneParameter] = []
        self.vent_chromosome_list: list[VentHole] = []

        self.grid_scale = grid_scale
        self.pattern_bound = (
            (pattern_bound[0][0] * grid_scale, pattern_bound[0][1] * grid_scale),
            (pattern_bound[1][0] * grid_scale, pattern_bound[1][1] * grid_scale),
        )

        self._reset_storage()

    def initialize(self):
        """
        Initialize the population
        """
        self._reset_storage()

        self._initialize_gene_by_random()

        self._save_storage()

        self._initialize_chromosome_by_genes()

        return self.vent_chromosome_list

    def _initialize_gene_by_random(self) -> None:
        for i in range(self.population_size):
            shape_rand_index = randint(0, len(self.shape_gene_pool) - 1)
            shape_gene = ShapeGene(
                gene_id=i,
                shape_parameter=self.shape_gene_pool[shape_rand_index](self.grid_scale),
            )
            shape_gene.mutate("rand")

            pattern_rand_index = randint(0, len(self.pattern_gene_pool) - 1)
            pattern_gene = PatternGene(
                gene_id=i,
                gene_parameter=self.pattern_gene_pool[pattern_rand_index](
                    self.grid_scale
                ),
            )
            pattern_gene.mutate("rand")

            self.shape_gene_list.append(shape_gene)
            self.patter_gene_list.append(pattern_gene)

    def _initialize_chromosome_by_genes(self) -> None:
        for i in range(self.population_size):
            vent_chromosome = VentHole(
                vent_id=i,
                gene_tuple=(
                    self.shape_gene_list[i],
                    self.patter_gene_list[i],
                ),
                pattern_bound=self.pattern_bound,
            )
            self.vent_chromosome_list.append(vent_chromosome)

    def _save_storage(self) -> None:
        Gene.parameter_storage.save()
        ShapeGene.pdf_storage.save()
        PatternGene.pdf_storage.save()

    def _reset_storage(self) -> None:
        Gene.parameter_storage.reset()
        ShapeGene.pdf_storage.reset()
        PatternGene.pdf_storage.reset()

        self.shape_gene_list = []
        self.patter_gene_list = []
        self.vent_chromosome_list = []

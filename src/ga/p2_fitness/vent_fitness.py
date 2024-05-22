from src.geometry.vector import V2_group
from src.ga.chromosome.vent_hole import VentHole
from src.ga.p2_fitness.fitness_calculator import FitnessCalculator


class VentFitnessCalculator(FitnessCalculator[VentHole]):
    """
    VentFitnessCalculator class
    """

    def __init__(
        self,
        min_criteria_value_list: list[float],
        effective_criteria_list: list[float],
        criteria_weight_list: list[float],
    ):
        """
        Args:
            min_criteria_value_list (`list[float]`): Minimum criteria values
            effective_criteria_list (`list[float]`): Effective criteria values
            criteria_weight_list (`list[float]`): Criteria weights
        """
        super().__init__(
            fitness_method_name="GPR",
            criteria_label_list=["drag", "max_temp", "avg_temp"],
            min_criteria_value_list=min_criteria_value_list,
            effective_criteria_list=effective_criteria_list,
            criteria_weight_list=criteria_weight_list,
        )

    def calculate(self, chromosome) -> list[float]:
        drag = self._calculate_drag(chromosome.pattern.pattern_matrix)
        max_temp = self._calculate_max_temp(chromosome.pattern.pattern_matrix)
        avg_temp = self._calculate_avg_temp(chromosome.pattern.pattern_matrix)
        return [drag, max_temp, avg_temp]

    def _calculate_drag(self, pattern_matrix: V2_group) -> float:
        return 0.0

    def _calculate_max_temp(self, pattern_matrix: V2_group) -> float:
        return 0.0

    def _calculate_avg_temp(self, pattern_matrix: V2_group) -> float:
        return 0.0

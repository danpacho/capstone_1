from typing import Literal, Union
from random import uniform

from src.geometry.vector import V2_group
from src.ga.chromosome.vent_hole import VentHole
from src.ga.p2_fitness.fitness_calculator import FitnessCalculator


class VentFitnessCalculator(FitnessCalculator[VentHole]):
    """
    VentFitnessCalculator class
    """

    def __init__(
        self,
        criteria_weight_list: tuple[float, float, float],
        drag_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        max_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        avg_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
    ):
        """
        Args:
            criteria_weight_list: Criteria weights, `("drag", "max_temp", "avg_temp")`
            drag_criterion: A tuple representing (direction, min, max) for drag
            max_temp_criterion: A tuple representing (direction, min, max) for max_temp
            avg_temp_criterion: A tuple representing (direction, min, max) for avg_temp
        """
        super().__init__(
            fitness_method_name="GPR",
            criteria_label_list=["drag", "max_temp", "avg_temp"],
            criteria_weight_list=list(criteria_weight_list),
        )

        self.drag_criterion = drag_criterion
        self.max_temp_criterion = max_temp_criterion
        self.avg_temp_criterion = avg_temp_criterion

    def calculate(self, chromosome) -> list[float]:
        drag = self._calculate_drag(chromosome.pattern.pattern_matrix)
        max_temp = self._calculate_max_temp(chromosome.pattern.pattern_matrix)
        avg_temp = self._calculate_avg_temp(chromosome.pattern.pattern_matrix)

        return [drag, max_temp, avg_temp]

    def judge_fitness(self, chromosome: VentHole) -> tuple[bool, float, float]:
        drag, max_temp, avg_temp = self.calculate(chromosome)
        drag_valid = self._is_valid(drag, self.drag_criterion)
        max_temp_valid = self._is_valid(max_temp, self.max_temp_criterion)
        avg_temp_valid = self._is_valid(avg_temp, self.avg_temp_criterion)

        valid = drag_valid and max_temp_valid and avg_temp_valid
        biased_fitness_score = self._calculate_fitness(
            (drag, max_temp, avg_temp), self.criteria_weight_list
        )
        fitness_score = self._calculate_fitness((drag, max_temp, avg_temp), [1, 1, 1])

        return valid, fitness_score, biased_fitness_score

    def _is_valid(
        self,
        result: float,
        criterion: tuple[Union[Literal["lower", "upper"]], float, float],
    ) -> bool:
        """
        Check if the result satisfies the criterion.

        Args:
            result: The value to be checked.
            criterion: A tuple containing the direction
            ('lower' or 'upper'), min value, and max value.

        Returns:
            bool: True if the result satisfies the criterion, False otherwise.
        """
        if criterion[0] == "lower":
            return result <= criterion[2]  # Use max value for 'lower' direction
        else:
            return result >= criterion[1]  # Use min value for 'upper' direction

    def _normalize(
        self, value: float, min_value: float, max_value: float, direction: str
    ) -> float:
        """
        Normalize the value based on the direction of better fitness.

        Args:
            value: The value to be normalized.
            min_value: The minimum value in the range.
            max_value: The maximum value in the range.
            direction: The direction indicating
            if a higher or lower value is better ('higher' or 'lower').

        Returns:
            float: The normalized value between 0 and 1.
        """
        if direction == "lower":
            return 1 - (value - min_value) / (max_value - min_value)
        else:  # 'higher'
            return (value - min_value) / (max_value - min_value)

    def _calculate_fitness(
        self, results: tuple[float, float, float], weights: tuple[float, float, float]
    ) -> float:
        """
        Calculate the overall fitness score by normalizing each criterion and applying weights.

        Args:
            results: A tuple containing the raw values for
            drag, max temperature, and average temperature.
            weights: A tuple containing the weights for
            drag, max temperature, and average temperature.

        Returns:
            float: The calculated fitness score.
        """
        drag, max_temp, avg_temp = results

        # Extract min and max values from the criteria
        drag_direction, drag_min, drag_max = self.drag_criterion
        max_temp_direction, max_temp_min, max_temp_max = self.max_temp_criterion
        avg_temp_direction, avg_temp_min, avg_temp_max = self.avg_temp_criterion

        # Normalize each criterion
        drag_norm = self._normalize(drag, drag_min, drag_max, direction=drag_direction)
        max_temp_norm = self._normalize(
            max_temp, max_temp_min, max_temp_max, direction=max_temp_direction
        )
        avg_temp_norm = self._normalize(
            avg_temp, avg_temp_min, avg_temp_max, direction=avg_temp_direction
        )

        # Calculate the fitness score
        fitness_score = (
            weights[0] * drag_norm
            + weights[1] * max_temp_norm
            + weights[2] * avg_temp_norm
        )

        return fitness_score

    def _calculate_drag(self, pattern_matrix: V2_group) -> float:
        """
        Calculate the drag value for the given pattern matrix.

        Args:
            pattern_matrix: The pattern matrix for the vent hole.

        Returns:
            float: The calculated drag value.
        """
        return uniform(0.25, 0.35)

    def _calculate_max_temp(self, pattern_matrix: V2_group) -> float:
        """
        Calculate the maximum temperature for the given pattern matrix.

        Args:
            pattern_matrix: The pattern matrix for the vent hole.

        Returns:
            float: The calculated maximum temperature.
        """
        return uniform(400, 500)

    def _calculate_avg_temp(self, pattern_matrix: V2_group) -> float:
        """
        Calculate the average temperature for the given pattern matrix.

        Args:
            pattern_matrix: The pattern matrix for the vent hole.

        Returns:
            float: The calculated average temperature.
        """
        return uniform(300, 400)

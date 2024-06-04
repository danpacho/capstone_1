from typing import Literal, Union

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from src.prediction.to_model_input import to_model_input
from src.ga.chromosome.vent_hole import VentHole
from src.ga.p2_fitness.fitness_calculator import FitnessCalculator


Criterion = tuple[Union[Literal["lower", "upper"]], float, float]
"""
Criterion: A tuple containing the direction `('lower' or 'upper')`, min value, and max value.

Example:
    ```python
    # Lower is better, range 0.2 to 0.5
    ("lower", 0.2, 0.5)

    # Higher is better, range 10 to 100
    ("higher", 10, 100) 
    ```
"""


class VentFitnessCalculator(FitnessCalculator[VentHole]):
    """
    VentFitnessCalculator class
    """

    def __init__(
        self,
        rf_models: tuple[
            RandomForestRegressor, RandomForestRegressor, RandomForestRegressor
        ],
        pca: PCA,
        drag_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        drag_std_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        avg_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        avg_temp_std_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        max_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        max_temp_std_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        criteria_weight_list: tuple[float, float, float],
    ):
        """
        Args:
            rf_models: tuple of RF models for drag, avg_temp, and max_temp.
            pca: PCA model for the pattern matrix.
            drag_criterion: The criterion for drag.
            drag_std_criterion: The criterion for drag std.
            avg_temp_criterion: The criterion for average temperature.
            avg_temp_std_criterion: The criterion for average temperature std.
            max_temp_criterion: The criterion for max temperature.
            max_temp_std_criterion: The criterion for max temperature std.
            criteria_weight_list: The weight list for drag, avg_temp, and max_temp.
        """
        super().__init__(
            fitness_method_name="RF",
            criteria_label_list=["drag", "avg_temp", "max_temp"],
            criteria_weight_list=list(criteria_weight_list),
        )

        self.rf_models = rf_models
        self.pca = pca

        self.drag_criterion = drag_criterion
        self.drag_std_criterion = drag_std_criterion
        self.avg_temp_criterion = avg_temp_criterion
        self.avg_temp_std_criterion = avg_temp_std_criterion
        self.max_temp_criterion = max_temp_criterion
        self.max_temp_std_criterion = max_temp_std_criterion

    def calculate(self, chromosome) -> list[float]:
        drag, avg_temp, max_temp = self._rf_predict(chromosome)

        return [drag[0], drag[1], avg_temp[0], avg_temp[1], max_temp[0], max_temp[1]]

    def _rf_predict(
        self, chromosome: VentHole
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        input_matrix = to_model_input(
            pca=self.pca,
            pattern_matrix=chromosome.pattern.pattern_matrix,
            bound=chromosome.pattern_bound,
            resolution=chromosome.pattern.pattern_unit.grid.k,
            flat=False,
        )
        drag, drag_std = self.rf_models[0].predict(input_matrix, return_std=True)
        drag = drag[0]
        drag_std = drag_std[0]

        avg_temp, avg_temp_std = self.rf_models[1].predict(
            input_matrix, return_std=True
        )
        avg_temp = avg_temp[0]
        avg_temp_std = avg_temp_std[0]

        max_temp, max_temp_std = self.rf_models[2].predict(
            input_matrix, return_std=True
        )
        max_temp = max_temp[0]
        max_temp_std = max_temp_std[0]

        return (drag, drag_std), (avg_temp, avg_temp_std), (max_temp, max_temp_std)

    def judge_fitness(
        self, chromosome: VentHole
    ) -> tuple[bool, float, float, tuple[float, float, float]]:
        drag, drag_std, avg_temp, avg_temp_std, max_temp, max_temp_std = self.calculate(
            chromosome
        )
        drag_valid = self._is_valid(drag, self.drag_criterion)
        drag_std_valid = self._is_valid(drag_std, self.drag_std_criterion)

        avg_temp_valid = self._is_valid(avg_temp, self.avg_temp_criterion)
        avg_temp_std_valid = self._is_valid(avg_temp_std, self.avg_temp_std_criterion)

        max_temp_valid = self._is_valid(max_temp, self.max_temp_criterion)
        max_temp_std_valid = self._is_valid(max_temp_std, self.max_temp_std_criterion)

        valid: bool = (
            drag_valid
            and drag_std_valid
            and avg_temp_valid
            and avg_temp_std_valid
            and max_temp_valid
            and max_temp_std_valid
        )

        biased_fitness_score: float = self._calculate_fitness(
            (drag, avg_temp, max_temp), self.criteria_weight_list
        )
        fitness_score: float = self._calculate_fitness(
            (drag, avg_temp, max_temp), (1, 1, 1)
        )

        return (
            valid,
            fitness_score,
            biased_fitness_score,
            (drag, avg_temp, max_temp),
        )

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
        self,
        results: tuple[
            float,
            float,
            float,
        ],
        weights: tuple[
            float,
            float,
            float,
        ],
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
        drag, avg_temp, max_temp = results

        # Extract min and max values from the criteria
        drag_direction, drag_min, drag_max = self.drag_criterion
        max_temp_direction, max_temp_min, max_temp_max = self.max_temp_criterion
        avg_temp_direction, avg_temp_min, avg_temp_max = self.avg_temp_criterion

        # Normalize each criterion
        drag_norm = self._normalize(drag, drag_min, drag_max, direction=drag_direction)
        avg_temp_norm = self._normalize(
            avg_temp, avg_temp_min, avg_temp_max, direction=avg_temp_direction
        )
        max_temp_norm = self._normalize(
            max_temp, max_temp_min, max_temp_max, direction=max_temp_direction
        )

        # Calculate the fitness score
        fitness_score = (
            weights[0] * drag_norm
            + weights[1] * avg_temp_norm
            + weights[2] * max_temp_norm
        )

        return fitness_score

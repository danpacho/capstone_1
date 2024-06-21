from typing import Literal, Union

from sklearn.decomposition import PCA

from src.geometry.vector import V3_group
from src.prediction.model_trainer import ModelTrainer
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


class VentFitnessCalculator(
    FitnessCalculator[
        VentHole,
        tuple[
            ModelTrainer,  # TODO: Plug optimized model
            ModelTrainer,  # TODO: Plug optimized model
            ModelTrainer,  # TODO: Plug optimized model
        ],
    ]
):
    """
    VentFitnessCalculator class
    """

    def __init__(
        self,
        model_trainer_tuple: tuple[ModelTrainer, ModelTrainer, ModelTrainer],
        drag_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        avg_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        max_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        criteria_weight_list: tuple[float, float, float],
        # Standard deviation criteria is optional
        drag_std_criterion: Union[
            tuple[Union[Literal["lower", "upper"]], float, float], None
        ] = None,
        avg_temp_std_criterion: Union[
            tuple[Union[Literal["lower", "upper"]], float, float], None
        ] = None,
        max_temp_std_criterion: Union[
            tuple[Union[Literal["lower", "upper"]], float, float], None
        ] = None,
    ):
        """
        Args:
            model_trainer_tuple: tuple of `ModelTrainer` for `drag`, `avg_temp`, `max_temp`.
            drag_criterion: The criterion for drag.
            drag_std_criterion: The criterion for drag std (optional, default = `None`).
            avg_temp_criterion: The criterion for average temperature.
            avg_temp_std_criterion: The criterion for average temperature std (optional, default = `None`).
            max_temp_criterion: The criterion for max temperature.
            max_temp_std_criterion: The criterion for max temperature std (optional, default = `None`).
            criteria_weight_list: The weight list for drag, avg_temp, and max_temp.
        """
        super().__init__(
            fitness_method_name="GPR",
            model_trainer_list=model_trainer_tuple,
            criteria_label_list=["drag", "avg_temp", "max_temp"],
            criteria_weight_list=list(criteria_weight_list),
        )

        # 1. Train the models
        for i, model_trainer in enumerate(self.model_trainer_list):
            model_trainer.train_model([0, 0])
            model = model_trainer.get_model()
            if i == 0:
                self._drag_model_trainer = model_trainer
                self._drag_model = model[0]
            elif i == 1:
                self._avg_model_trainer = model_trainer
                self._avg_temp_model = model[1]
            elif i == 2:
                self._max_model_trainer = model_trainer
                self._max_temp_model = model[2]

            self.pca: PCA = model_trainer.get_pca()

        self.drag_criterion = drag_criterion
        self.drag_std_criterion = drag_std_criterion
        self.avg_temp_criterion = avg_temp_criterion
        self.avg_temp_std_criterion = avg_temp_std_criterion
        self.max_temp_criterion = max_temp_criterion
        self.max_temp_std_criterion = max_temp_std_criterion

    def get_model(self, model_type: Literal["drag", "avg_temp", "max_temp"]):
        """
        Get the ModelTrainer.

        Args:
            model_type: The type of model to get.

        Returns:
            `ModelTrainer`
        """
        if model_type == "drag":
            return self._drag_model
        if model_type == "avg_temp":
            return self._avg_temp_model
        if model_type == "max_temp":
            return self._max_temp_model

    def calculate(self, chromosome) -> tuple[
        float,
        Union[float, None],
        float,
        Union[float, None],
        float,
        Union[float, None],
    ]:
        drag, avg_temp, max_temp = self._predict(chromosome)

        return (drag[0], drag[1], avg_temp[0], avg_temp[1], max_temp[0], max_temp[1])

    def _predict_single(
        self,
        model_trainer: ModelTrainer,
        model: any,
        model_input: V3_group,
    ) -> tuple[
        float,
        Union[float, None],
    ]:
        if model_trainer.can_calculate_std:
            result, std = model.predict(model_input, return_std=True)
            result: float = result[0]
            std: float = std[0]
            return (result, std)

        result = model.predict(model_input)
        result: float = result[0]
        return (result, None)

    def _predict(self, chromosome: VentHole) -> tuple[
        tuple[float, Union[float, None]],
        tuple[float, Union[float, None]],
        tuple[float, Union[float, None]],
    ]:
        input_matrix = to_model_input(
            pca=self.pca,
            pattern_matrix=chromosome.pattern.pattern_matrix,
            bound=chromosome.pattern_bound,
            resolution=chromosome.pattern.pattern_unit.grid.k,
            flat=False,
        )
        drag, drag_std = self._predict_single(
            self._drag_model_trainer, self._drag_model, input_matrix
        )
        avg_temp, avg_temp_std = self._predict_single(
            self._avg_model_trainer, self._avg_temp_model, input_matrix
        )
        max_temp, max_temp_std = self._predict_single(
            self._max_model_trainer, self._max_temp_model, input_matrix
        )

        return (drag, drag_std), (avg_temp, avg_temp_std), (max_temp, max_temp_std)

    def judge_fitness(
        self, chromosome: VentHole
    ) -> tuple[bool, float, float, tuple[float, float, float]]:
        drag, drag_std, avg_temp, avg_temp_std, max_temp, max_temp_std = self.calculate(
            chromosome
        )
        drag_valid = self._is_valid(drag, self.drag_criterion)
        drag_std_valid = (
            self._is_valid(drag_std, self.drag_std_criterion)
            if drag_std is not None
            else True
        )

        avg_temp_valid = self._is_valid(avg_temp, self.avg_temp_criterion)
        avg_temp_std_valid = (
            self._is_valid(avg_temp_std, self.avg_temp_std_criterion)
            if avg_temp_std is not None
            else True
        )

        max_temp_valid = self._is_valid(max_temp, self.max_temp_criterion)
        max_temp_std_valid = (
            self._is_valid(max_temp_std, self.max_temp_std_criterion)
            if max_temp_std is not None
            else True
        )

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

import torch
import numpy as np

from typing import Literal, Union

from src.prediction.cnn.cnn_trainer import CNNTrainer
from src.grid.grid import Grid
from src.prediction.cnn.to_image_matrix import to_image_matrix
from src.geometry.vector import V3_group
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


class VentFitnessCalculatorCNN(
    FitnessCalculator[
        VentHole,
        tuple[
            CNNTrainer,
            CNNTrainer,
            CNNTrainer,
        ],
    ]
):
    """
    VentFitnessCalculator class
    """

    def __init__(
        self,
        model_trainer_tuple: tuple[CNNTrainer, CNNTrainer, CNNTrainer],
        grid_bound: tuple[tuple[float, float], tuple[float, float]],
        grid_resolution: float,
        drag_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        avg_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        max_temp_criterion: tuple[Union[Literal["lower", "upper"]], float, float],
        criteria_weight_list: tuple[float, float, float],
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
            fitness_method_name="DCNN",
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
                self._drag_model = model
            elif i == 1:
                self._avg_model_trainer = model_trainer
                self._avg_temp_model = model
            elif i == 2:
                self._max_model_trainer = model_trainer
                self._max_temp_model = model

        self.drag_criterion = drag_criterion
        self.avg_temp_criterion = avg_temp_criterion
        self.max_temp_criterion = max_temp_criterion

        self.full_grid_matrix = Grid(
            bound=grid_bound, k=1 / grid_resolution
        ).generate_grid(scale=1, x_major_iteration=True)

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

    def calculate(self, chromosome) -> tuple[float, float, float]:
        return self._predict(chromosome)

    def to_torch_image_tensor(self, image_matrix: np.ndarray) -> torch.Tensor:
        """
        Convert the image matrix to a torch image tensor.

        Args:
            image_matrix: The image matrix as a numpy array of shape (height, width).

        Returns:
            The torch image tensor.
        """
        img_tensor = torch.from_numpy(image_matrix).float()  # Convert to torch tensor
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and channel dimensions
        img_tensor = img_tensor.to(self._drag_model_trainer.model_device)

        return img_tensor

    def _predict_single(
        self,
        model: any,
        model_input: V3_group,
    ) -> float:
        img_tensor = self.to_torch_image_tensor(model_input)
        result = model(img_tensor)
        result = result.squeeze(1)
        result = result.item()
        return result

    def plot_img_tensor(self, img_tensor: torch.Tensor, name: str):
        """
        Plot the image tensor.

        Args:
            img_tensor: The image tensor.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(2, 2))
        plt.title(name)
        plt.imshow(img_tensor.cpu().squeeze(0).squeeze(0).detach().numpy(), cmap="gray")
        plt.show()

    def _predict(self, chromosome: VentHole) -> tuple[float, float, float]:
        input_matrix = to_image_matrix(
            full_grid_matrix=self.full_grid_matrix,
            pattern_matrix=chromosome.pattern.pattern_matrix,
            grid_resolution=1 / chromosome.pattern.pattern_unit.grid.k,
            grid_width=(
                chromosome.pattern.pattern_transformation_matrix.p_bound_x_max
                - chromosome.pattern.pattern_transformation_matrix.p_bound_x_min
            ),
        )
        # self.plot_img_tensor(
        #     self.to_torch_image_tensor(input_matrix), chromosome.chromosome_id
        # )
        drag = self._predict_single(self._drag_model, input_matrix)
        avg_temp = self._predict_single(self._avg_temp_model, input_matrix)
        max_temp = self._predict_single(self._max_temp_model, input_matrix)

        return (drag, avg_temp, max_temp)

    def judge_fitness(
        self, chromosome: VentHole
    ) -> tuple[bool, float, float, tuple[float, float, float]]:
        drag, avg_temp, max_temp = self.calculate(chromosome)
        drag_valid = self._is_valid(drag, self.drag_criterion)
        avg_temp_valid = self._is_valid(avg_temp, self.avg_temp_criterion)
        max_temp_valid = self._is_valid(max_temp, self.max_temp_criterion)

        valid: bool = drag_valid and avg_temp_valid and max_temp_valid

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

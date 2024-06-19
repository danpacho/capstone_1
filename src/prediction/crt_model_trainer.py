"""
CRTModelTrainer class which is responsible for training a Classification and Regression Trees (CRT) model.
"""

from sklearn.tree import DecisionTreeRegressor

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class CrtModelTrainer(
    ModelTrainer[
        tuple[
            DecisionTreeRegressor,
            DecisionTreeRegressor,
            DecisionTreeRegressor,
        ]
    ]
):
    """
    CRT model trainer class.
    """

    def __init__(
        self,
        # crt configs ------ start
        crt_drag_config: float,
        crt_avg_temp_config: float,
        crt_max_temp_config: float,
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "CRT",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = crt_drag_config
        self.avg_temp_config = crt_avg_temp_config
        self.max_temp_config = crt_max_temp_config

    def train_model(
        self,
        test_boundary=None,
    ) -> tuple[DecisionTreeRegressor, DecisionTreeRegressor, DecisionTreeRegressor]:
        """
        Trains a Classification and Regression Trees (CRT) model.

        Returns:
            - tuple[DecisionTreeRegressor, DecisionTreeRegressor, DecisionTreeRegressor]: The trained CRT model.
            - `(crt for drag, crt for average temperature, crt for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(
            use_original_input=False, test_boundary=test_boundary
        )

        crt_drag = DecisionTreeRegressor(random_state=self.drag_config)
        crt_avg_temp = DecisionTreeRegressor(random_state=self.avg_temp_config)
        crt_max_temp = DecisionTreeRegressor(random_state=self.max_temp_config)

        crt_output_drag = output_matrix[:, 0]
        crt_output_avg_temp = output_matrix[:, 1]
        crt_output_max_temp = output_matrix[:, 2]

        crt_drag.fit(input_matrix, crt_output_drag)
        crt_avg_temp.fit(input_matrix, crt_output_avg_temp)
        crt_max_temp.fit(input_matrix, crt_output_max_temp)

        self._model = (crt_drag, crt_avg_temp, crt_max_temp)

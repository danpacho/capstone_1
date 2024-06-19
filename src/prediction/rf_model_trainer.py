"""
ModelTrainer class which is responsible for training a random Forest Regressor (rf) model.
"""

from sklearn.ensemble import RandomForestRegressor

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class RfModelTrainer(
    ModelTrainer[
        tuple[
            RandomForestRegressor,
            RandomForestRegressor,
            RandomForestRegressor,
        ]
    ]
):
    """
    Random Forest model trainer class.
    """

    def __init__(
        self,
        # Random_Forest_configs ------ start
        # Random Forest_configs: tuple [n_estimators, random_state]
        rf_drag_config: tuple[int, int],
        rf_avg_temp_config: tuple[int, int],
        rf_max_temp_config: tuple[int, int],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "Random_Forest",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = rf_drag_config
        self.avg_temp_config = rf_avg_temp_config
        self.max_temp_config = rf_max_temp_config

    def train_model(
        self,
        test_boundary=None,
    ) -> tuple[RandomForestRegressor, RandomForestRegressor, RandomForestRegressor]:
        """
        Trains a Random Forrest Regressor model.

        Returns:
            - tuple[RandomForrestRegressor, RandomForrestRegressor, RandomForrestRegressor]: The trained GPR model.
            - `(Random_Forest for drag, Random_Forest for average temperature, Random_Forest for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(
            use_original_input=False, test_boundary=test_boundary
        )

        rf_drag = RandomForestRegressor(
            n_estimators=self.drag_config[0],
            random_state=self.drag_config[1],
        )
        rf_avg_temp = RandomForestRegressor(
            n_estimators=self.avg_temp_config[0],
            random_state=self.avg_temp_config[1],
        )
        rf_max_temp = RandomForestRegressor(
            n_estimators=self.max_temp_config[0],
            random_state=self.max_temp_config[1],
        )

        rf_output_drag = output_matrix[:, 0]
        rf_output_avg_temp = output_matrix[:, 1]
        rf_output_max_temp = output_matrix[:, 2]

        rf_drag.fit(input_matrix, rf_output_drag)
        rf_avg_temp.fit(input_matrix, rf_output_avg_temp)
        rf_max_temp.fit(input_matrix, rf_output_max_temp)

        self._model = (rf_drag, rf_avg_temp, rf_max_temp)

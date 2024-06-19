"""
ETRModelTrainer class which is responsible for training a Extra Trees Regressor (ETR) model.
"""

from sklearn.ensemble import ExtraTreesRegressor

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class EtrModelTrainer(
    ModelTrainer[
        tuple[
            ExtraTreesRegressor,
            ExtraTreesRegressor,
            ExtraTreesRegressor,
        ]
    ]
):
    """
    ETR model trainer class.
    """

    def __init__(
        self,
        # etr configs ------ start
        etr_drag_config: tuple[int, int],
        etr_avg_temp_config: tuple[int, int],
        etr_max_temp_config: tuple[int, int],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "ETR",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = etr_drag_config
        self.avg_temp_config = etr_avg_temp_config
        self.max_temp_config = etr_max_temp_config

    def train_model(
        self,
        test_boundary=None,
    ) -> tuple[ExtraTreesRegressor, ExtraTreesRegressor, ExtraTreesRegressor]:
        """
        Trains a Extra Trees Regressor (ETR) model.

        Returns:
            - tuple[ExtraTreesRegressor, ExtraTreesRegressor, ExtraTreesRegressor]: The trained etr model.
            - `(etr for drag, etr for average temperature, etr for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(
            use_original_input=False, test_boundary=test_boundary
        )

        etr_drag = ExtraTreesRegressor(
            n_estimators=self.drag_config[0], random_state=self.drag_config[1]
        )
        etr_avg_temp = ExtraTreesRegressor(
            n_estimators=self.avg_temp_config[0], random_state=self.avg_temp_config[1]
        )
        etr_max_temp = ExtraTreesRegressor(
            n_estimators=self.max_temp_config[0], random_state=self.max_temp_config[1]
        )

        etr_output_drag = output_matrix[:, 0]
        etr_output_avg_temp = output_matrix[:, 1]
        etr_output_max_temp = output_matrix[:, 2]

        etr_drag.fit(input_matrix, etr_output_drag)
        etr_avg_temp.fit(input_matrix, etr_output_avg_temp)
        etr_max_temp.fit(input_matrix, etr_output_max_temp)

        self._model = (etr_drag, etr_avg_temp, etr_max_temp)

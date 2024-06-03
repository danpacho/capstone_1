"""
ModelTrainer class which is responsible for training a Gradient Boosting Regression (GBR) model.
"""

from xgboost import xgb

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class GBRModelTrainer(
    ModelTrainer[
        tuple[
            xgb,
            xgb,
            xgb,
        ]
    ]
):
    """
    Gradient_Boosting_Regression model trainer class.
    """

    def __init__(
        self,
        # gbr_configs ------ start
        # Gradient_Boosting_Regression_configs: tuple [n_estimators, random_state]
        gbr_drag_config: tuple[int, int],
        gbr_avg_temp_config: tuple[int, int],
        gbr_max_temp_config: tuple[int, int],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "gbr",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = gbr_drag_config
        self.avg_temp_config = gbr_avg_temp_config
        self.max_temp_config = gbr_max_temp_config

    def train_model(
        self,
    ) -> tuple[xgb, xgb, xgb]:
        """
        Trains a Gradient Boosting Regressor model.

        Returns:
            - tuple[GradientBoostingRegressor, GradientBoostingRegressor, GradientBoostingRegressor]: The trained GPR model.
            - `(gbr for drag, gbr for average temperature, gbr for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(use_original_input=False)

        gbr_drag = xgb(
            n_estimators=self.drag_config[0],
            random_state=self.drag_config[1],
        )
        gbr_avg_temp = xgb(
            n_estimators=self.avg_temp_config[0],
            random_state=self.avg_temp_config[1],
        )
        gbr_max_temp = xgb(
            n_estimators=self.max_temp_config[0],
            random_state=self.max_temp_config[1],
        )

        gbr_output_drag = output_matrix[:, 0]
        gbr_output_avg_temp = output_matrix[:, 1]
        gbr_output_max_temp = output_matrix[:, 2]

        gbr_drag.fit(input_matrix, gbr_output_drag)
        gbr_avg_temp.fit(input_matrix, gbr_output_avg_temp)
        gbr_max_temp.fit(input_matrix, gbr_output_max_temp)

        self._model = (gbr_drag, gbr_avg_temp, gbr_max_temp)

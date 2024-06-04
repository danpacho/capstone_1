"""
ModelTrainer class which is responsible for training a AdaBoostRegressor (ADA) model.
"""

from sklearn.ensemble import AdaBoostRegressor

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class AdaModelTrainer(
    ModelTrainer[tuple[AdaBoostRegressor, AdaBoostRegressor, AdaBoostRegressor]]
):
    """
    AdaBoostRegressor model trainer class.
    """

    def __init__(
        self,
        # ada_configs ------ start
        # AdaBoostRegressor_configs: tuple [n_estimators, random_state]
        ada_drag_config: tuple[int, int],
        ada_avg_temp_config: tuple[int, int],
        ada_max_temp_config: tuple[int, int],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "ADA",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = ada_drag_config
        self.avg_temp_config = ada_avg_temp_config
        self.max_temp_config = ada_max_temp_config

    def train_model(
        self,
    ) -> tuple[AdaBoostRegressor, AdaBoostRegressor, AdaBoostRegressor]:
        """
        Trains a AdaBoostRegressor model.

        Returns:
            - tuple[AdaBoostRegressor, AdaBoostRegressor, AdaBoostRegressor]: The trained ada model.
            - `(ada for drag, ada for average temperature, ada for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(use_original_input=False)

        ada_drag = AdaBoostRegressor(
            n_estimators=self.drag_config[0], random_state=self.drag_config[1]
        )
        ada_avg_temp = AdaBoostRegressor(
            n_estimators=self.avg_temp_config[0], random_state=self.avg_temp_config[1]
        )
        ada_max_temp = AdaBoostRegressor(
            n_estimators=self.max_temp_config[0], random_state=self.max_temp_config[1]
        )

        ada_output_drag = output_matrix[:, 0]
        ada_output_avg_temp = output_matrix[:, 1]
        ada_output_max_temp = output_matrix[:, 2]

        ada_drag.fit(input_matrix, ada_output_drag)
        ada_avg_temp.fit(input_matrix, ada_output_avg_temp)
        ada_max_temp.fit(input_matrix, ada_output_max_temp)

        self._model = (ada_drag, ada_avg_temp, ada_max_temp)

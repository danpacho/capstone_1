"""
EnsemModelTrainer class which is responsible for training a Decesion Tree based Ensemble(ensem) model.
"""

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from src.prediction.regression.regression_model_trainer import ModelTrainer


# pylint: disable=invalid-name


class EnsemModelTrainer(
    ModelTrainer[
        tuple[
            BaggingRegressor,
            BaggingRegressor,
            BaggingRegressor,
        ]
    ]
):
    """
    Ensem model trainer class.
    """

    def __init__(
        self,
        # ensem configs ------ start
        ensem_drag_config: tuple[int, int, int],
        ensem_avg_temp_config: tuple[int, int, int],
        ensem_max_temp_config: tuple[int, int, int],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "ENSEM",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.drag_config = ensem_drag_config
        self.avg_temp_config = ensem_avg_temp_config
        self.max_temp_config = ensem_max_temp_config

    def train_model(
        self,
        test_boundary=None,
    ) -> tuple[BaggingRegressor, BaggingRegressor, BaggingRegressor]:
        """
        Trains a Ensemble (ensem) model.

        Returns:
            - tuple[BaggingRegressor, BaggingRegressor, BaggingRegressor]: The trained ensem model.
            - `(ensem for drag, ensem for average temperature, ensem for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(
            use_original_input=False, test_boundary=test_boundary
        )

        ensem_drag = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.drag_config[0]),
            n_estimators=self.drag_config[1],
            random_state=self.drag_config[2],
        )
        ensem_avg_temp = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.avg_temp_config[0]),
            n_estimators=self.avg_temp_config[1],
            random_state=self.avg_temp_config[2],
        )
        ensem_max_temp = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=self.max_temp_config[0]),
            n_estimators=self.max_temp_config[1],
            random_state=self.max_temp_config[2],
        )

        ensem_output_drag = output_matrix[:, 0]
        ensem_output_avg_temp = output_matrix[:, 1]
        ensem_output_max_temp = output_matrix[:, 2]

        ensem_drag.fit(input_matrix, ensem_output_drag)
        ensem_avg_temp.fit(input_matrix, ensem_output_avg_temp)
        ensem_max_temp.fit(input_matrix, ensem_output_max_temp)

        self._model = (ensem_drag, ensem_avg_temp, ensem_max_temp)

"""
GPRModelTrainer class which is responsible for training a Gaussian Process Regressor (GPR) model.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from src.prediction.model_trainer import ModelTrainer

# pylint: disable=invalid-name


class GPRModelTrainer(
    ModelTrainer[
        tuple[
            GaussianProcessRegressor,
            GaussianProcessRegressor,
            GaussianProcessRegressor,
        ]
    ]
):
    """
    GPR model trainer class.
    """

    def __init__(
        self,
        # gpr configs ------ start
        gpr_kernel: Kernel,
        gpr_drag_config: tuple[int, float],
        gpr_avg_temp_config: tuple[int, float],
        gpr_max_temp_config: tuple[int, float],
        # grid configs ------ end
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
        desired_variance: float = 0.95,
    ) -> None:
        super().__init__(
            "GPR",
            grid_scale,
            grid_resolution,
            grid_bound_width,
            grid_bound,
            desired_variance,
        )

        self.kernel = gpr_kernel
        self.drag_config = gpr_drag_config
        self.avg_temp_config = gpr_avg_temp_config
        self.max_temp_config = gpr_max_temp_config

    def train_model(
        self,
    ) -> tuple[
        GaussianProcessRegressor, GaussianProcessRegressor, GaussianProcessRegressor
    ]:
        """
        Trains a Gaussian Process Regressor (GPR) model.

        Returns:
            - tuple[GaussianProcessRegressor, GaussianProcessRegressor, GaussianProcessRegressor]: The trained GPR model.
            - `(GPR for drag, GPR for average temperature, GPR for maximum temperature)`
        """
        input_matrix, output_matrix = self.get_train_set(use_original_input=False)

        gpr_drag = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.drag_config[0],
            alpha=self.drag_config[1],
        )
        gpr_avg_temp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.avg_temp_config[0],
            alpha=self.avg_temp_config[1],
        )
        gpr_max_temp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.max_temp_config[0],
            alpha=self.max_temp_config[1],
        )

        gpr_output_drag = output_matrix[:, 0]
        gpr_output_avg_temp = output_matrix[:, 1]
        gpr_output_max_temp = output_matrix[:, 2]

        gpr_drag.fit(input_matrix, gpr_output_drag)
        gpr_avg_temp.fit(input_matrix, gpr_output_avg_temp)
        gpr_max_temp.fit(input_matrix, gpr_output_max_temp)

        self._model = (gpr_drag, gpr_avg_temp, gpr_max_temp)

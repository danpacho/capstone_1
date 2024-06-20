from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Generic, TypeVar
from scipy.stats import spearmanr

ModelType = TypeVar("ModelType")


class ModelMetrics(Generic[ModelType]):
    def __init__(self, test_kit) -> None:
        """
        Initializes the ModelMetrics.
        """
        # if not isinstance(test_kit, np.ndarray):
        #     raise NotImplementedError
        self.test_kit = test_kit

    def drag_metrics(self, drag_pred) -> tuple[float, float, float]:
        """
        Calculates the Spearman correlation, p-value and R^2 score of the drag prediction.

        Returns:
        rho: float
            The Spearman correlation
        p_val: float
            The p-value of the Spearman correlation
        r2: float
            The R^2 score
        """
        rho, p_val = spearmanr(self.test_kit[:, 0], drag_pred)
        r2 = r2_score(self.test_kit[:, 0], drag_pred)
        return rho, p_val, r2

    def ave_temp_metrics(self, ave_temp_pred) -> tuple[float, float, float]:
        """
        Calculates the Spearman correlation, p-value and Mean Squared Error of the average temperature prediction.

        Returns:
        rho: float
            The Spearman correlation
        p_val: float
            The p-value of the Spearman correlation
        mse: float
            The Mean Squared Error
        """
        rho, p_val = spearmanr(self.test_kit[:, 1], ave_temp_pred)
        mse = mean_squared_error(self.test_kit[:, 1], ave_temp_pred)
        return rho, p_val, mse

    def max_temp_metrics(self, max_temp_pred) -> float:
        """
        Calculates the Spearman correlation, p-value and Mean Absolute Error of the max temperature prediction.

        Returns:
        rho: float
            The Spearman correlation
        p_val: float
            The p-value of the Spearman correlation
        mae: float
            The Mean Absolute Error
        """
        rho, p_val = spearmanr(self.test_kit[:, 2], max_temp_pred)
        mae = mean_absolute_error(self.test_kit[:, 2], max_temp_pred)
        return rho, p_val, mae

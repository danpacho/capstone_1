from abc import abstractmethod
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from typing import Generic, TypeVar

ModelType = TypeVar("ModelType")

class ModelMetrics(Generic[ModelType]):
    def __init__(
            self,
            test_kit
        )-> None:
        """
        Initializes the ModelMetrics.
        """
        # if not isinstance(test_kit, np.ndarray):
        #     raise NotImplementedError
        self.test_kit = test_kit

    def drag_metrics(self, drag_pred) -> float:
        return r2_score(self.test_kit[:, 0], drag_pred)
    
    def ave_temp_metrics(self, ave_temp_pred) -> float:
        return mean_squared_error(self.test_kit[:, 1], ave_temp_pred)

    def max_temp_metrics(self, max_temp_pred) -> float:
        return mean_absolute_error(self.test_kit[:, 2], max_temp_pred)
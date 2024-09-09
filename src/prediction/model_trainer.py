"""
ModelTrainer, a class that trains a model based on the provided data.
"""

from abc import abstractmethod
from typing import Union

import hashlib
import uuid
import os


class ModelTrainer:
    """
    ModelTrainer, a class that trains a model based on the provided data.

    Attributes:
        model_name (str): Name of the model.
        data_path (str): Path to the data folder.
        train_path (str): Path to the training folder.
        grid_scale (float): Scale of the grid.
        grid_resolution (float): Resolution of the grid.
        grid_bound_width (float): Width of the grid boundary.
        grid_bound (Union[tuple[tuple[float, float], tuple[float, float]], None]): Boundaries of the grid.
        train_config (dict[str, str]): Training configuration.
        train_id (str): Training ID.
    """

    root_path: str = os.path.join(os.getcwd(), "model")
    """
    Root path of the model trainer, `cwd()/model`.
    """

    def __init__(
        self,
        model_name: str,
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: Union[tuple[tuple[float, float], tuple[float, float]], None] = None,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            model_name (`str`): Name of the model.
            grid_scale (`float`): Scale of the grid.
            grid_resolution (`float`): Resolution of the grid.
            grid_bound_width (`float`): Width of the grid boundary.
            grid_bound (`Union[tuple[tuple[float, float], tuple[float, float]], None]`): Boundaries of the grid.
        """
        ModelTrainer._box_title(f"Model Trainer: {model_name}")

        self.model_name = model_name.lower()

        self.data_path = os.path.join(ModelTrainer.root_path, "data")
        self.train_path = os.path.join(ModelTrainer.root_path, "train", self.model_name)

        self.grid_scale = grid_scale
        self.grid_resolution = grid_resolution
        self.grid_bound_width = grid_bound_width
        self.grid_bound = grid_bound

        self.train_config: dict[str, str] = {
            "grid_scale": str(self.grid_scale),
            "grid_resolution": str(self.grid_resolution),
            "grid_bound_width": str(self.grid_bound_width),
            "grid_bound": str(self.grid_bound) if self.grid_bound else "None",
        }
        seed_str = "".join(self.train_config.values())
        self.train_id: str = ModelTrainer._generate_uuid_from_seed(seed_str)

    @abstractmethod
    def train_model(
        self,
        test_boundary: Union[tuple[int, int], None] = None,
    ) -> None:
        """
        Trains the model.
        """
        raise NotImplementedError

    @staticmethod
    def _generate_uuid_from_seed(seed_string: str) -> str:
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes[:16]))

    @staticmethod
    def _box_title(title: str) -> None:
        """
        Prints a boxed title for logging purposes.

        Args:
            title (str): The title to be printed.
        """
        log = f"| [ModelTrainer]: {title} |"
        log_len = len(log)
        print("-" * log_len)
        print(log)
        print("-" * log_len)

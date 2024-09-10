import torch
import torch.nn as nn
from src.prediction.model_trainer import ModelTrainer


class CNNTrainer(ModelTrainer):
    """
    CNN Trainer
    """

    def __init__(
        self,
        # Model
        model: nn.Module,
        model_name: str,
        model_device: str,
        model_load_path: str,
        # Grid
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        super().__init__(
            model_name, grid_scale, grid_resolution, grid_bound_width, grid_bound
        )
        self.model = model
        self.model_device = model_device
        self.model_load_path = model_load_path

    def train_model(self, _: tuple[int, int] | None = None) -> None:
        self.load_model()

    def load_model(self) -> None:
        self.model.to(self.model_device)
        self.model.load_state_dict(torch.load(self.model_load_path))
        self.model.eval()
        print("Loaded the best model based on validation loss.")

    def get_model(self):
        return self.model

from dataclasses import dataclass
import numpy as np


@dataclass
class GridCell:
    """
    Grid cell to store the grid unit
    """

    k: int
    coord: np.ndarray[np.float64]


class Grid:
    """
    Grid class to generate a grid of cells
    """

    def __init__(
        self, k: float, bound: tuple[tuple[int, int], tuple[int, int]]
    ) -> None:
        self.k: float = k
        self.bound_x: tuple[int, int] = bound[0]
        self.bound_y: tuple[int, int] = bound[1]
        self.grid_cells: list[GridCell] = []

    def generate_grid(self) -> list[GridCell]:
        grid: list[GridCell] = []
        cell_x_count = int((self.bound_x[1] - self.bound_x[0]) / self.k)
        cell_y_count = int((self.bound_y[1] - self.bound_y[0]) / self.k)

        for i in range(cell_x_count):
            for j in range(cell_y_count):
                x = self.bound_x[0] + self.k * i
                y = self.bound_y[0] + self.k * j
                coord = np.array([x, y])
                grid.append(GridCell(k=self.k, coord=coord))

        self.grid_cells = grid
        return grid

    def fit_to_grid(
        self, arbitrary_points: list[np.ndarray[np.float64]]
    ) -> list[GridCell]:
        grid: list[GridCell] = []
        coord_set: set[str] = set()

        for point in arbitrary_points:
            x, y = point
            fitted_x = (x // self.k) * self.k
            fitted_y = (y // self.k) * self.k
            id = f"{fitted_x}_{fitted_y}"

            if id in coord_set:
                continue
            coord_set.add(id)

            coord = np.array([fitted_x, fitted_y])
            grid.append(GridCell(k=self.k, coord=coord))

        return grid

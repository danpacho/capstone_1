from dataclasses import dataclass
import numpy as np


@dataclass
class GridCell:
    """
    Grid cell to store the grid unit
    """

    k: int
    coord: np.ndarray[np.float64]

    @property
    def x(self) -> float:
        return self.coord[0]

    @property
    def y(self) -> float:
        return self.coord[1]

    @property
    def coord_list(self) -> list[float]:
        """
        Get the coordinate as a `list`
        """
        return self.coord.tolist()

    @property
    def id(self) -> str:
        """
        Get the id of the grid cell
        """
        return f"{self.x}_{self.y}"

    def __repr__(self) -> str:
        return f"GridCell(k={self.k}, coord={self.coord})"


class Grid:
    """
    Grid class to generate a grid of cells
    """

    def __init__(
        self, k: float, bound: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        self.k: float = k
        self.bound_x: tuple[float, float] = bound[0]
        self.bound_y: tuple[float, float] = bound[1]
        self.grid_cells: list[GridCell] = []

    def generate_grid(self) -> list[GridCell]:
        """
        Generate grid cells

        Returns:
            list[GridCell] - grid cells
        """
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

    @staticmethod
    def discretize_points(
        points: list[np.ndarray[np.float64]],
        k: float,
    ) -> list[np.ndarray[np.float64]]:
        """
        Discretize points into grid cells

        Args:
            points: list[np.ndarray[np.float64]] - arbitrary points
            k: float - grid cell size
        """
        discretized_points = []
        coord_set: set[str] = set()

        for point in points:
            x, y = point
            fitted_x = (x // k) * k
            fitted_y = (y // k) * k

            def add_coord_to_discretized_points(coord) -> bool:
                coord_id = f"{coord[0]}_{coord[1]}"
                if coord_id in coord_set:
                    return False

                coord_set.add(coord_id)
                discretized_points.append(coord)
                return True

            target_coords: list[list[float, float]] = [
                [fitted_x, fitted_y],
                [fitted_x + k, fitted_y],  # compensate for the missing points
            ]

            for target_coord in target_coords:
                add_coord_to_discretized_points(np.array(target_coord))

        return discretized_points

    def fit_to_grid(
        self, arbitrary_points: list[np.ndarray[np.float64]]
    ) -> list[GridCell]:
        fitted_points = Grid.discretize_points(arbitrary_points, self.k)
        grid_cells = [GridCell(k=self.k, coord=point) for point in fitted_points]
        return grid_cells

    def __repr__(self) -> str:
        return f"Grid(k={self.k}, bound_x={self.bound_x}, bound_y={self.bound_y})"

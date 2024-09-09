from dataclasses import dataclass
import numpy as np

from src.geometry.vector import V, V2_group


@dataclass
class GridCell:
    """
    Grid cell to store the grid unit
    """

    k: int
    coord: V2_group

    @property
    def x(self) -> float:
        """
        Get the x coordinate
        """
        return self.coord[0]

    @property
    def y(self) -> float:
        """
        Get the y coordinate
        """
        return self.coord[1]

    @property
    def coord_list(self) -> list[float]:
        """
        Get the coordinate as a `list`
        """
        return self.coord.tolist()

    @property
    def grid_id(self) -> str:
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
        self.grid_matrix: V2_group = V.initialize_matrix_2d()

    def generate_grid(
        self, scale: float = 1, x_major_iteration: bool = False
    ) -> np.ndarray[np.float64]:
        """
        Generate grid vector

        Args:
            scale: float - scale factor
            x_major_iteration: bool - iterate x first

        Returns:
            Gv - grid cell vector
        """
        grid_matrix = np.empty((0, 2), dtype=np.float64)
        cell_x_count = int((self.bound_x[1] - self.bound_x[0]) / self.k)
        cell_y_count = int((self.bound_y[1] - self.bound_y[0]) / self.k)

        if x_major_iteration:
            for i in range(cell_y_count):
                for j in range(cell_x_count):
                    x = self.bound_x[0] + self.k * j
                    y = self.bound_y[0] + self.k * i
                    coord_v: np.ndarray[np.float64] = np.array(
                        [x * scale, y * scale], dtype=np.float64
                    )
                    grid_matrix = np.append(grid_matrix, [coord_v], axis=0)
            self.grid_matrix = grid_matrix
            return grid_matrix

        for i in range(cell_x_count):
            for j in range(cell_y_count):
                x = self.bound_x[0] + self.k * i
                y = self.bound_y[0] + self.k * j
                coord_v: np.ndarray[np.float64] = np.array(
                    [x * scale, y * scale], dtype=np.float64
                )
                grid_matrix = np.append(grid_matrix, [coord_v], axis=0)

        self.grid_matrix = grid_matrix
        return grid_matrix

    @staticmethod
    def discretize_points(
        points: V2_group,
        k: float,
    ) -> V2_group:
        """
        Discretize points into grid vector

        Args:
            points: Gv - arbitrary points
            k: float - grid cell size
        """
        discretized_matrix = V.initialize_matrix_2d()
        coord_set: set[str] = set()

        for point in points:
            x, y = point
            fitted_x = (x // k) * k
            fitted_y = (y // k) * k

            def add_coord_to_discretized_points(
                coord: V2_group, discretized_points: V2_group
            ) -> V2_group:
                coord_id = f"{coord[0]}_{coord[1]}"
                if coord_id in coord_set:
                    return discretized_points

                coord_set.add(coord_id)
                discretized_points = V.append_v2(discretized_points, coord)
                return discretized_points

            fitting_vec_groups: V2_group = np.array(
                [
                    [fitted_x, fitted_y],
                    [fitted_x + k, fitted_y],  # compensate for the missing points
                ]
            )

            for fitting_vec in fitting_vec_groups:
                discretized_matrix = add_coord_to_discretized_points(
                    np.array(fitting_vec), discretized_matrix
                )

        return discretized_matrix

    def fit_to_grid(self, arbitrary_points: V2_group) -> V2_group:
        """
        Fit arbitrary points to grid cells
        """
        fitted_points: V2_group = Grid.discretize_points(arbitrary_points, self.k)
        return fitted_points

    def __repr__(self) -> str:
        return f"Grid(k={self.k}, bound_x={self.bound_x}, bound_y={self.bound_y})"

    @property
    def grid_cell(self) -> list[GridCell]:
        """
        Get the grid cells as a list
        """
        return [GridCell(self.k, coord) for coord in self.grid_matrix]

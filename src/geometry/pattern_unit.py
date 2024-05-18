import numpy as np

from math import ceil
from typing import Callable, Literal

from src.grid.grid import Grid, GridCell


class Shape:
    """
    shape class
    args:
        bbox: shape bounding box, ((x_min, x_max), (y_min, y_max))
        area_functions: list[Callable[[float, float], bool]] - area functions
    """

    def __init__(
        self,
        w: float,
        h: float,
        area_functions: list[Callable[[float, float], bool]],
    ) -> None:
        self.w = w
        self.h = h
        self.bbox = ((-w / 2, w / 2), (-h / 2, h / 2))
        self.area_function = area_functions

    @property
    def bbox_x(self) -> tuple[float, float]:
        return self.bbox[0]

    @property
    def bbox_y(self) -> tuple[float, float]:
        return self.bbox[1]

    def __repr__(self) -> str:
        return f"Shape(bbox={self.bbox})"


class PatternUnit:
    """
    pattern unit
    args:
        P: np.ndarray[np.float64] - pattern unit vector
        d: float - pattern unit origin distance
        grid: Grid - grid generator
    """

    def __init__(
        self,
        shape: Shape,
        k: float,
    ) -> None:
        self.shape = shape
        # initialize grid
        self.grid = Grid(k=k, bound=(shape.bbox))
        self.grid.generate_grid()

        self.shape_points: list[GridCell] = []
        self.generate_shape_points()

    def generate_shape_points(self, deduplicate: bool = True) -> None:
        # reset shape points
        self.shape_points = []

        shape_point_ids: set[str] = set()
        for area_f in self.shape.area_function:
            for cell in self.grid.grid_cells:
                if area_f(cell.x, cell.y) and cell.id:
                    if deduplicate:
                        if cell.id in shape_point_ids:
                            continue
                        shape_point_ids.add(cell.id)
                        self.shape_points.append(cell)
                    else:
                        self.shape_points.append(cell)

    @property
    def w(self) -> float:
        return self.shape.bbox_x[1] - self.shape.bbox_x[0]

    @property
    def h(self) -> float:
        return self.shape.bbox_y[1] - self.shape.bbox_y[0]

    def __repr__(self) -> str:
        return f"PatternUnit(shape={self.shape}, grid={self.grid})"


class Transformer:
    @staticmethod
    def translate_x(
        points: list[np.ndarray[np.float64]],
        dx: float,
        is_3dim: bool = True,
    ) -> list[np.ndarray[np.float64]]:
        return [
            point + np.array([dx, 0, 0] if is_3dim else [dx, 0]) for point in points
        ]

    @staticmethod
    def translate_y(
        points: list[np.ndarray[np.float64]],
        dy: float,
        is_3dim: bool = True,
    ) -> list[np.ndarray[np.float64]]:
        return [
            point + np.array([0, dy, 0] if is_3dim else [0, dy]) for point in points
        ]

    @staticmethod
    def rotate(
        points: list[np.ndarray[np.float64]], angle: float, is_3dim: bool = True
    ) -> list[np.ndarray[np.float64]]:
        return [
            np.array(
                [
                    point[0] * np.cos(angle) - point[1] * np.sin(angle),
                    point[0] * np.sin(angle) + point[1] * np.cos(angle),
                    angle,
                ]
                if is_3dim
                else [
                    point[0] * np.cos(angle) - point[1] * np.sin(angle),
                    point[0] * np.sin(angle) + point[1] * np.cos(angle),
                ]
            )
            for point in points
        ]

    @staticmethod
    def mirror_x(
        points: list[np.ndarray[np.float64]],
        is_3dim: bool = True,
    ) -> list[np.ndarray[np.float64]]:
        return [
            np.array(
                [
                    -point[0],
                    point[1],
                    point[2],
                ]
                if is_3dim
                else [
                    -point[0],
                    point[1],
                ]
            )
            for point in points
        ]

    @staticmethod
    def mirror_y(
        points: list[np.ndarray[np.float64]],
        is_3dim: bool = True,
    ) -> list[np.ndarray[np.float64]]:
        return [
            np.array(
                [
                    point[0],
                    -point[1],
                    point[2],
                ]
                if is_3dim
                else [
                    point[0],
                    -point[1],
                ]
            )
            for point in points
        ]

    @staticmethod
    def transform(
        points: list[np.ndarray[np.float64]],
        dx: float,
        dy: float,
        angle: float,
        is_3dim: bool = True,
    ) -> list[np.ndarray[np.float64]]:
        return Transformer.translate_y(
            Transformer.translate_x(
                Transformer.rotate(points, angle, is_3dim), dx, is_3dim
            ),
            dy,
            is_3dim,
        )


class PatternTransformation:
    """
    Pattern transformation information

    Dx: float - translation in x direction
    Dy: float - translation in y direction
    Phi: float - rotation angle

    Od: float - distance from origin `(0, 0)`
    """

    default = 0

    def __init__(
        self,
        Dx: float = default,
        Dy: float = default,
        Phi: float = default,
        Rotate_count: int = default,
        Od: float = default,
    ) -> None:
        self.Dx = Dx
        self.Dy = Dy
        self.Phi = Phi
        self.Rotate_count = Rotate_count
        self.Od = Od

    def is_safe_rotation(self, h: float) -> bool:
        if self.Phi == self.default:
            return True

        alpha = np.arctan(h / (2 * self.Od))
        min_phi = np.arctan(2 * alpha)
        print(f"min_phi: {min_phi}, Phi: {self.Phi}")
        return self.Phi >= min_phi

    def min_phi(self, h: float) -> float:
        if self.Od == self.default:
            return self.default
        alpha = np.arctan(h / (2 * self.Od))
        return np.arctan(2 * alpha)

    # TODO: Refactor it with parameterized function f(t) = [x(t), y(t)]
    @property
    def pattern_type(
        self,
    ) -> Literal["circular", "grid", "corn"]:
        if (
            self.Dx != self.default
            and self.Phi != self.default
            and self.Rotate_count == self.default
        ):
            return "circular"

        if (
            self.Dx != self.default
            and self.Phi != self.default
            and self.Rotate_count != self.default
        ):
            return "corn"

        if self.Dx != self.default and self.Dy != self.default:
            return "grid"

    def __repr__(self) -> str:
        return f"PatternTransformation(dx={self.Dx}, dy={self.Dy}, phi={self.Phi}, d={self.Od})"


class PatternTransformationVector:
    """
    Pattern transformation vector
    """

    def __init__(
        self,
        pattern_unit: PatternUnit,
        pattern_transformation: PatternTransformation,
        pattern_bound: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        self.pattern_unit = pattern_unit
        self.pattern_transformation = pattern_transformation
        self.pattern_bound = pattern_bound

        self.Step: float = pattern_unit.w + pattern_transformation.Dx

        self.L: list[np.ndarray[np.float64]] = []
        self.generate_pattern_origin_vector()

    @property
    def p_bound_x(self) -> tuple[float, float]:
        return self.pattern_bound[0]

    @property
    def p_bound_y(self) -> tuple[float, float]:
        return self.pattern_bound[1]

    @property
    def p_bound_x_min(self) -> float:
        return self.p_bound_x[0]

    @property
    def p_bound_x_max(self) -> float:
        return self.p_bound_x[1]

    @property
    def p_bound_y_min(self) -> float:
        return self.p_bound_y[0]

    @property
    def p_bound_y_max(self) -> float:
        return self.p_bound_y[1]

    def generate_pattern_origin_vector(self) -> None:
        L_vector: list[np.ndarray[np.float64]] = []

        # linear transformation (for x > 0)
        L_positive: list[np.ndarray[np.float64]] = []

        base_step = self.pattern_unit.w / 2 + self.pattern_transformation.Od
        positive_pattern_count = (
            round((self.p_bound_x_max - base_step) // self.Step) + 1
        )

        for i in range(positive_pattern_count):
            dx = i * self.Step + base_step
            positive_dy = 0
            phi = 0

            L_positive.append(np.array([dx, positive_dy, phi]))

        # rotation transformation (for 0 < phi < 2*pi)
        L_rotation: list[np.ndarray[np.float64]] = []
        # translation transformation (for p_bound_y_min <= dy <= p_bound_y_max)
        L_translation: list[np.ndarray[np.float64]] = []

        if self.pattern_transformation.pattern_type == "circular":
            if not self.pattern_transformation.is_safe_rotation(self.pattern_unit.h):
                raise ValueError(
                    "[ERROR] Rotation crash detected. Please check the rotation angle."
                )

            rotation_group: list[float] = [
                (i + 1) * self.pattern_transformation.Phi
                for i in range(int(2 * np.pi / self.pattern_transformation.Phi) - 1)
            ]
            for angle in rotation_group:
                L_rotation.extend(
                    [
                        np.array([dx, dy, angle])
                        for dx, dy, _ in Transformer.transform(L_positive, 0, 0, angle)
                    ]
                )

        elif self.pattern_transformation.pattern_type == "corn":
            if not self.pattern_transformation.is_safe_rotation(self.pattern_unit.h):
                raise ValueError(
                    "[ERROR] Rotation crash detected. Please check the rotation angle."
                )
            corn_rotation_group: list[float] = []
            for i in range(round((self.pattern_transformation.Rotate_count) / 2)):
                corn_rotation_group.extend(
                    [
                        (i + 1) * self.pattern_transformation.Phi,
                        -1 * (i + 1) * self.pattern_transformation.Phi,
                    ]
                )
            for angle in corn_rotation_group:
                L_rotation.extend(
                    [
                        np.array([dx, dy, angle])
                        for dx, dy, _ in Transformer.rotate(L_positive, angle)
                    ]
                )

        elif self.pattern_transformation.pattern_type == "grid":
            # X-axis mirror
            L_positive.extend(Transformer.mirror_x(L_positive))

            step_y = self.pattern_unit.h + self.pattern_transformation.Dy
            grid_iter = (
                ceil(((self.p_bound_y_max - self.p_bound_y_min) / 2) / step_y) - 1
            )
            for i in range(grid_iter):
                positive_dy = (i + 1) * step_y
                negative_dy = -1 * (i + 1) * step_y
                L_translation.extend(Transformer.translate_y(L_positive, positive_dy))
                L_translation.extend(Transformer.translate_y(L_positive, negative_dy))

        # Collect L groups
        L_vector.extend(L_positive)
        L_vector.extend(L_rotation)
        L_vector.extend(L_translation)

        self.L = L_vector

    def __repr__(self) -> str:
        return f"PatternTransformationVector(pattern_unit={self.pattern_unit}, pattern_transformation={self.pattern_transformation}, pattern_bound={self.pattern_bound})"


class Pattern:
    def __init__(
        self,
        pattern_unit: PatternUnit,
        pattern_transformation_vector: PatternTransformationVector,
    ) -> None:
        self.pattern_unit = pattern_unit
        self.pattern_transformation_vector = pattern_transformation_vector

        self.pattern_points: list[GridCell] = []
        self.generate_pattern_points()

    def generate_pattern_points(self) -> None:
        self.pattern_points: list[GridCell] = []

        for transform_L in self.pattern_transformation_vector.L:
            transformed_points = Grid.discretize_points(
                Transformer.transform(
                    [cell.coord for cell in self.pattern_unit.shape_points],
                    transform_L[0],  # dx
                    transform_L[1],  # dy
                    transform_L[2],  # angle,
                    is_3dim=False,
                ),
                self.pattern_unit.grid.k,
            )
            self.pattern_points.extend(
                [
                    GridCell(k=self.pattern_unit.grid.k, coord=point)
                    for point in transformed_points
                ]
            )

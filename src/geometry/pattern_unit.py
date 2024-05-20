from math import ceil
from typing import Callable, Literal, Union
from dataclasses import dataclass

import numpy as np

from src.geometry.vector import V, V2_group, V3_group
from src.grid.grid import Grid


class Shape:
    """
    shape class
    args:
        w: float - bounding box width, bbox origin is (0, 0)
        h: float - bounding box height, bbox origin is (0, 0)
        area_functions: list[Callable[[float, float], bool]] - area functions that define the shape, should be form an closed area
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

        self.shape_matrix: V2_group = V.initialize_matrix_2d()
        self.generate_shape_matrix()

    def generate_shape_matrix(self, deduplicate: bool = True) -> None:
        """
        Generate shape points
        """
        # reset shape points
        self.shape_matrix: V2_group = V.initialize_matrix_2d()

        shape_point_ids: set[str] = set()
        for area_f in self.shape.area_function:
            for cell in self.grid.grid_matrix:
                x = cell[0]
                y = cell[1]
                cell_id = f"{x}_{y}"
                if area_f(x, y) and cell_id:
                    if deduplicate:
                        if cell_id in shape_point_ids:
                            continue
                        shape_point_ids.add(cell_id)
                        self.shape_matrix = V.append_v2(self.shape_matrix, cell)
                    else:
                        self.shape_matrix = V.append_v2(self.shape_matrix, cell)

    def update_area_functions(
        self, area_functions: list[Callable[[float, float], bool]]
    ) -> None:
        """
        Update the area functions and regenerate the shape matrix
        """
        self.shape.area_function = area_functions
        self.generate_shape_matrix()

    @property
    def w(self) -> float:
        """
        width of the shape, `(-w/2, w/2)`
        """
        return self.shape.bbox_x[1] - self.shape.bbox_x[0]

    @property
    def h(self) -> float:
        """
        height of the shape, `(-h/2, h/2)`
        """
        return self.shape.bbox_y[1] - self.shape.bbox_y[0]

    def __repr__(self) -> str:
        return f"PatternUnit(shape={self.shape}, grid={self.grid})"


class Transformer:
    @staticmethod
    def translate_x(
        T_matrix: Union[V2_group, V3_group],
        dx: float,
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        translated_points = [
            (
                V.vec3(
                    point[0] + dx,
                    point[1],
                    point[2],
                )
                if is_3dim
                else V.vec2(
                    point[0] + dx,
                    point[1],
                )
            )
            for point in T_matrix
        ]
        return np.array(translated_points)

    @staticmethod
    def translate_y(
        T_matrix: Union[V2_group, V3_group],
        dy: float,
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        translated_points = [
            (
                V.vec3(
                    point[0],
                    point[1] + dy,
                    point[2],
                )
                if is_3dim
                else V.vec2(
                    point[0],
                    point[1] + dy,
                )
            )
            for point in T_matrix
        ]
        return np.array(translated_points)

    @staticmethod
    def rotate(
        T_matrix: Union[V2_group, V3_group], angle: float, is_3dim: bool = True
    ) -> Union[V2_group, V3_group]:
        rotated_points = [
            (
                V.vec3(
                    point[0] * np.cos(angle) - point[1] * np.sin(angle),
                    point[0] * np.sin(angle) + point[1] * np.cos(angle),
                    point[2],
                )
                if is_3dim
                else V.vec2(
                    point[0] * np.cos(angle) - point[1] * np.sin(angle),
                    point[0] * np.sin(angle) + point[1] * np.cos(angle),
                )
            )
            for point in T_matrix
        ]

        return np.array(rotated_points)

    @staticmethod
    def mirror_x(
        T_matrix: Union[V2_group, V3_group],
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        mirrored_points = [
            (
                V.vec3(
                    -point[0],
                    point[1],
                    point[2],
                )
                if is_3dim
                else V.vec2(
                    -point[0],
                    point[1],
                )
            )
            for point in T_matrix
        ]
        return np.array(mirrored_points)

    @staticmethod
    def mirror_y(
        T_matrix: Union[V2_group, V3_group],
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        mirrored_points = [
            (
                V.vec3(
                    point[0],
                    -point[1],
                    point[2],
                )
                if is_3dim
                else V.vec2(
                    point[0],
                    -point[1],
                )
            )
            for point in T_matrix
        ]
        return np.array(mirrored_points)

    @staticmethod
    def transform_rt(
        T_matrix: Union[V2_group, V3_group],
        dx: float,
        dy: float,
        angle: float,
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        """
        Translation -> Rotation transformation
        """
        return Transformer.translate_y(
            Transformer.translate_x(
                Transformer.rotate(T_matrix, angle, is_3dim), dx, is_3dim
            ),
            dy,
            is_3dim,
        )


@dataclass
class PatternTransformation:
    """
    Pattern transformation information

    di: float - distance from origin `(0, 0)`

    dx: float - translation in x direction
    dy: float - translation in y direction
    phi: float - rotation angle
    """

    DEFAULT_VALUE: int = 0

    dx: float = DEFAULT_VALUE
    dy: float = DEFAULT_VALUE
    di: float = DEFAULT_VALUE
    phi: float = DEFAULT_VALUE
    rot_count: int = DEFAULT_VALUE

    def is_safe_rotation(self, h: float) -> bool:
        """
        Check if the rotation is safe, via `min_phi` function
        """
        if self.phi == self.DEFAULT_VALUE:
            return True

        min_phi = self.min_phi(h)
        print(f"min_phi: {min_phi}, Phi: {self.phi}")
        return self.phi >= min_phi

    def min_phi(self, h: float) -> float:
        """
        Get the minimum rotation angle
        """
        if self.di == self.DEFAULT_VALUE:
            return self.DEFAULT_VALUE
        alpha = np.arctan(h / (2 * self.di))
        return np.arctan(2 * alpha)

    # TODO: Refactor it with parameterized function f(t) = [x(t), y(t)]
    @property
    def pattern_type(
        self,
    ) -> Literal["circular", "grid", "corn"]:
        """
        Determine the pattern type
        """
        if (
            self.dx != self.DEFAULT_VALUE
            and self.phi != self.DEFAULT_VALUE
            and self.rot_count == self.DEFAULT_VALUE
        ):
            return "circular"

        if (
            self.dx != self.DEFAULT_VALUE
            and self.phi != self.DEFAULT_VALUE
            and self.rot_count != self.DEFAULT_VALUE
        ):
            return "corn"

        if self.dx != self.DEFAULT_VALUE and self.dy != self.DEFAULT_VALUE:
            return "grid"

    def __repr__(self) -> str:
        return f"PatternTransformation(dx={self.dx}, dy={self.dy}, phi={self.phi}, d={self.di})"


class PatternTransformationMatrix:
    """
    Pattern transformation matrix group T

    Attributes:
    pattern_unit: `PatternUnit`
        The pattern unit, shape
    pattern_transformation: `PatternTransformation`
        The pattern transformation information
    pattern_bound: `tuple[tuple[float, float], tuple[float, float]]`
        The pattern bound, (x_min, x_max), (y_min, y_max)
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

        self.p_step: float = pattern_unit.w + pattern_transformation.dx

        self.T_matrix: V3_group = V.initialize_matrix_3d()
        self.generate_pattern_transformation_matrix()

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

    def generate_pattern_transformation_matrix(self) -> None:
        """
        Generate the pattern transformation matrix
        - `3D` matrix, where each row is a transformation vector `t_i = [dx_i, dy_i, phi_i]`
        - Generated based on the `PatternTransformation` information
        """
        T_matrix = V.initialize_matrix_3d()

        # linear transformation (for x > 0)
        T_x_positive = V.initialize_matrix_3d()

        base_step = self.pattern_unit.w / 2 + self.pattern_transformation.di
        positive_pattern_count = (
            round((self.p_bound_x_max - base_step) // self.p_step) + 1
        )

        for i in range(positive_pattern_count):
            dx = i * self.p_step + base_step
            positive_dy = 0
            phi = 0

            T_x_positive = V.append_v3(T_x_positive, V.vec3(dx, positive_dy, phi))

        # rotation transformation (for 0 < phi < 2*pi)
        T_rotation = V.initialize_matrix_3d()
        # translation transformation (for p_bound_y_min <= dy <= p_bound_y_max)
        T_translation = V.initialize_matrix_3d()

        if self.pattern_transformation.pattern_type == "circular":
            if not self.pattern_transformation.is_safe_rotation(self.pattern_unit.h):
                raise ValueError(
                    "[ERROR] Rotation crash detected. Please check the rotation angle."
                )

            rotation_group: list[float] = [
                (i + 1) * self.pattern_transformation.phi
                for i in range(int(2 * np.pi / self.pattern_transformation.phi) - 1)
            ]
            for angle in rotation_group:
                T_rotated = Transformer.rotate(T_x_positive, angle)
                T_rotated = np.array([V.vec3(dx, dy, angle) for dx, dy, _ in T_rotated])
                T_rotation = V.combine_mat_v3(T_rotation, T_rotated)

        elif self.pattern_transformation.pattern_type == "corn":
            if not self.pattern_transformation.is_safe_rotation(self.pattern_unit.h):
                raise ValueError(
                    "[ERROR] Rotation crash detected. Please check the rotation angle."
                )
            corn_rotation_group: list[float] = []
            for i in range(round((self.pattern_transformation.rot_count) / 2)):
                corn_rotation_group.extend(
                    [
                        (i + 1) * self.pattern_transformation.phi,
                        -1 * (i + 1) * self.pattern_transformation.phi,
                    ]
                )
            for angle in corn_rotation_group:
                T_rotated = Transformer.rotate(T_x_positive, angle)
                T_rotated = np.array([V.vec3(dx, dy, angle) for dx, dy, _ in T_rotated])
                T_rotation = V.combine_mat_v3(T_rotation, T_rotated)

        elif self.pattern_transformation.pattern_type == "grid":
            # X-axis mirror
            T_x_positive = V.combine_mat_v3(
                T_x_positive, Transformer.mirror_x(T_x_positive)
            )

            step_y = self.pattern_unit.h + self.pattern_transformation.dy
            grid_iter = (
                ceil(((self.p_bound_y_max - self.p_bound_y_min) / 2) / step_y) - 1
            )
            for i in range(grid_iter):
                positive_dy = (i + 1) * step_y
                negative_dy = -1 * (i + 1) * step_y
                T_translation = V.combine_mat_v3(
                    T_translation, Transformer.translate_y(T_x_positive, positive_dy)
                )
                T_translation = V.combine_mat_v3(
                    T_translation, Transformer.translate_y(T_x_positive, negative_dy)
                )

        # Collect L groups
        T_matrix = V.combine_mat_v3(T_matrix, T_x_positive)
        T_matrix = V.combine_mat_v3(T_matrix, T_rotation)
        T_matrix = V.combine_mat_v3(T_matrix, T_translation)

        self.T_matrix = T_matrix

    def __repr__(self) -> str:
        return f"PatternTransformationVector(pattern_unit={self.pattern_unit}, pattern_transformation={self.pattern_transformation}, pattern_bound={self.pattern_bound})"


class Pattern:
    """
    Pattern class

    Attributes:
    pattern_unit: `PatternUnit`
        The pattern unit, shape
    pattern_transformation_vector: `PatternTransformationMatrix`
        The pattern transformation matrix group T
    """

    def __init__(
        self,
        pattern_unit: PatternUnit,
        pattern_transformation_vector: PatternTransformationMatrix,
    ) -> None:
        self.pattern_unit = pattern_unit
        self.pattern_transformation_vector = pattern_transformation_vector

        self.pattern_matrix: V2_group = V.initialize_matrix_2d()
        self.generate_pattern_matrix()

    def generate_pattern_matrix(self) -> None:
        """
        Generate the pattern matrix
        1. pattern_unit's `2D` matrix, where each row is a point `p_i = [x_i, y_i]`
        2. pattern_transformation_matrix's `3D` matrix,
        where each row is a transformation vector `t_i = [dx_i, dy_i, phi_i]`

        Returns:
            Pattern is generated by applying the transformation matrix group `T`
            to the `pattern unit`'s shape matrix
        """
        self.pattern_matrix = V.initialize_matrix_2d()

        for T_vec in self.pattern_transformation_vector.T_matrix:
            transformed_t_vec = Grid.discretize_points(
                Transformer.transform_rt(
                    self.pattern_unit.shape_matrix,
                    T_vec[0],  # dx
                    T_vec[1],  # dy
                    T_vec[2],  # angle,
                    is_3dim=False,
                ),
                self.pattern_unit.grid.k,
            )
            self.pattern_matrix = V.combine_mat_v2(
                self.pattern_matrix, transformed_t_vec
            )

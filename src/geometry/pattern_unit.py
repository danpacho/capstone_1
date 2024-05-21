from math import ceil, floor
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
        width of the shape
        """
        return self.shape.bbox_x[1] - self.shape.bbox_x[0]

    @property
    def h(self) -> float:
        """
        height of the shape
        """
        return self.shape.bbox_y[1] - self.shape.bbox_y[0]

    def __repr__(self) -> str:
        return f"PatternUnit(shape={self.shape}, grid={self.grid})"


class Transformer:
    """
    Transformer class
    """

    @staticmethod
    def translate_x(
        matrix: Union[V2_group, V3_group],
        dx: float,
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        """
        Translate the matrix in x direction
        """
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
            for point in matrix
        ]
        return np.array(translated_points)

    @staticmethod
    def translate_y(
        matrix: Union[V2_group, V3_group],
        dy: float,
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        """
        Translate the matrix in y direction
        """
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
            for point in matrix
        ]
        return np.array(translated_points)

    @staticmethod
    def rotate(
        matrix: Union[V2_group, V3_group], angle: float, is_3dim: bool = True
    ) -> Union[V2_group, V3_group]:
        """
        Rotate the matrix by angle `radian`
        """
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
            for point in matrix
        ]

        return np.array(rotated_points)

    @staticmethod
    def mirror_x(
        matrix: Union[V2_group, V3_group],
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        """
        Mirror the matrix in x direction
        """
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
            for point in matrix
        ]
        return np.array(mirrored_points)

    @staticmethod
    def mirror_y(
        matrix: Union[V2_group, V3_group],
        is_3dim: bool = True,
    ) -> Union[V2_group, V3_group]:
        """
        Mirror the matrix in y direction
        """
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
            for point in matrix
        ]
        return np.array(mirrored_points)

    @staticmethod
    def transform_rt(
        matrix: Union[V2_group, V3_group],
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
                Transformer.rotate(matrix, angle, is_3dim), dx, is_3dim
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
    phi: float - rotation angle, `radian`
    rot_count: int - rotation count, for corn pattern

    DEFAULT_VALUE: int - default value for the attributes
    """

    pattern_type: Union[Literal["grid", "circular", "corn"]]

    DEFAULT_VALUE: Literal[0] = 0

    dx: float = DEFAULT_VALUE
    dy: float = DEFAULT_VALUE
    di: float = DEFAULT_VALUE
    phi: float = DEFAULT_VALUE
    rot_count: int = DEFAULT_VALUE

    @property
    def phi_angle(self) -> float:
        """
        Get the rotation angle in `degree`
        """
        return np.rad2deg(self.phi)

    def min_phi(self, h: float) -> float:
        """
        Get the minimum rotation angle
        """
        if self.di == self.DEFAULT_VALUE:
            return self.DEFAULT_VALUE  # 0deg
        alpha = np.arctan(h / (2 * self.di))
        return np.arctan(2 * alpha)

    def is_safe_rotation(self, h: float) -> bool:
        """
        Check if the rotation is safe, via `min_phi` function
        """
        # Rotation not activated
        if self.phi == self.DEFAULT_VALUE:
            return True

        # Rotation activated & di is 0 then it is collision
        if self.di == self.DEFAULT_VALUE:
            return False

        return self.phi >= self.min_phi(h)

    def fix_rotation_count(self, log: bool = False) -> None:
        """
        Fix the rotation count, for geometrical collision situations
        """
        if self.rot_count == self.DEFAULT_VALUE:
            return

        max_rotated = self.phi * self.rot_count
        if max_rotated >= np.pi:
            self.rot_count = floor(np.pi / self.phi)
            if log:
                print(
                    f"[PatternTransformation]: Fixing rotation count to {self.rot_count}"
                )

    def fix_rotation(
        self, h: float, safe_delta: float = 0.01, log: bool = False
    ) -> None:
        """
        Fix the rotation angle, for geometrical collision situations
        """
        if self.is_safe_rotation(h):
            return

        # Fix invalid > di
        if self.di == self.DEFAULT_VALUE:
            if log:
                print("[PatternTransformation]: Fixing di to 1")
            self.di = 1

        # Fix invalid > phi
        min_phi = self.min_phi(h) + safe_delta

        self.phi = min_phi
        if log:
            print(
                f"[PatternTransformation]: Fixing rotation angle to {self.phi_angle} deg"
            )

    def is_safe_translation(self, safe_delta: float = 0.25) -> bool:
        """
        Check if the translation is safe, via `min_translation` function
        """
        if self.pattern_type == "grid":
            return self.dx >= safe_delta and self.dy >= safe_delta

        return self.dx >= safe_delta and self.di >= safe_delta

    def fix_translation(self, safe_delta: float = 0.25, log: bool = False) -> None:
        """
        Fix the translation, for geometrical collision situations
        """
        if self.is_safe_translation(safe_delta):
            return

        if self.pattern_type == "grid":
            if log:
                print(f"[PatternTransformation]: Fixing translation to {safe_delta}")
            self.dx = safe_delta
            self.dy = safe_delta
        else:
            if log:
                print(f"[PatternTransformation]: Fixing translation to {safe_delta}")
            self.dx = safe_delta
            self.di = safe_delta

    def __repr__(self) -> str:
        return f"PatternTransformation(dx={self.dx}, dy={self.dy}, di={self.di}, phi={self.phi}, rot_count={self.rot_count})"


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
    auto_fix_collision: bool
        If `True`, automatically fix the collision situation
    """

    def __init__(
        self,
        pattern_unit: PatternUnit,
        pattern_transformation: PatternTransformation,
        pattern_bound: tuple[tuple[float, float], tuple[float, float]],
        auto_fix_collision: bool = True,
    ) -> None:
        self.pattern_unit = pattern_unit
        self.pattern_transformation = pattern_transformation
        self.pattern_bound = pattern_bound
        self.auto_fix_collision = auto_fix_collision

        # check collision
        self.resolve_collision()

        self.p_step_x: float = pattern_unit.w + pattern_transformation.dx
        self.p_step_y: float = pattern_unit.h + pattern_transformation.dy

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

    def resolve_collision(self) -> None:
        """
        Resolve the rotation collision problem
        """
        if self.auto_fix_collision:
            # Fix rotation
            rotation_fix = ["circular", "corn"]
            if self.pattern_transformation.pattern_type in rotation_fix:
                self.pattern_transformation.fix_rotation(self.pattern_unit.h, True)

            # Fix rotation count
            if self.pattern_transformation.pattern_type == "corn":
                self.pattern_transformation.fix_rotation_count(True)

            # Fix translation
            self.pattern_transformation.fix_translation()

        else:
            if not self.pattern_transformation.is_safe_rotation(self.pattern_unit.h):
                raise ValueError(
                    f"[PatternTransformationMatrix] rotation angle is not safe, please fix the collision situation, min_phi: {self.pattern_transformation.min_phi(self.pattern_unit.h)} but phi: {self.pattern_transformation.phi}"
                )
            if not self.pattern_transformation.is_safe_translation():
                raise ValueError(
                    f"[PatternTransformationMatrix] translation distance is not safe, please fix the collision situation, dx: {self.pattern_transformation.dx}, dy: {self.pattern_transformation.dy}"
                )

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
            round((self.p_bound_x_max - base_step) // self.p_step_x) + 1
        )

        for i in range(positive_pattern_count):
            dx = i * self.p_step_x + base_step
            positive_dy = 0
            phi = 0

            T_x_positive = V.append_v3(T_x_positive, V.vec3(dx, positive_dy, phi))

        # rotation transformation (for 0 < phi < 2*pi)
        T_rotation = V.initialize_matrix_3d()
        # translation transformation (for p_bound_y_min <= dy <= p_bound_y_max)
        T_translation = V.initialize_matrix_3d()

        if self.pattern_transformation.pattern_type == "circular":
            print(f"circular phi_angle: {self.pattern_transformation.phi_angle}")
            rotation_group: list[float] = [
                (i + 1) * self.pattern_transformation.phi
                for i in range(int(2 * np.pi / self.pattern_transformation.phi) - 1)
            ]
            for angle in rotation_group:
                T_rotated = Transformer.rotate(T_x_positive, angle)
                T_rotated = np.array([V.vec3(dx, dy, angle) for dx, dy, _ in T_rotated])
                T_rotation = V.combine_mat_v3(T_rotation, T_rotated)

        elif self.pattern_transformation.pattern_type == "corn":
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

            grid_iter = (
                ceil(((self.p_bound_y_max - self.p_bound_y_min) / 2) / self.p_step_y)
                - 1
            )
            for i in range(grid_iter):
                positive_dy = (i + 1) * self.p_step_y
                T_translation = V.combine_mat_v3(
                    T_translation, Transformer.translate_y(T_x_positive, positive_dy)
                )
                negative_dy = -1 * (i + 1) * self.p_step_y
                T_translation = V.combine_mat_v3(
                    T_translation, Transformer.translate_y(T_x_positive, negative_dy)
                )

        # Collect L groups
        T_matrix = V.combine_mat_v3(T_matrix, T_x_positive)
        T_matrix = V.combine_mat_v3(T_matrix, T_rotation)
        T_matrix = V.combine_mat_v3(T_matrix, T_translation)

        self.T_matrix = T_matrix

    def __repr__(self) -> str:
        return f"PatternTransformationMatrix(pattern_unit={self.pattern_unit}, pattern_transformation={self.pattern_transformation}, pattern_bound={self.pattern_bound})"


class Pattern:
    """
    Pattern class

    Attributes:
    pattern_transformation_vector: `PatternTransformationMatrix`
        The pattern transformation matrix group T
    """

    def __init__(
        self,
        pattern_transformation_matrix: PatternTransformationMatrix,
    ) -> None:
        self.pattern_transformation_matrix = pattern_transformation_matrix

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

        for T_vec in self.pattern_transformation_matrix.T_matrix:
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

    @property
    def pattern_unit(self) -> PatternUnit:
        """
        Get the pattern unit
        """
        return self.pattern_transformation_matrix.pattern_unit

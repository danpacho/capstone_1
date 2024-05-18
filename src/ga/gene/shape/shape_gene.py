import numpy as np

from dataclasses import dataclass
from typing import Callable

from src.geometry.pattern_unit import Shape, PatternUnit
from src.ga.gene.gene import Gene
from src.storage.vector_storage import VectorStorage
from src.storage.stochastic_storage import StochasticStorage


@dataclass
class ShapeParameter:
    """
    ShapeParameter class
    """

    label: str
    bbox: tuple[float, float, float]
    """
    The bounding box of the shape

    Example:
    ```python
    bbox = (-10, 10, 0.25)
    # x: -10 to 10, y: -10 to 10, k: 0.25
    ```
    """

    a_f: list[Callable[[tuple[float, float], np.ndarray[np.float64]], bool]]
    """
    Area functions for the shape
    
    Example:
    ```python
    a_f = [
        lambda p, params: 
            (p[0] ** 2 + p[1] ** 2) >= params[0] 
            # x^2 + y^2 >= r_inner^2
            and 
            (p[0] ** 2 + p[1] ** 2) <= params[1],
            # x^2 + y^2 <= r_outer^2
    ]
    ```
    """
    parameter_id_list: list[str]
    """
    Unique label for each parameter

    Example:
    ```python
    parameter_ld_list = ["r_inner", "r_outer"]
    ```
    """
    parameter_boundary_list: list[tuple[float, float]]
    """
    The boundary of the parameters

    Example:
    ```python
    parameter_boundary_list = [(2, 6), (7, 10)]
    # 2 <= r_inner <= 6, 7 <= r_outer <= 10
    ```
    """

    def __post_init__(self):
        # Validate label count
        if len(self.label) == 0:
            raise ValueError("Shape id must not be empty.")

        # Validate parameter count
        if self.parameter_count != len(self.parameter_id_list):
            raise ValueError(
                "parameter_count does not match the length of parameter_label_list."
            )

        # Validate parameter boundary list count
        if self.parameter_count != len(self.parameter_boundary_list):
            raise ValueError(
                "parameter_count does not match the length of parameter_boundary_list."
            )

    @property
    def w(self) -> float:
        """
        The width of the bbox
        """
        return self.bbox[0]

    @property
    def h(self) -> float:
        """
        The height of the bbox
        """
        return self.bbox[1]

    @property
    def k(self) -> float:
        """
        The resolution of the bbox, lower k means higher resolution
        """
        return self.bbox[2]

    @property
    def parameter_count(self) -> int:
        """
        The number of parameters
        """
        return len(self.parameter_id_list)

    def get_a_f_with_params(
        self, parameter_list: np.ndarray[np.float64]
    ) -> list[Callable[[tuple[float, float]], bool]]:
        """
        Returns the area functions with the parameters
        """

        def to_af_with_params(
            f: Callable[[tuple[float, float], np.ndarray[np.float64]], bool]
        ) -> Callable[[tuple[float, float]], bool]:
            return lambda x, y: f((x, y), parameter_list)

        # Adapt the area parameter functions to the shape
        return [to_af_with_params(f) for f in self.a_f]

    def __repr__(self) -> str:
        return f"ShapeParameter({self.label}):" + "\n".join(
            [
                f"  {self.parameter_id_list[i]}: {self.parameter_boundary_list[i]}"
                for i in range(len(self.parameter_id_list))
            ]
        )


class ShapeGene(Gene):
    """
    ShapeGene class

    Attributes:
    param: ShapeParameter
        The shape parameter of the gene
    pattern_unit: PatternUnit
        The pattern unit of the gene
    parameter_storage: Storage[dict[str, list[float]]]
        The storage for the gene parameters

        Example:
        ```json
        {
            DonutShape_1_r_inner: [1.0, 2.0, 3.0],
            DonutShape_1_r_outer: [4.0, 5.0, 6.0]
        }
        ```
    """

    parameter_storage: VectorStorage
    """
    The storage for the gene parameters

    Example:
    ```json
    # db.json
    {
        "DonutShape_1": [1.0, 2.0], # parameter list for DonutShape_1
        "DonutShape_2": [4.0, 5.0]  # parameter list for DonutShape_2
    }
    ```
    
    """
    pdf_storage: StochasticStorage
    """
    The storage for the gene parameters pdf

    Example:
    ```json
    # db.json
    {
        "r_inner": [1.1, 0.2, 1.4, 12, 5.5, 6.6, ...],
        "r_outer": [2.2, 0.3, 2.5, 13, 6.5, 7.6, ...]
    }
    """

    def __init__(
        self,
        shape_parameter: ShapeParameter,
    ) -> None:

        super().__init__(
            shape_parameter.label,
            shape_parameter.parameter_count,
            shape_parameter.parameter_boundary_list,
        )

        self.param = shape_parameter
        # Adapt the area parameter functions to the shape
        self.pattern_unit = PatternUnit(
            Shape(shape_parameter.w, shape_parameter.h, shape_parameter.k),
            shape_parameter.get_a_f_with_params(self.parameter_list),
        )

    def print_parameter_info(self) -> None:
        parameter_info_array = "[ "
        for i, parameter in enumerate(self.parameter_list):
            parameter_info_array += f"{self.param.parameter_id_list[i]}: {parameter}"
        parameter_info_array += "]"

        return f"parameters: {parameter_info_array}"

    @property
    def parameter_table(self) -> dict[str, float]:
        """
        Returns the parameters of the gene in a dictionary format

        Example:
        ```python
        parameters = {
            "r1": 0.5, "r2": 0.3, "r3": 0.1
        }
        ```
        """
        return {
            self.param.parameter_id_list[i]: self.parameter_list[i]
            for i in range(len(self.parameter_list))
        }

    def update_gene(self, new_parameter_list: list[float]) -> None:
        # Assume that parameter_list is already updated by `mutate` or `mutate_at`
        # 1. Update parameter storage
        self.parameter_storage.insert_field(self.label, self.parameter_list)
        # 2. Update pdf storage
        for param_id in self.param.parameter_id_list:
            # Delete the old parameter list
            self.pdf_storage.delete_list(param_id, self.parameter_list)
            # Insert new parameter list
            self.pdf_storage.insert_field(param_id, new_parameter_list)

    def load_gene(self) -> None:
        # Load parameters from the parameter storage
        self.parameter_list = self.parameter_storage.inquire(self.label)

    def remove_gene(self) -> None:
        # Remove the gene information
        self.parameter_storage.delete_field(self.label)
        # Remove parameters at the pdf
        for i, param_id in enumerate(self.param.parameter_id_list):
            self.pdf_storage.delete_single(param_id, self.parameter_list[i])

    def mutate(self) -> None:
        raise NotImplementedError

    def mutate_at(self, at: int) -> None:
        raise NotImplementedError

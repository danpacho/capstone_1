from dataclasses import dataclass
from typing import Callable, Literal, Union

import numpy as np

from src.geometry.vector import V2_group
from src.geometry.pattern_unit import Shape, PatternUnit
from src.ga.gene.gene import Gene
from src.storage.stochastic_storage import StochasticStorage


@dataclass
class ShapeGeneParameter:
    """
    ShapeGeneParameter class
    """

    label: str
    bbox: tuple[float, float, float]
    """
    The bounding box of the shape

    Example:
    ```python
    bbox = (10, 10, 0.25)
    # width = 10, height = 10, resolution = 0.25
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
    parameter_id_list = ["r_inner", "r_outer"]
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
        # Validate bbox
        if len(self.bbox) != 3:
            raise ValueError("bbox must have 3 elements.")
        for val in self.bbox:
            if val <= 0:
                raise ValueError("bbox values must be greater than 0.")

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

        for i, (lower, upper) in enumerate(self.parameter_boundary_list):
            if lower >= upper:
                raise ValueError(
                    f"parameter_boundary_list[{i}] must have lower < upper."
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
            f: Callable[[tuple[float, float], V2_group], bool]
        ) -> Callable[[tuple[float, float]], bool]:
            return lambda x, y: f((x, y), parameter_list)

        # Adapt the area parameter functions to the shape
        return [to_af_with_params(f) for f in self.a_f]

    def __repr__(self) -> str:
        return f"ShapeParameter({self.label})"


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
    """

    pdf_storage: StochasticStorage = StochasticStorage("__shape_gene_pdf")
    """
    The storage for the gene parameters pdf

    Example:
    ```json
    # __shape_gene_pdf.json
    {
        "r_inner": [1.1, 0.2, 1.4, 12, 5.5, 6.6, ...],
        "r_outer": [2.2, 0.3, 2.5, 13, 6.5, 7.6, ...]
    }
    """

    def __init__(self, shape_parameter: ShapeGeneParameter, gene_id: str) -> None:
        super().__init__(
            shape_parameter.label,
            gene_id,
            shape_parameter.parameter_count,
            shape_parameter.parameter_boundary_list,
        )

        self.param = shape_parameter
        # Adapt the area parameter functions to the shape
        self.pattern_unit = PatternUnit(
            Shape(
                shape_parameter.w,
                shape_parameter.h,
                shape_parameter.get_a_f_with_params(self.parameter_list),
            ),
            shape_parameter.k,
        )

        self.parameter_list = np.zeros(
            shape_parameter.parameter_count, dtype=np.float64
        )

        self.load_gene()

    def print_parameter_info(self) -> None:
        parameter_info_array = "[ "
        for i, parameter in enumerate(self.parameter_list):
            parameter_info_array += f"{self.param.parameter_id_list[i]}: {parameter}, "
        parameter_info_array += "]"

        print(f"parameter_list: {parameter_info_array}")

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

    def _update_pattern_unit(self) -> None:
        self.pattern_unit.update_area_functions(
            area_functions=self.param.get_a_f_with_params(self.parameter_list)
        )

    def update_gene(self, new_parameter_list: np.ndarray[np.float64]) -> None:
        # Assume that parameter_list is already updated by `mutate` or `mutate_at`
        prev_parameter_list: list[float] = self.parameter_list.tolist()
        curr_parameter_list: list[float] = new_parameter_list.tolist()
        # 1. Replace parameter list at the parameter storage
        self.parameter_storage.insert_field(self.label, curr_parameter_list)
        # 2. Remove prev and add new parameters at the pdf
        for i, param_id in enumerate(self.param.parameter_id_list):
            # 2.1 Delete the old parameter list
            self.pdf_storage.delete_single(param_id, prev_parameter_list[i])
            # 2.2 Insert the new parameter list
            self.pdf_storage.insert_single(param_id, curr_parameter_list[i])

        # 3. Update the parameter list
        self.parameter_list = new_parameter_list

        # 4. Update the pattern unit
        self._update_pattern_unit()

    def load_gene(self) -> None:
        # 1. Load parameters from the parameter storage
        inquired = self.parameter_storage.inquire(self.label)
        self.parameter_list = (
            np.array(inquired, dtype=np.float64)
            if inquired
            else np.zeros(self.param.parameter_count, dtype=np.float64)
        )

        # 2. Insert the gene if it does not exist
        if inquired is None:
            # 2.1. Insert the gene to the parameter storage
            self.parameter_storage.insert_field(
                self.label, self.parameter_list.tolist()
            )
            # 2.2. Insert the gene to the pdf storage
            for i, param_id in enumerate(self.param.parameter_id_list):
                self.pdf_storage.insert_single(param_id, self.parameter_list[i])

        # 3. Update the pattern unit
        self._update_pattern_unit()

    def remove_gene(self) -> None:
        # 1. Remove the gene information
        self.parameter_storage.delete_field(self.label)
        # 2. Remove parameters at the pdf
        for i, param_id in enumerate(self.param.parameter_id_list):
            self.pdf_storage.delete_single(param_id, self.parameter_list[i])

    def get_mutate_methods(
        self,
    ) -> list[Literal["rand", "rand_gaussian", "avg", "top5", "bottom5", "preserve"]]:
        return ["rand", "rand_gaussian", "avg", "top5", "bottom5", "preserve"]

    def mutate(
        self,
        method: Union[
            Literal["rand", "rand_gaussian", "avg", "top5", "bottom5", "preserve"]
        ],
    ) -> None:

        if method == "rand":
            self.update_gene(self._mutate_random())
        elif method == "rand_gaussian":
            self.update_gene(self._mutate_gaussian_random())
        elif method == "avg":
            self.update_gene(self._mutate_avg())
        elif method == "top5":
            self.update_gene(self._mutate_top5())
        elif method == "bottom5":
            self.update_gene(self._mutate_bottom5())
        elif method == "preserve":
            pass

    def _mutate_random(self) -> np.ndarray[np.float64]:
        return self.get_rand_parameter_list()

    def _mutate_gaussian_random(self) -> np.ndarray[np.float64]:
        gaussian_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.param.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            gaussian_parameter_list[i] = self.pdf_storage.pick_random(param_id)

        return gaussian_parameter_list

    def _mutate_top5(self) -> np.ndarray[np.float64]:
        top5_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.param.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            top5_parameter_list[i] = self.pdf_storage.pick_top5(param_id)

        return top5_parameter_list

    def _mutate_bottom5(self) -> np.ndarray[np.float64]:
        bottom5_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.param.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            bottom5_parameter_list[i] = self.pdf_storage.pick_bottom5(param_id)

        return bottom5_parameter_list

    def _mutate_avg(self) -> np.ndarray[np.float64]:
        avg_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.param.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            avg_parameter_list[i] = self.pdf_storage.pick_avg(param_id)

        self.parameter_list = avg_parameter_list

    def mutate_at(
        self, method: Union[Literal["rand", "rand_gaussian", "avg", "top5"]], at: int
    ) -> None:
        if at < 0 or at >= self.param.parameter_count:
            raise ValueError(
                "Mutation at value must be in the range of the parameter count"
            )

        if method == "rand":
            new_parameter_list = self._mutate_random_at(at)
        elif method == "rand_gaussian":
            new_parameter_list = self._mutate_gaussian_random_at(at)
        elif method == "avg":
            new_parameter_list = self._mutate_avg_at(at)
        elif method == "top5":
            new_parameter_list = self._mutate_top5_at(at)

        self.update_gene(new_parameter_list)

        # Update the pattern unit
        self._update_pattern_unit()

    def _mutate_random_at(self, at: int) -> np.ndarray[np.float64]:
        parameter_list = self.parameter_list.copy()
        parameter_list[at] = self.get_rand_parameter_at(at)
        return parameter_list

    def _mutate_gaussian_random_at(self, at: int) -> np.ndarray[np.float64]:
        parameter_list = self.parameter_list.copy()
        parameter_list[at] = self.pdf_storage.pick_random(
            self.param.parameter_id_list[at]
        )
        return parameter_list

    def _mutate_top5_at(self, at: int) -> np.ndarray[np.float64]:
        parameter_list = self.parameter_list.copy()
        parameter_list[at] = self.pdf_storage.pick_top5(
            self.param.parameter_id_list[at]
        )
        return parameter_list

    def _mutate_avg_at(self, at: int) -> np.ndarray[np.float64]:
        parameter_list = self.parameter_list.copy()
        parameter_list[at] = self.pdf_storage.pick_avg(self.param.parameter_id_list[at])
        return parameter_list

    def __repr__(self) -> str:
        return f"ShapeGene({self.label}, parameter_table={self.parameter_table})"

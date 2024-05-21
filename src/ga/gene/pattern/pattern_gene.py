from dataclasses import dataclass, field
from math import floor, pi
from typing import List, Literal, Union, Tuple

import numpy as np

from src.ga.gene.gene import Gene
from src.geometry.pattern_unit import (
    PatternTransformation,
)
from src.storage.stochastic_storage import (
    StochasticStorage,
)


@dataclass
class PatternGeneParameter:
    """
    PatternGeneParameter class
    """

    label: str
    parameter_boundary_list: List[Tuple[float, float]]
    """
    The boundary of the parameters

    Example:
    ```python
    grid_boundary = [
        (1.0, 5.0),     # dx
        (2.0, 5.0),     # dy
    ]
    circular_boundary = [
        (2.0, 5.0)      # di
        (1.0, 5.0),     # dx
        (0.0, 360.0),   # phi
    ]
    corn_boundary = [
        (2.0, 5.0)      # di
        (1.0, 5.0),     # dx
        (0.0, 360.0),   # phi
        (2, 10),        # rot_count
    ]
    ```
    """

    parameter_id_list: List[Literal["di", "dx", "dy", "phi", "rot_count"]]
    """
    The list of parameter ids
    """

    pattern_type: Union[
        Literal["grid", "grid_strict", "circular", "circular_strict", "corn"]
    ]

    transformation: PatternTransformation = field(init=False)

    def __post_init__(self):
        extracted_pattern_type = self.pattern_type.split("_")[0]
        self.transformation = PatternTransformation(pattern_type=extracted_pattern_type)

        for val in self.parameter_boundary_list:
            for boundary in val:
                if boundary <= 0:
                    raise ValueError("The boundary value must be greater than 0")
            if val[0] >= val[1]:
                raise ValueError(
                    "The lower boundary must be less than the upper boundary"
                )

    def update_transformation_params(
        self,
        transformation: Tuple[float, float, float, float, int],
    ) -> dict[str, float]:
        """
        Update the transformation parameters

        Args:
            transformation: tuple[float, float, float, float, int]
                The transformation parameters
                `(di, dx, dy, phi, rot_count)`
        """
        di, dx, dy, phi, rot_count = transformation

        # Ensure that the rot_count is an integer
        rot_count = int(rot_count)

        if self.pattern_type == "grid_strict":
            # Fix to half of the dx for fitting perfectly
            di = dx / 2

        elif self.pattern_type == "circular_strict":
            if phi == 0:
                phi = 0

            rotation_count = floor(2 * pi / phi)

            modified_phi: float = 2 * pi / rotation_count
            phi = modified_phi

        self.transformation.di = di
        self.transformation.dx = dx
        self.transformation.dy = dy
        self.transformation.phi = phi
        self.transformation.rot_count = rot_count

        return {
            "di": self.transformation.di,
            "dx": self.transformation.dx,
            "dy": self.transformation.dy,
            "phi": self.transformation.phi,
            "rot_count": self.transformation.rot_count,
        }

    @property
    def parameter_count(self) -> int:
        """
        Returns the number of parameters
        """
        return len(self.parameter_id_list)


class PatternGene(Gene):
    """
    PatternGene class
    """

    pdf_storage: StochasticStorage = StochasticStorage("__pattern_gene_pdf")
    """
    The storage for the gene parameters pdf

    Example:
    ```json
    # __pattern_gene_pdf.json
    {
        "phi": [1.1, 0.2, 1.4, 12, 5.5, 6.6, ...],
        "dx": [2.2, 0.3, 2.5, 13, 6.5, 7.6, ...],
    }
    """

    def __init__(
        self,
        gene_parameter: PatternGeneParameter,
        gene_id: str,
    ):
        super().__init__(
            label=gene_parameter.label,
            gene_id=gene_id,
            parameter_count=gene_parameter.parameter_count,
            parameter_boundary_list=gene_parameter.parameter_boundary_list,
        )

        self.param = gene_parameter
        self.parameter_list = np.zeros(self.parameter_count, dtype=np.float64)

        self.load_gene()

    @property
    def pattern_transformation(self) -> PatternTransformation:
        """
        Returns the pattern transformation
        """
        return self.param.transformation

    def _get_fixed_transformation(
        self, new_parameter_list: np.ndarray[np.float64]
    ) -> np.ndarray[np.float64]:
        """
        Fix the transformation parameters with pattern type
        1. modify the transformation object permanently
        2. return the fixed transformation parameter list
        """
        # (di, dx, dy, phi, rot_count)
        ordered_params: list[float] = [0] * 5
        for i, param_id in enumerate(self.param.parameter_id_list):
            if param_id == "di":
                ordered_params[0] = new_parameter_list[i]
            elif param_id == "dx":
                ordered_params[1] = new_parameter_list[i]
            elif param_id == "dy":
                ordered_params[2] = new_parameter_list[i]
            elif param_id == "phi":
                ordered_params[3] = new_parameter_list[i]
            elif param_id == "rot_count":
                ordered_params[4] = new_parameter_list[i]

        self.param.update_transformation_params(tuple(ordered_params))

        fixed_parameter_list = np.zeros(self.parameter_count, dtype=np.float64)
        for i, param_id in enumerate(self.param.parameter_id_list):
            if param_id == "di":
                fixed_parameter_list[i] = self.param.transformation.di
            elif param_id == "dx":
                fixed_parameter_list[i] = self.param.transformation.dx
            elif param_id == "dy":
                fixed_parameter_list[i] = self.param.transformation.dy
            elif param_id == "phi":
                fixed_parameter_list[i] = self.param.transformation.phi
            elif param_id == "rot_count":
                fixed_parameter_list[i] = self.param.transformation.rot_count

        return fixed_parameter_list

    def print_parameter_info(self) -> None:
        parameter_info_array = "[ "
        for i in range(self.parameter_count):
            parameter_info_array += (
                f"{self.param.parameter_id_list[i]}: {self.parameter_list[i]}, "
            )
        parameter_info_array += "]"

        print(f"parameter_list: {parameter_info_array}")

    @property
    def parameter_table(self) -> dict[str, float]:
        """
        Returns the parameter table

        Example:
        ```python
        pattern_gene.parameter_table = {
            "di": 1.0,
            "dx": 2.0,
            "phi": 4.0,
            "rot_count": 5.0,
        }
        """
        return {
            self.param.parameter_id_list[i]: self.parameter_list[i]
            for i in range(self.parameter_count)
        }

    def update_gene(self, new_parameter_list: np.ndarray[np.float64]) -> None:
        # 0. Get the fixed transformation
        new_parameter_list = self._get_fixed_transformation(new_parameter_list)

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

            # Check ror_count for integer
            if param_id == "rot_count":
                curr_parameter_list[i] = int(curr_parameter_list[i])
                new_parameter_list[i] = int(new_parameter_list[i])

            self.pdf_storage.insert_single(param_id, curr_parameter_list[i])

        # 3. Update the parameter list
        self.parameter_list = new_parameter_list

    def load_gene(self) -> None:
        # 1. Load the gene from the parameter storage
        # Node that inquired is sorted in the order of the parameter_id_list
        inquired = self.parameter_storage.inquire(self.label)
        self.parameter_list = (
            self._get_fixed_transformation(np.array(inquired, dtype=np.float64))
            if inquired
            else np.zeros(self.parameter_count, dtype=np.float64)
        )

        # 2. Insert the gene if it does not exist
        if inquired is None:
            # 2.1 Insert the gene to the parameter storage
            self.parameter_storage.insert_field(
                self.label, self.parameter_list.tolist()
            )
            # 2.2 Insert the gene to the pdf storage
            for i, param_id in enumerate(self.param.parameter_id_list):
                self.pdf_storage.insert_single(param_id, self.parameter_list[i])

    def remove_gene(self) -> None:
        # 1. Remove the gene from the parameter storage
        self.parameter_storage.delete_field(self.label)
        # 2. Remove the gene from the pdf storage
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
            new_parameter_list = self._mutate_random()
        elif method == "rand_gaussian":
            new_parameter_list = self._mutate_rand_gaussian()
        elif method == "avg":
            new_parameter_list = self._mutate_avg()
        elif method == "top5":
            new_parameter_list = self._mutate_top5()
        elif method == "bottom5":
            new_parameter_list = self._mutate_bottom5()
        elif method == "preserve":
            pass

        self.update_gene(new_parameter_list)

    def _mutate_random(self) -> np.ndarray[np.float64]:
        return self.get_rand_parameter_list()

    def _mutate_rand_gaussian(self) -> np.ndarray[np.float64]:
        gaussian_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            gaussian_parameter_list[i] = self.pdf_storage.pick_avg(param_id)

        return gaussian_parameter_list

    def _mutate_top5(self) -> np.ndarray[np.float64]:
        top5_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            top5_parameter_list[i] = self.pdf_storage.pick_top5(param_id)

        return top5_parameter_list

    def _mutate_bottom5(self) -> np.ndarray[np.float64]:
        bottom5_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            bottom5_parameter_list[i] = self.pdf_storage.pick_bottom5(param_id)

        return bottom5_parameter_list

    def _mutate_avg(self) -> np.ndarray[np.float64]:
        avg_parameter_list: np.ndarray[np.float64] = np.zeros(
            self.parameter_count, dtype=np.float64
        )
        for i, param_id in enumerate(self.param.parameter_id_list):
            avg_parameter_list[i] = self.pdf_storage.pick_avg(param_id)

        return avg_parameter_list

    def mutate_at(self, method: str, at: int) -> None:
        if at < 0 or at >= self.parameter_count:
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
        return (
            f"PatternGene({self.param.label}, parameter_table={self.parameter_table})"
        )

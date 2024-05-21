from abc import abstractmethod
from copy import deepcopy

import numpy as np

from src.storage.storage_unit import Storage


class Gene:
    """
    Gene class

    Attributes:
    label: str
        The label of the gene, unique to the gene
    gene_id: str
        The unique identifier of the gene, unique to the gene should be manually provided
    parameters: np.ndarray[np.float64]
        The parameters of the gene
    parameter_boundary: list[tuple[float, float]]
        The boundary of the parameters
    """

    parameter_storage: Storage[list[float]] = Storage("__gene_parameters")
    """
    The storage for the gene parameters

    Example:
    ```json
    # __gene_parameters.json
    {
        "DonutShape_1": [1.0, 2.0], # parameter list for DonutShape_1
        "DonutShape_2": [4.0, 5.0]  # parameter list for DonutShape_2
    }
    ```
    """

    def __init__(
        self,
        label: str,
        gene_id: str,
        parameter_count: int,
        parameter_boundary_list: list[tuple[float, float]],
    ) -> None:

        self.label = label
        self.gene_id = gene_id
        self.label = f"{label}_{self.gene_id}"  # e.g. SomeShape_2, where 2 means the instance number

        self.parameter_boundary_list = parameter_boundary_list
        self.parameter_count = parameter_count
        self.parameter_list: np.ndarray[np.float64] = np.zeros(
            parameter_count, dtype=np.float64
        )

    def update_gene_id(self, gene_id: str) -> None:
        """
        Updates the gene ID

        Warning:
            - It may cause data loss at store, do not manually change it.
            - It should be modified by the `Chromosome` class
        """
        self.remove_gene()

        self.gene_id = gene_id
        self.label = f"{self.label}_{self.gene_id}"

    @abstractmethod
    def print_parameter_info(self) -> None:
        """
        Returns the string representation of the gene
        Should describe the gene in a human-readable format

        Example:
        parameters: `[r: 0.5, g: 0.3, b: 0.1]`
        """
        raise NotImplementedError

    @abstractmethod
    def update_gene(self, new_parameter_list: np.ndarray[np.float64]) -> None:
        """
        Updates a gene to a store

        Args:
        storage: Storage
            The storage object to save the gene to
        new_parameter_list: np.ndarray[np.float64]
            The new parameter list to replace the gene's current parameter list
        """
        raise NotImplementedError

    @abstractmethod
    def load_gene(self) -> None:
        """
        Loads a gene from a store

        Args:
        storage: Storage
            The storage object to load the gene from
        """
        raise NotImplementedError

    @abstractmethod
    def remove_gene(self) -> None:
        """
        Removes a gene from a store

        Args:
        storage: Storage
            The storage object to remove the gene from
        """
        raise NotImplementedError

    @abstractmethod
    def mutate(self, method: str) -> None:
        """
        Modifies the gene in a user-defined manner

        Args:
        method: str
            The mutation method to use, e.g. `"random", "gaussian", "top5", "avg"`
        """
        raise NotImplementedError

    @abstractmethod
    def get_mutate_methods(self) -> list[str]:
        """
        Returns the mutation methods
        """
        raise NotImplementedError

    @abstractmethod
    def mutate_at(self, method: str, at: int) -> None:
        """
        Modifies the gene at a specific position in a user-defined manner

        Args:
        method: str
            The mutation method to use, e.g. `"random", "gaussian", "top5", "avg"`
        """
        raise NotImplementedError

    def copy_gene(self) -> "Gene":
        """
        Returns a copy of the gene
        """
        return deepcopy(self)

    def mutate_safe(self, method: str) -> None:
        """
        Modifies the gene in a user-defined manner
        """
        copied = self.copy_gene()
        copied.mutate(method)
        return copied

    def mutate_at_safe(self, method: str, at: int) -> None:
        """
        Modifies the gene at a specific position in a user-defined manner
        """
        copied = self.copy_gene()
        copied.mutate_at(method, at)
        return copied

    def get_rand_parameter_at(self, at: int) -> np.float64:
        """
        Returns a random parameter value within the parameter boundary
        """
        if at < 0 or at >= self.parameter_count:
            raise ValueError("Index out of bounds")

        lower_bounds, upper_bounds = self.parameter_boundary_list[at]
        return np.random.uniform(lower_bounds, upper_bounds)

    def get_rand_parameter_list(self) -> np.ndarray[np.float64]:
        """
        Returns a list of random parameter values within the parameter boundary
        """
        randomized_parameters = np.zeros(self.parameter_count, dtype=np.float64)
        for i in range(self.parameter_count):
            lower_bound, upper_bound = self.parameter_boundary_list[i]
            randomized_parameters[i] = np.random.uniform(lower_bound, upper_bound)

        return randomized_parameters

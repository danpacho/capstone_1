import numpy as np

from abc import abstractmethod
from copy import deepcopy


class Gene:
    """
    Gene class

    Attributes:
    label: str
        The label of the gene, unique to the gene
    gene_id: uuid
        The unique identifier of the gene, unique to the gene
    parameters: np.ndarray[np.float64]
        The parameters of the gene
    parameter_boundary: list[tuple[float, float]]
        The boundary of the parameters
    """

    _id_counter = 0

    def __init__(
        self,
        label: str,
        parameter_count: int,
        parameter_boundary_list: list[tuple[float, float]],
    ) -> None:
        self._id_counter += 1

        self.label = label
        self.gene_id = self._id_counter
        self.label = f"{label}_{self.gene_id}"  # e.g. SomeShape_2, where 2 means the instance number

        self.parameter_boundary_list = parameter_boundary_list
        self.parameter_list = np.zeros(parameter_count, dtype=np.float64)

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
    def update_gene(self, new_parameter_list: list[float]) -> None:
        """
        Updates a gene to a store

        Args:
        storage: Storage
            The storage object to save the gene to
        new_parameter_list: list[float]
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
        if at < 0 or at >= len(self.parameter_list):
            raise ValueError("Index out of bounds")

        lower_bounds, upper_bounds = self.parameter_boundary_list[at]
        return np.random.uniform(lower_bounds, upper_bounds)

    def get_rand_parameter_list(self) -> np.ndarray[np.float64]:
        """
        Returns a list of random parameter values within the parameter boundary
        """
        randomized_parameters = np.zeros(len(self.parameter_list), dtype=np.float64)
        for i in range(len(self.parameter_list)):
            lower_bound, upper_bound = self.parameter_boundary_list[i]
            randomized_parameters[i] = np.random.uniform(lower_bound, upper_bound)

        return randomized_parameters

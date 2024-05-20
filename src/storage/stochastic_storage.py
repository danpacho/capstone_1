from functools import wraps
from typing import Callable, Any

import numpy as np

from src.pdf.gaussian import GaussianDistribution
from src.storage.vector_storage import VectorSortedStorage


def update_gaussian(func: Callable) -> Callable:
    """
    Decorator to update the Gaussian distribution of the field before calling the function.
    """

    @wraps(func)
    def wrapper(self, field: str, *args, **kwargs) -> Any:
        # pylint: disable=protected-access
        self._update_gaussian(field)
        return func(self, field, *args, **kwargs)

    return wrapper


class StochasticStorage(VectorSortedStorage):
    """
    Stochastic storage class.
    """

    def __init__(self, root_filename: str):
        super().__init__(root_filename, "StochasticStorage")
        # dont need to load the pdf from the file, as it will be updated on the fly
        self._pdf: dict[str, GaussianDistribution] = {}

    def _update_gaussian(self, field: str) -> None:
        if field in self._db:
            # use the list_inquire, don't have to know the exact order of the data
            samples = np.array(self._db[field].list_inquire)
            if field not in self._pdf:
                self._pdf[field] = GaussianDistribution(samples)
            else:
                self._pdf[field].replace_sample(samples)
        else:
            raise KeyError(f"Field '{field}' does not exist in the storage.")

    @update_gaussian
    def pick_avg(self, field: str) -> float:
        """
        Pick the average value of the field.
        """
        return self._pdf[field].pick_avg()

    @update_gaussian
    def pick_random(self, field: str) -> float:
        """
        Pick a random value from the field.
        """
        return self._pdf[field].pick_rand()

    @update_gaussian
    def pick_top5(self, field: str) -> float:
        """
        Pick the top 5% value of the field.
        """
        return self._pdf[field].pick_top5()

    @update_gaussian
    def pick_bottom5(self, field: str) -> float:
        """
        Pick the bottom 5% value of the field.
        """
        return self._pdf[field].pick_bottom5()

    @update_gaussian
    def plot_distribution(
        self,
        field: str,
    ) -> None:
        """
        Draw the distribution(Histogram) of the field.
        """
        self._pdf[field].plot_distribution(
            xlabel=field, ylabel="Density", title=f"pdf({field})"
        )

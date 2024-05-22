from abc import abstractmethod


class PopularizationInitializer:
    """
    PopularizationInitializer class
    """

    def __init__(self, population_size: int):
        self.population_size = population_size

    @abstractmethod
    def initialize(self):
        """
        Initialize the population
        """
        raise NotImplementedError

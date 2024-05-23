from typing import Literal, Union
import numpy as np
import matplotlib.pyplot as plt

from src.storage.stochastic_storage import StochasticStorage


class EvolutionStorage:
    """
    EvolutionStorage class

    fitness and biased fitness storage:
    ```json
    # __evolution.json
    {
        "1": [
            1.01, 1.02, 1.03, 1.04, ...
        ],
        "2": [
            1.07, 1.08, 1.09, 1.10, ...
        ],
        "3": [
            1.11, 1.12, 1.13, 1.14, ...
        ]
    }
    ```
    """

    def __init__(self, root_filenames: tuple[str, str]):
        self.fitness_storage = StochasticStorage(root_filenames[0])
        self.biased_fitness_storage = StochasticStorage(root_filenames[1])

    def save(self) -> None:
        """
        Save the storage
        """
        self.fitness_storage.save()
        self.biased_fitness_storage.save()

    def reset(self) -> None:
        """
        Reset the storage
        """
        self.fitness_storage.reset()
        self.biased_fitness_storage.reset()

    def insert_fitness_result(
        self, generation: int, fitness_result: tuple[float, float]
    ) -> None:
        """
        Insert the fitness result
        """
        self.fitness_storage.insert_single(generation, fitness_result[0])
        self.biased_fitness_storage.insert_single(generation, fitness_result[1])

    def plot_fitness(
        self,
        xlabel: str = "Generation",
        ylabel: str = "Fitness Value",
        title: str = "Evolution of Fitness over Generations",
        storage: Union[Literal["fitness", "biased_fitness"]] = "fitness",
    ) -> None:
        """
        Plot the fitness

        Args:
            xlabel (`str`): The label of the x-axis
            ylabel (`str`): The label of the y-axis
            title (`str`): The title of the plot
            storage (`Union[Literal["fitness", "biased_fitness"]]`): The storage to plot
        """
        generations = (
            list(map(int, self.fitness_storage.keys))
            if storage == "fitness"
            else list(map(int, self.biased_fitness_storage.keys))
        )
        generations.sort()
        fitness_values = [
            (
                self.fitness_storage.inquire((generation))
                if storage == "fitness"
                else self.biased_fitness_storage.inquire((generation))
            )
            for generation in generations
        ]

        avg_fitness = []
        std_fitness = []
        median_fitness = []
        q1_fitness = []
        q3_fitness = []
        min_fitness = []
        max_fitness = []

        for values in fitness_values:
            avg_fitness.append(np.mean(values))
            std_fitness.append(np.std(values))
            median_fitness.append(np.median(values))
            q1_fitness.append(np.percentile(values, 25))
            q3_fitness.append(np.percentile(values, 75))
            min_fitness.append(np.min(values))
            max_fitness.append(np.max(values))

        avg_fitness = np.array(avg_fitness)
        std_fitness = np.array(std_fitness)
        median_fitness = np.array(median_fitness)
        q1_fitness = np.array(q1_fitness)
        q3_fitness = np.array(q3_fitness)
        min_fitness = np.array(min_fitness)
        max_fitness = np.array(max_fitness)

        plt.figure(figsize=(10, 6))
        plt.plot(
            generations, avg_fitness, label="Average Fitness", color="blue", marker="o"
        )

        # Plotting standard deviation as a shaded area
        plt.fill_between(
            generations,
            avg_fitness - std_fitness,
            avg_fitness + std_fitness,
            color="blue",
            alpha=0.2,
            label="Standard Deviation",
        )

        # Plotting median fitness
        plt.plot(
            generations,
            median_fitness,
            label="Median Fitness",
            color="green",
            marker="o",
        )

        # Plotting quartiles as a shaded area
        plt.fill_between(
            generations,
            q1_fitness,
            q3_fitness,
            color="green",
            alpha=0.2,
            label="Quartiles (Q1 to Q3)",
        )

        # Plotting min and max fitness
        plt.plot(
            generations,
            min_fitness,
            label="Min Fitness",
            color="red",
            linestyle="--",
            marker="x",
        )
        plt.plot(
            generations,
            max_fitness,
            label="Max Fitness",
            color="purple",
            linestyle="--",
            marker="x",
        )

        # Adding labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        # Adding grid for better readability
        plt.grid(True)

        # Display the plot
        plt.show()

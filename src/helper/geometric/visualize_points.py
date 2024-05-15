import matplotlib.pyplot as plt
from grid import GridCell


def visualize_points(cell_list: list[GridCell]) -> None:

    for cell in cell_list:
        x_lower, y_lower = cell.coord
        x_upper = x_lower + cell.k
        y_upper = y_lower + cell.k
        plt.fill_between(
            [
                y_lower,
                y_upper,
            ],
            x_lower,
            x_upper,
            color="gray",
            alpha=0.5,
        )

    plt.show()
    return

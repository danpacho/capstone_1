import matplotlib.pyplot as plt

from src.grid.grid import GridCell


def visualize_points(cell_list: list[GridCell]) -> None:
    """
    visualize points
    args:
        cell_list: list[GridCell] - grid cell list
    """
    plt.figure(figsize=(10, 10))
    plt.axis("equal")

    cell_table: dict[str, int] = {}

    for cell in cell_list:
        x_lower = cell.x
        y_lower = cell.y
        x_upper = x_lower + cell.k
        y_upper = y_lower + cell.k

        if cell.id not in cell_table:
            cell_table[cell.id] = 1
        else:
            cell_table[cell.id] += 1

        plt.fill_between(
            [
                x_lower,
                x_upper,
            ],
            y_lower,
            y_upper,
            color="gray" if cell_table[cell.id] == 1 else "red",
            alpha=0.5 if cell_table[cell.id] == 1 else 1,
        )

    plt.show()

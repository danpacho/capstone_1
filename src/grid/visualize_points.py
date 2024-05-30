import matplotlib.pyplot as plt

from src.geometry.vector import V2_group


def visualize_points(
    grid_cell_v: V2_group, k: float, fig_size: tuple[int, int] = (10, 10)
) -> None:
    """
    visualize points
    args:
        grid_cell_v: Gv - grid cell list
    """
    plt.figure(figsize=fig_size)
    plt.axis("equal")

    cell_table: dict[str, int] = {}

    for cell in grid_cell_v:
        x_lower = cell[0]
        y_lower = cell[1]
        x_upper = x_lower + k
        y_upper = y_lower + k
        grid_id = f"{x_lower}_{y_lower}"

        if grid_id not in cell_table:
            cell_table[grid_id] = 1
        else:
            cell_table[grid_id] += 1

        plt.fill_between(
            [
                x_lower,
                x_upper,
            ],
            y_lower,
            y_upper,
            color="gray" if cell_table[grid_id] == 1 else "red",
            alpha=0.5 if cell_table[grid_id] == 1 else 1,
        )

    plt.show()

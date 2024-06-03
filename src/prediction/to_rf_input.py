# pylint: disable=invalid-name

import numpy as np
from sklearn.decomposition import PCA

from src.grid.grid import Grid
from src.geometry.vector import V2_group, V3_group, V


def to_rf_input(
    pca: PCA,
    pattern_matrix: V2_group,
    bound: tuple[tuple[float, float], tuple[float, float]],
    resolution: float,
    flat: bool = False,
) -> V3_group:
    """
    Convert the `pattern matrix` into the `rf input`
    by adding the `-1` or `1` value to the `pattern matrix`.

    (`[x, y, fill=1|empty=-1]`) - V3_group

    Args:
        pca (`PCA`): PCA object to transform the input
        pattern_matrix (`np.array`): Pattern matrix to convert
        bound (`tuple[tuple[float, float], tuple[float, float]]`): Bound of the pattern matrix
        resolution (`float`): Resolution of the pattern matrix
        flat (`bool`): Flatten the input

    Returns:
        ```md
        pattern_matrix: np.array([1,1], [1,2], ...)
            ▼
        pattern_input
            ▼
        rf_input: np.array([0,0,-1], [0,1,-1], ... ,[1,1,1], [1,2,1], ...)
        ```
    """
    full_coord = Grid(bound=bound, k=resolution).generate_grid()

    pattern_coord_key: set[str] = set()
    for pattern in pattern_matrix:
        pattern_id = f"{pattern[0]}_{pattern[1]}"
        pattern_coord_key.add(pattern_id)

    rf_input: V3_group = V.initialize_matrix_3d()
    rf_coord_key: set[str] = set()

    FILLED = 1
    EMPTY = -1

    for coord in full_coord:
        coord_id = f"{coord[0]}_{coord[1]}"
        if coord_id in rf_coord_key:
            continue

        rf_coord_key.add(coord_id)

        if coord_id in pattern_coord_key:
            rf_input = V.append_v3(rf_input, np.array([coord[0], coord[1], FILLED]))
        else:
            rf_input = V.append_v3(rf_input, np.array([coord[0], coord[1], EMPTY]))

    if full_coord.shape[0] != rf_input.shape[0]:
        raise ValueError(
            "The shape of the `rf_input` is not equal to the `full_coord`."
        )

    if rf_input.shape[1] != 3:
        raise ValueError("The shape of the `rf_input` is not equal to 3.")

    final_input = rf_input.reshape(1, -1)[0] if flat else rf_input.reshape(1, -1)
    final_input = pca.transform(final_input)

    return final_input

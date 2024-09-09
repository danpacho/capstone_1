import numpy as np
from src.geometry.vector import V3_group


def to_original_input(model_input: V3_group) -> np.ndarray[np.float64]:
    """
    Convert the `model input` into the `original input`
    by removing the `-1` or `1` value from the `model input`.

    (`[x, y, fill=1|empty=-1]`) - V3_group

    Args:
        model_input (`np.array`): Model input to convert

    Returns:
        ```md
        model_input: np.array([0,0,-1, 0,1,-1, ... , 1,1,1, 1,2,1])
            ▼
        pattern_input
            ▼
        pattern_matrix: np.array([1,1], [1,2], ...)
        ```
    """
    original_input: np.ndarray[np.float64] = np.array([])

    for i in range(0, model_input.shape[0], 3):
        x = model_input[i]
        y = model_input[i + 1]
        fill = model_input[i + 2]

        if fill == -1:
            continue
        original_input = np.append(original_input, [x, y])

    return original_input.reshape(-1, 2)

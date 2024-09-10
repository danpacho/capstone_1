import numpy as np


def to_image_matrix(
    grid_width: int,
    grid_resolution: int,
    full_grid_matrix: np.ndarray[np.float64],
    pattern_matrix: np.ndarray[np.float64],
) -> np.ndarray[np.int8]:
    """
    Transforms the pattern matrix into a binary image matrix.

    Args:
        full_grid_matrix (np.ndarray[np.float64]): Full grid matrix.
        pattern_matrix (np.ndarray[np.float64]): Pattern matrix.

    Returns:
        np.ndarray[np.float64]: Image matrix.
    """
    image_width: np.int8 = int(grid_width * grid_resolution)
    binary_image_matrix: np.ndarray[np.int8] = np.zeros(
        (image_width, image_width), dtype=np.int8
    )

    pattern_coord_key = set(f"{pattern[0]}_{pattern[1]}" for pattern in pattern_matrix)

    # Define binary values
    FILLED: np.int8 = 1
    EMPTY: np.int8 = 0

    row_count = 0
    col_count = 0

    for index, coord in enumerate(full_grid_matrix):
        if row_count >= image_width or col_count >= image_width:
            break

        coord_id: str = f"{coord[0]}_{coord[1]}"
        if coord_id in pattern_coord_key:
            binary_image_matrix[row_count][col_count] = FILLED
        else:
            binary_image_matrix[row_count][col_count] = EMPTY

        col_count += 1

        if index % image_width == 0:
            row_count += 1
            col_count = 0

    return binary_image_matrix

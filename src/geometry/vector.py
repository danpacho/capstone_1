from numpy import append, float64, empty, array
from numpy.typing import NDArray

V2_group = NDArray[float64]
"""
Vector 2D

Example:

    ```python
    coord_2d: V2_group = np.array([1.0, 2.0])
    grid_mat_2d: V2_group = np.array([coord_2d, coord_2d])
    ```
"""

V3_group = NDArray[float64]
"""
Vector 3D

Example:
    
    ```python
    coord_3d: V3_group = np.array([1.0, 2.0, 3.0])
    grid_mat_3d: V3_group = np.array([coord_3d, coord_3d])
    ```
"""


class V:
    """
    Vector group, 2D and 3D
    """

    @staticmethod
    def initialize_matrix_2d() -> V2_group:
        """
        Initialize Nx2 matrix

        Returns:
            matrix

        Example:

            ```python
            v_mat = V2.initialize_v2_matrix()
            ```
        """
        return empty((0, 2), dtype=float64)

    @staticmethod
    def initialize_matrix_3d() -> V3_group:
        """
        Initialize Nx3 matrix

        Returns:
            matrix

        Example:

            ```python
            v_mat = V2.initialize_v3_matrix()
            ```
        """
        return empty((0, 3), dtype=float64)

    @staticmethod
    def vec2(x: float, y: float) -> V2_group:
        """
        Create a 2-dim vector

        Returns:
            vector 2D
        """
        return array([x, y], dtype=float64)

    @staticmethod
    def vec3(x: float, y: float, z: float) -> V3_group:
        """
        Create a 3-dim vector

        Returns:
            vector 3D
        """
        return array([x, y, z], dtype=float64)

    @staticmethod
    def append_v2(v2_matrix: V2_group, v2_vec: V2_group) -> V2_group:
        """
        Append V2 vector to the V2 matrix

        Returns:
            appended V2 matrix

        Example:

            ```python
            v2 = np.array([1.0, 2.0])
            v2_list = np.array([v2])
            v2_list = V2.append_v2(v2_list, v2)
            ```
        """
        return append(v2_matrix, [v2_vec], axis=0)

    @staticmethod
    def append_v3(v3_matrix: V3_group, v3_vec: V3_group) -> V3_group:
        """
        Append V3 vector to the V3 matrix

        Returns:
            appended V3 matrix

        Example:

            ```python
            v3 = np.array([1.0, 2.0, 3.0])
            v3_list = np.array([v3])
            v3_list = V2.append_v3(v3_list, v3)
            ```
        """
        return append(v3_matrix, [v3_vec], axis=0)

    @staticmethod
    def combine_mat_v2(v2_matrix1: V2_group, v2_matrix2: V2_group) -> V2_group:
        """
        Combine two V2 matrices

        Returns:
            combined V2 matrix

        Example:

            ```python
            v2_list1 = np.array([[1,2], [3,4]])
            v2_list2 = np.array([[5,6], [7,8]])
            v2_list = V2.combine_matrix(v2_list1, v2_list2)
            ```
        """
        return append(v2_matrix1, v2_matrix2, axis=0)

    @staticmethod
    def combine_mat_v3(v3_matrix1: V3_group, v3_matrix2: V3_group) -> V3_group:
        """
        Combine two V3 matrices

        Returns:
            combined V3 matrix

        Example:

            ```python
            v3_list1 = np.array([[1,2,3], [4,5,6]])
            v3_list2 = np.array([[7,8,9], [10,11,12]])
            v3_list = V2.combine_matrix(v3_list1, v3_list2)
            ```
        """
        return append(v3_matrix1, v3_matrix2, axis=0)

import numpy as np
import unittest
from src.geometry.vector import V


class TestV(unittest.TestCase):
    def test_initialize_matrix_2d(self):
        v_mat = V.initialize_matrix_2d()
        self.assertIsInstance(v_mat, np.ndarray)
        self.assertEqual(v_mat.shape, (0, 2))

    def test_initialize_matrix_3d(self):
        v_mat = V.initialize_matrix_3d()
        self.assertIsInstance(v_mat, np.ndarray)
        self.assertEqual(v_mat.shape, (0, 3))

    def test_vec2(self):
        v = V.vec2(1.0, 2.0)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(v.shape, (2,))
        np.testing.assert_array_equal(v, np.array([1.0, 2.0]))

    def test_vec3(self):
        v = V.vec3(1.0, 2.0, 3.0)
        self.assertIsInstance(v, np.ndarray)
        self.assertEqual(v.shape, (3,))
        np.testing.assert_array_equal(v, np.array([1.0, 2.0, 3.0]))

    def test_append_v2(self):
        v2_matrix = np.array([[1.0, 2.0]])
        v2_vec = np.array([3.0, 4.0])
        v2_list = V.append_v2(v2_matrix, v2_vec)
        self.assertIsInstance(v2_list, np.ndarray)
        self.assertEqual(v2_list.shape, (2, 2))
        np.testing.assert_array_equal(v2_list, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_append_v3(self):
        v3_matrix = np.array([[1.0, 2.0, 3.0]])
        v3_vec = np.array([4.0, 5.0, 6.0])
        v3_list = V.append_v3(v3_matrix, v3_vec)
        self.assertIsInstance(v3_list, np.ndarray)
        self.assertEqual(v3_list.shape, (2, 3))
        np.testing.assert_array_equal(
            v3_list, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    def test_combine_mat_v2(self):
        v2_matrix1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        v2_matrix2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        v2_list = V.combine_mat_v2(v2_matrix1, v2_matrix2)
        self.assertIsInstance(v2_list, np.ndarray)
        self.assertEqual(v2_list.shape, (4, 2))
        np.testing.assert_array_equal(
            v2_list, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        )

    def test_combine_mat_v3(self):
        v3_matrix1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v3_matrix2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        v3_list = V.combine_mat_v3(v3_matrix1, v3_matrix2)
        self.assertIsInstance(v3_list, np.ndarray)
        self.assertEqual(v3_list.shape, (4, 3))
        np.testing.assert_array_equal(
            v3_list,
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
            ),
        )


if __name__ == "__main__":
    unittest.main()

"""
GPRModelTrainer class which is responsible for training a Gaussian Process Regressor (GPR) model.
"""

from typing import Tuple, Union
import hashlib
import json
import os
import uuid

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.decomposition import PCA

from src.grid.grid import Grid

# pylint: disable=invalid-name


class GPRModelTrainer:
    """
    Class to handle the training of a Gaussian Process Regressor (GPR) model.

    Attributes:
        vent_scale (float): The scale of the vent.
        vent_resolution (float): The resolution of the vent.
        vent_bound_width (float): The width of the vent bound.
        vent_bound (Union[tuple[tuple[float, float], tuple[float, float]], None]): The vent bound.
        ROOT_PATH (str): The root path of the script.
        TRAIN_PATH (str): The training data path derived from the root path.
    """

    ROOT_PATH = os.getcwd()
    TRAIN_PATH = os.path.join(ROOT_PATH, "train")

    def __init__(
        self,
        vent_scale: float,
        vent_resolution: float,
        vent_bound_width: float,
        vent_bound: Union[tuple[tuple[float, float], tuple[float, float]], None] = None,
        desired_variance: float = 0.95,
    ) -> None:
        self._gpr: tuple[
            GaussianProcessRegressor, GaussianProcessRegressor, GaussianProcessRegressor
        ] = None
        self._pca: PCA = None

        self.vent_scale = vent_scale
        self.vent_resolution = vent_resolution
        self.vent_bound_width = vent_bound_width
        self.vent_bound = vent_bound

        self.desired_variance = desired_variance

    def _box_title(self, title: str) -> None:
        """
        Prints a boxed title for logging purposes.

        Args:
            title (str): The title to be printed.
        """
        log = f"| {title} |"
        log_len = len(log)
        print("-" * log_len)
        print(log)
        print("-" * log_len)

    def _get_dp_set_list(
        self,
    ) -> np.ndarray[np.float64]:
        """
        Retrieves the point data list from a CSV file.

        Returns:
            np.ndarray[np.float64]: The point data list.
        """
        self._box_title("Getting point data list")
        dp_set = np.loadtxt(
            os.path.join(GPRModelTrainer.TRAIN_PATH, "dp/dpset.csv"), delimiter=","
        )
        return dp_set

    def _extract_dot_coordinates(
        self,
        point_data_list: np.ndarray[np.float64],
        grid_w: float,
        scale: float,
    ) -> list[tuple[float, float]]:
        """
        Extracts dot coordinates from the point data list.

        Args:
            point_data_list (np.ndarray[np.float64]): List of point data.
            grid_w (float): The grid width.
            scale (float): The scale factor.

        Returns:
            list[tuple[float, float]]: Transformed dot coordinates.
        """
        dot_coordinates: list[tuple[float, float]] = []
        for i in range(0, 5):
            row_data = point_data_list[19 - i]
            for j in range(3, 17):
                if (row_data - 2 ** (19 - j)) >= 0:
                    row_data = row_data - 2 ** (19 - j)
                else:
                    dot_coordinates.append((20 - i, j + 1))

        for i in range(5, 10):
            row_data = point_data_list[19 - i]
            for j in range(2, 18):
                if (row_data - 2 ** (19 - j)) >= 0:
                    row_data = row_data - 2 ** (19 - j)
                else:
                    dot_coordinates.append((20 - i, j + 1))

        for i in range(10, 15):
            row_data = point_data_list[19 - i]
            for j in range(1, 19):
                if (row_data - 2 ** (19 - j)) >= 0:
                    row_data = row_data - 2 ** (19 - j)
                else:
                    dot_coordinates.append((20 - i, j + 1))

        for i in range(15, 20):
            row_data = point_data_list[19 - i]
            for j in range(0, 20):
                if (row_data - 2 ** (19 - j)) >= 0:
                    row_data = row_data - 2 ** (19 - j)
                else:
                    dot_coordinates.append((20 - i, j + 1))

        transformed_dot_coordinates: list[tuple[float, float]] = []
        grid_scale = (grid_w / 2) / 10
        for dot in dot_coordinates:
            transformed_x = (-dot[0] + 10) * scale * grid_scale
            transformed_y = (dot[1] - 10) * scale * grid_scale
            transformed_dot_coordinates.append((transformed_x, transformed_y))

        return transformed_dot_coordinates

    def _generate_uuid_from_seed(self, seed_string: str):
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return uuid.UUID(bytes=hash_bytes[:16])

    def _extract_hole_BL_origin_coordinates(
        self,
        grid_w: float,
        scale: float,
        result_entry: set[int],
    ) -> dict[int, list[tuple[float, float]]]:
        """
        Extracts hole BL origin coordinates based on the grid width and scale.

        Args:
            grid_w (float): The grid width.
            scale (float): The scale factor.
            result_entry (set[int]): Set of result entries.

        Returns:
            dict[int, list[tuple[float, float]]]: Hole BL origin coordinates.
        """
        point_data = self._get_dp_set_list()
        result_table = {}
        for i, point in enumerate(point_data):
            if i not in result_entry:
                continue
            dot_coordinates = self._extract_dot_coordinates(
                point, grid_w=grid_w, scale=scale
            )
            result_table[i] = dot_coordinates
        return result_table

    def _parse_result(
        self,
    ) -> dict[int, Tuple[float, float, float]]:
        """
        Parses the result file to extract drag, average temperature, and maximum temperature.

        Returns:
            dict[int, Tuple[float, float, float]]: Parsed result data.
        """
        file_path = os.path.join(GPRModelTrainer.TRAIN_PATH, "dp/result.csv")
        self._box_title(f"Parsing result from {file_path}")
        parsed_result = {}
        data = np.loadtxt(file_path, delimiter=",", dtype=str)

        idx_ptr = 0
        for _, row in enumerate(data, start=0):
            if len(row) >= 3:
                last_three = row[-3:]
                if all(field.strip() for field in last_three):
                    try:
                        parsed_result[idx_ptr] = tuple(map(float, last_three))
                        idx_ptr += 1
                    except ValueError:
                        pass
        return parsed_result

    def _create_vent_hole_matrix(
        self,
        hole_origin_coords: list[tuple[float, float]],
        resolution: float,
        scale: float,
        grid_w: float,
    ) -> np.ndarray[np.float64]:
        """
        Creates a vent hole matrix based on hole origin coordinates.

        Args:
            hole_origin_coords (list[tuple[float, float]]): List of hole origin coordinates.
            resolution (float): Resolution of the grid.
            scale (float): Scale factor.
            grid_w (float): Grid width.

        Returns:
            np.ndarray[np.float64]: Vent hole matrix.
        """
        vent_hole_result_matrix = np.zeros((0, 2), dtype=np.float64)
        for hole_coord in hole_origin_coords:
            x, y = hole_coord
            x_u, y_u = x / scale, y / scale
            k = 1 / resolution
            vent_hole_matrix = Grid(
                k=k,
                bound=((x_u, x_u + k * resolution), (y_u, y_u + k * resolution)),
            ).generate_grid(scale)

            for point in vent_hole_matrix:
                p_x, p_y = point
                p_x_u, p_y_u = p_x / scale, p_y / scale
                is_inside_of_vent = (
                    p_x_u >= -grid_w / 2
                    and p_x_u <= grid_w / 2
                    and p_y_u >= -grid_w / 2
                    and p_y_u <= grid_w / 2
                    and (
                        p_y_u <= (1 / 5) * p_x_u + 2 / 5 * grid_w
                        and p_y_u >= -(1 / 5) * p_x_u - 2 / 5 * grid_w
                    ),
                )

                if is_inside_of_vent:
                    vent_hole_result_matrix = np.append(
                        vent_hole_result_matrix, [[p_x, p_y]], axis=0
                    )
        return vent_hole_result_matrix

    def _create_whole_vent_hole_matrix(
        self,
        hold_origin_table: dict[int, list[tuple[float, float]]],
        resolution: float,
        scale: float,
        grid_w: float,
    ) -> dict[int, np.ndarray[np.float64]]:
        """
        Creates a whole vent hole matrix for each hole origin coordinate.

        Args:
            hold_origin_table (dict[int, list[tuple[float, float]]]): Table of hole origin coordinates.
            resolution (float): Resolution of the grid.
            scale (float): Scale factor.
            grid_w (float): Grid width.

        Returns:
            dict[int, np.ndarray[np.float64]]: Whole vent hole matrix.
        """
        vent_hole_whole_matrix_table = {}
        for key, value in hold_origin_table.items():
            res = self._create_vent_hole_matrix(
                hole_origin_coords=value,
                resolution=resolution,
                scale=scale,
                grid_w=grid_w,
            )
            vent_hole_whole_matrix_table[key] = res
        return vent_hole_whole_matrix_table

    def _to_gpr_train_set_x(
        self,
        full_grid_matrix: np.ndarray[np.float64],
        pattern_matrix: np.ndarray[np.float64],
        flat: bool = False,
    ) -> np.ndarray[np.float64]:
        """
        Converts the pattern matrix into a GPR input matrix by adding a binary value indicating presence or absence of the pattern.

        Args:
            full_grid_matrix (np.ndarray[np.float64]): Full grid matrix.
            pattern_matrix (np.ndarray[np.float64]): Pattern matrix.
            flat (bool): Whether to flatten the resulting matrix.

        Returns:
            np.ndarray[np.float64]: GPR input matrix.
        """
        pattern_coord_key = set(
            f"{pattern[0]}_{pattern[1]}" for pattern in pattern_matrix
        )
        gpr_input = np.empty((0, 3), dtype=np.float64)
        gpr_coord_key: set[str] = set()

        for coord in full_grid_matrix:
            coord_id: str = f"{coord[0]}_{coord[1]}"
            if coord_id in gpr_coord_key:
                continue
            gpr_coord_key.add(coord_id)

            FILLED = 1
            EMPTY = -1

            gpr_input = (
                np.append(gpr_input, [np.array([coord[0], coord[1], FILLED])], axis=0)
                if coord_id in pattern_coord_key
                else np.append(
                    gpr_input, [np.array([coord[0], coord[1], EMPTY])], axis=0
                )
            )

        if full_grid_matrix.shape[0] != gpr_input.shape[0]:
            raise ValueError(
                "The shape of the `gpr_input` is not equal to the `full_coord`."
            )
        if gpr_input.shape[1] != 3:
            raise ValueError("The shape of the `gpr_input` is not equal to 3.")

        return gpr_input.reshape(1, -1)[0] if flat else gpr_input.reshape(1, -1)

    def _create_vent_hole_gpr_input_matrix(
        self,
        vent_hole_whole_matrix_table: dict[int, np.ndarray[np.float64]],
        resolution: float,
        scale: float,
        bound: tuple[tuple[float, float], tuple[float, float]],
        flat: bool = False,
    ) -> dict[int, np.ndarray[np.float64]]:
        """
        Creates a GPR input matrix for the vent holes.

        Args:
            vent_hole_whole_matrix_table (dict[int, np.ndarray[np.float64]]): Table of vent hole matrices.
            resolution (float): Resolution of the grid.
            scale (float): Scale factor.
            bound (tuple[tuple[float, float], tuple[float, float]]): Boundaries of the grid.
            flat (bool): Whether to flatten the resulting matrix.

        Returns:
            dict[int, np.ndarray[np.float64]]: GPR input matrix table for the vent holes.
        """
        vent_hole_gpr_input_matrix_table = {}
        full_grid_matrix = Grid(bound=bound, k=1 / resolution).generate_grid(scale)

        for key, value in vent_hole_whole_matrix_table.items():
            gpr_input = self._to_gpr_train_set_x(
                full_grid_matrix=full_grid_matrix,
                pattern_matrix=value,
                flat=flat,
            )
            vent_hole_gpr_input_matrix_table[key] = gpr_input

        return vent_hole_gpr_input_matrix_table

    def _create_optimized_pca(
        self,
        gpr_input_matrix: np.ndarray[np.float64],
        gpr_output_matrix: np.ndarray[np.float64],
    ) -> None:
        """
        Creates an optimized PCA model.

        Args:
            gpr_input_matrix (np.ndarray[np.float64]): GPR input matrix.
            gpr_output_matrix (np.ndarray[np.float64]): GPR output matrix.

        Returns:
            PCA: Optimized PCA model.
        """
        feature_count: int = gpr_output_matrix.shape[0]

        pca_optimal_founding = PCA(n_components=feature_count)
        pca_optimal_founding.fit(gpr_input_matrix, gpr_output_matrix)

        self._box_title("PCA optimal founding initialized")
        cumulative_variance = np.cumsum(pca_optimal_founding.explained_variance_ratio_)

        optimal_feature_count: int = (
            np.argmax(cumulative_variance >= self.desired_variance) + 1
        )

        self._box_title(
            f"Optimal feature count founded: {optimal_feature_count} for desired variance: {self.desired_variance}"
        )

        self._pca = PCA(n_components=optimal_feature_count)
        self._pca.fit(gpr_input_matrix, gpr_output_matrix)

    def _create_gpr_training_set(
        self,
        vent_hole_gpr_matrix_table: dict[int, np.ndarray],
        result_table: dict[int, Tuple[float, float, float]],
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """
        Creates a GPR training set.

        Args:
            vent_hole_gpr_matrix_table (dict[int, np.ndarray]): Table of vent hole GPR matrices.
            result_table (dict[int, Tuple[float, float, float]]): Table of results.

        Returns:
            (gp_input_matrix, gpr_dim_reduction_input_matrix, gpr_output_matrix): GPR training set.
        """
        gpr_input_matrix = list(vent_hole_gpr_matrix_table.values())
        gpr_output_matrix = [
            result_table[key] for key in vent_hole_gpr_matrix_table.keys()
        ]

        gpr_input_matrix = np.vstack(gpr_input_matrix)
        gpr_output_matrix = np.vstack(gpr_output_matrix)

        self._create_optimized_pca(gpr_input_matrix, gpr_output_matrix)

        # Apply PCA -> dimension reduction
        gpr_dim_reduction_input_matrix = self._pca.fit_transform(gpr_input_matrix)

        return (
            gpr_input_matrix,
            gpr_dim_reduction_input_matrix,
            gpr_output_matrix,
        )

    def _train_gpr_model(
        self,
        gpr_input_matrix: np.ndarray,
        gpr_output_matrix: np.ndarray,
    ) -> tuple[
        GaussianProcessRegressor, GaussianProcessRegressor, GaussianProcessRegressor
    ]:
        """
        Trains a Gaussian Process Regressor (GPR) model.

        Args:
            gpr_input_matrix (np.ndarray): GPR input matrix.
            gpr_output_matrix (np.ndarray): GPR output matrix.
            length_scale (float): Length scale.
            sigma_f (float): Signal variance.
            sigma_n (float): Noise variance.

        Returns:
            GaussianProcessRegressor: Trained GPR model.
        """
        # Define kernel and GaussianProcessRegressor
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))

        gpr_drag = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-5
        )
        gpr_avg_temp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-3
        )
        gpr_max_temp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-3
        )

        gpr_output_drag = gpr_output_matrix[:, 0]
        gpr_output_avg_temp = gpr_output_matrix[:, 1]
        gpr_output_max_temp = gpr_output_matrix[:, 2]

        gpr_drag.fit(gpr_input_matrix, gpr_output_drag)
        gpr_avg_temp.fit(gpr_input_matrix, gpr_output_avg_temp)
        gpr_max_temp.fit(gpr_input_matrix, gpr_output_max_temp)

        self._gpr = (gpr_drag, gpr_avg_temp, gpr_max_temp)

        return self._gpr

    def get_pca(self) -> PCA:
        """
        Get the PCA model.

        Returns:
            PCA: PCA model.
        """
        if self._pca is None:
            original_gpr_input_matrix, _, gpr_output_matrix = self.get_training_set(
                use_original_input=True
            )
            self._create_optimized_pca(original_gpr_input_matrix, gpr_output_matrix)

        return self._pca

    def get_trained_gpr_models(
        self,
    ) -> tuple[
        GaussianProcessRegressor, GaussianProcessRegressor, GaussianProcessRegressor
    ]:
        """
        Retrieves or trains a GPR models based on the given configuration.

        Returns:
            `(drag_gpr, avg_temp_gpr, max_temp_gpr)`: Trained GPR models.
        """
        self._box_title(
            f"[TRAINER] GPR model for config:\n> vent_scale: {self.vent_scale}\n> vent_resolution: {self.vent_resolution}\n> vent_bound_width: {self.vent_bound_width}\n> vent_bound: {self.vent_bound}"
        )
        config_id = str(
            hash(
                (
                    self.vent_scale,
                    self.vent_resolution,
                    self.vent_bound_width,
                    self.vent_bound,
                )
            )
        )
        config_path = os.path.join(GPRModelTrainer.TRAIN_PATH, config_id)

        if os.path.exists(config_path):
            self._box_title(f"[TRAINER]: trained model found in {config_path}")
            train_x = np.load(os.path.join(config_path, "input.npy"))
            train_y = np.load(os.path.join(config_path, "output.npy"))
            return self._train_gpr_model(train_x, train_y)

        self._box_title("[TRAINER]: trained model not found, training the model")
        self.vent_bound = (
            self.vent_bound
            if self.vent_bound
            else (
                (-self.vent_bound_width / 2, self.vent_bound_width / 2),
                (-self.vent_bound_width / 2, self.vent_bound_width / 2),
            )
        )
        result_table = self._parse_result()
        result_entry = set(result_table.keys())
        hole_origin_table = self._extract_hole_BL_origin_coordinates(
            grid_w=self.vent_bound_width,
            scale=self.vent_scale,
            result_entry=result_entry,
        )
        vent_hole_whole_matrix_table = self._create_whole_vent_hole_matrix(
            hold_origin_table=hole_origin_table,
            scale=self.vent_scale,
            resolution=self.vent_resolution,
            grid_w=self.vent_bound_width,
        )
        vent_hole_gpr_input_matrix_table = self._create_vent_hole_gpr_input_matrix(
            vent_hole_whole_matrix_table=vent_hole_whole_matrix_table,
            resolution=self.vent_resolution,
            scale=self.vent_scale,
            bound=self.vent_bound,
        )
        gpr_input_matrix, gpr_dim_reduction_input_matrix, gpr_output_matrix = (
            self._create_gpr_training_set(
                vent_hole_gpr_matrix_table=vent_hole_gpr_input_matrix_table,
                result_table=result_table,
            )
        )

        gpr = self._train_gpr_model(
            gpr_dim_reduction_input_matrix,
            gpr_output_matrix,
        )

        self._box_title(f"[TRAINER]: saving the training data into {config_path}")
        os.makedirs(config_path)
        label_info: dict[str, str] = {
            "vent_scale": str(self.vent_scale),
            "vent_resolution": str(self.vent_resolution),
            "vent_bound_width": str(self.vent_bound_width),
            "vent_bound": str(self.vent_bound),
        }
        with open(
            os.path.join(config_path, "model.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(label_info, f, ensure_ascii=False, indent=4)

        np.save(os.path.join(config_path, "input_original.npy"), gpr_input_matrix)
        np.save(os.path.join(config_path, "input.npy"), gpr_dim_reduction_input_matrix)
        np.save(os.path.join(config_path, "output.npy"), gpr_output_matrix)

        return gpr

    def get_training_set(
        self,
        use_original_input: bool = False,
    ) -> Union[
        tuple[np.ndarray[np.float64], np.ndarray[np.float64]],
        tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]],
        None,
    ]:
        """
        Get the training set for the given configuration.

        Args:
            use_original_input (bool): Whether to use the original input.

        Returns:
            (train_x, train_y): Training set.
            or
            (origin_x, train_x, train_y): Training set with original input.
            or
            None: None if the training set does not exist.
        """
        config_id = str(
            hash(
                (
                    self.vent_scale,
                    self.vent_resolution,
                    self.vent_bound_width,
                    self.vent_bound,
                )
            )
        )
        config_path = os.path.join(GPRModelTrainer.TRAIN_PATH, config_id)

        if not os.path.exists(config_path):
            return None

        train_x = np.load(os.path.join(config_path, "input.npy"))
        train_y = np.load(os.path.join(config_path, "output.npy"))
        origin_x = np.load(os.path.join(config_path, "input_original.npy"))

        return (
            (train_x, train_y)
            if not use_original_input
            else (origin_x, train_x, train_y)
        )

"""
ModelTrainer, a class that trains a model based on the provided data.
"""

from abc import abstractmethod
from typing import Generic, Literal, Tuple, TypeVar, Union

import hashlib
import json
import uuid
import os

import numpy as np
from sklearn.decomposition import PCA

from src.grid.grid import Grid


ModelType = TypeVar("ModelType")


class ModelTrainer(Generic[ModelType]):
    """
    ModelTrainer, a class that trains a model based on the provided data.

    Attributes:
        model_name (str): Name of the model.
        data_path (str): Path to the data folder.
        train_path (str): Path to the training folder.
        grid_scale (float): Scale of the grid.
        grid_resolution (float): Resolution of the grid.
        grid_bound_width (float): Width of the grid boundary.
        grid_bound (Union[tuple[tuple[float, float], tuple[float, float]], None]): Boundaries of the grid.
        train_config (dict[str, str]): Training configuration.
        train_id (str): Training ID.
        desired_variance (float): Desired variance for PCA dimension reduction level optimization.
    """

    root_path: str = os.path.join(os.getcwd(), "model")
    """
    Root path of the model trainer, `cwd()/model`.
    """

    def __init__(
        self,
        model_name: str,
        grid_scale: float,
        grid_resolution: float,
        grid_bound_width: float,
        grid_bound: Union[tuple[tuple[float, float], tuple[float, float]], None] = None,
        desired_variance: float = 0.95,
        can_calculate_std: bool = False,
    ) -> None:
        """
        Initializes the ModelTrainer.

        Args:
            model_name (`str`): Name of the model.
            grid_scale (`float`): Scale of the grid.
            grid_resolution (`float`): Resolution of the grid.
            grid_bound_width (`float`): Width of the grid boundary.
            grid_bound (`Union[tuple[tuple[float, float], tuple[float, float]], None]`): Boundaries of the grid.
            desired_variance (`float`): Desired variance for `PCA` dimension reduction level optimization.
            can_calculate_std (`bool`): Whether the model can calculate the standard deviation.
        """
        super().__init__(
            model_name=model_name,
            grid_scale=grid_scale,
            grid_resolution=grid_resolution,
            grid_bound_width=grid_bound_width,
            grid_bound=grid_bound,
        )

        self._box_title(f"Model Trainer: {model_name}")

        self.model_name = model_name.lower()

        self.data_path = os.path.join(ModelTrainer.root_path, "data")
        self.train_path = os.path.join(ModelTrainer.root_path, "train", self.model_name)

        self.grid_scale = grid_scale
        self.grid_resolution = grid_resolution
        self.grid_bound_width = grid_bound_width
        self.grid_bound = grid_bound

        self.can_calculate_std = can_calculate_std

        self.train_config: dict[str, str] = {
            "grid_scale": str(self.grid_scale),
            "grid_resolution": str(self.grid_resolution),
            "grid_bound_width": str(self.grid_bound_width),
            "grid_bound": str(self.grid_bound) if self.grid_bound else "None",
            "desired_variance": str(desired_variance),
        }
        seed_str = "".join(self.train_config.values())
        self.train_id: str = self._generate_uuid_from_seed(seed_str)

        self.train_set: Union[None, np.ndarray[np.float64]] = None
        self.test_set: Union[None, np.ndarray[np.float64]] = None

        self._pca: PCA = None
        self.desired_variance = desired_variance

        self._model: ModelType = None

    @abstractmethod
    def train_model(
        self,
        test_boundary: Union[tuple[int, int], None] = None,
    ) -> None:
        """
        Trains the model.
        """
        raise NotImplementedError

    def get_model(self) -> ModelType:
        """
        Retrieves the model.

        Returns:
            ModelType: The model.
        """
        if self._model is None:
            self.train_model()

        return self._model

    def _generate_uuid_from_seed(self, seed_string: str) -> str:
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes[:16]))

    def get_pca(self) -> PCA:
        """
        Retrieves the PCA model.

        Returns:
            PCA: PCA model.
        """
        if self._pca is None:
            train_x_origin, _, train_y = self.get_train_set(use_original_input=True)
            self._create_optimized_pca(train_x_origin, train_y)

        return self._pca

    def _create_optimized_pca(
        self,
        input_matrix: np.ndarray[np.float64],
        output_matrix: np.ndarray[np.float64],
    ) -> None:
        """
        Creates an optimized PCA model.

        Args:
            input_matrix (np.ndarray[np.float64]): GPR input matrix.
            output_matrix (np.ndarray[np.float64]): GPR output matrix.

        Returns:
            PCA: Optimized PCA model.
        """
        feature_count: int = output_matrix.shape[0]

        pca_optimal_founding = PCA(n_components=feature_count)
        pca_optimal_founding.fit(input_matrix, output_matrix)

        self._box_title("PCA optimal founding initialized")
        cumulative_variance = np.cumsum(pca_optimal_founding.explained_variance_ratio_)

        optimal_feature_count: int = (
            np.argmax(cumulative_variance >= self.desired_variance) + 1
        )

        self._box_title(
            f"Optimal feature count founded: {optimal_feature_count} for desired variance: {self.desired_variance}"
        )

        self._pca = PCA(n_components=optimal_feature_count)
        self._pca.fit(input_matrix, output_matrix)

    def _box_title(self, title: str) -> None:
        """
        Prints a boxed title for logging purposes.

        Args:
            title (str): The title to be printed.
        """
        log = f"| [ModelTrainer]: {title} |"
        log_len = len(log)
        print("-" * log_len)
        print(log)
        print("-" * log_len)

    def _get_trainset(
        self,
        test_boundary: Union[tuple[float, float], None] = None,
    ) -> np.ndarray[np.float64]:
        """
        Retrieves the training data set.

        Args:
            test_boundary (Union[tuple[float, float], None]): Boundary of the test set.
            if provided, the test set will be extracted from the training set.

        Returns:
            np.ndarray[np.float64]: The training data set.

        Examples:
            >>> trainer._get_trainset()
            >>> trainer._get_trainset(test_boundary=(0, 10))
        """
        if self.train_set is not None and self.test_set is not None:
            return self.train_set

        self._box_title("Getting point data list")
        self.train_set = np.loadtxt(
            os.path.join(self.data_path, "dpset.csv"), delimiter=","
        )

        if test_boundary is not None:
            test_set = self.train_set[test_boundary[0] : test_boundary[1] + 1]
            self.train_set = np.concatenate(
                (
                    self.train_set[: test_boundary[0]],
                    self.train_set[test_boundary[1] + 1 :],
                )
            )
            self.test_set = test_set

        return self.train_set

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

    def _extract_hole_BL_origin_coordinates(
        self,
        point_data: np.ndarray[np.float64],
        point_range_info: tuple[Literal["test", "train"], Union[tuple[int, int], None]],
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
            test_boundary (Union[tuple[float, float], None]): Boundary of the test set.

        Returns:
            coords (dict[int, list[tuple[float, float]]]): Hole BL origin coordinates
        """
        result_table = {}

        point_type, point_range = point_range_info
        if point_range is None:
            point_range = (0, len(point_data) - 1)

        if point_type == "test":
            for i, point in enumerate(point_data):
                if point_range[0] <= i <= point_range[1]:
                    if i in result_entry:
                        result_table[i] = self._extract_dot_coordinates(
                            point_data[i], grid_w=grid_w, scale=scale
                        )
            return result_table

        for i, point in enumerate(point_data):
            if i > point_range[1] or i < point_range[0]:
                if i in result_entry:
                    result_table[i] = self._extract_dot_coordinates(
                        point, grid_w=grid_w, scale=scale
                    )

        return result_table

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
            mat (np.ndarray[np.float64]): Vent hole matrix.
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

    def _create_total_vent_hole_matrix_from_origin_table(
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
        vent_hole_whole_matrix_table: dict[int, np.ndarray[np.float64]] = {}

        for key, value in hold_origin_table.items():
            res = self._create_vent_hole_matrix(
                hole_origin_coords=value,
                resolution=resolution,
                scale=scale,
                grid_w=grid_w,
            )
            vent_hole_whole_matrix_table[key] = res

        return vent_hole_whole_matrix_table

    def _to_trainable_input_matrix(
        self,
        full_grid_matrix: np.ndarray[np.float64],
        pattern_matrix: np.ndarray[np.float64],
        flat: bool = False,
    ) -> np.ndarray[np.float64]:
        """
        Converts the pattern matrix into a input matrix by adding a binary value indicating presence or absence of the pattern.

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
        origin_input_matrix = np.empty((0, 3), dtype=np.float64)
        origin_input_coord_key: set[str] = set()

        for coord in full_grid_matrix:
            coord_id: str = f"{coord[0]}_{coord[1]}"
            if coord_id in origin_input_coord_key:
                continue
            origin_input_coord_key.add(coord_id)

            FILLED = 1
            EMPTY = -1

            origin_input_matrix = (
                np.append(
                    origin_input_matrix,
                    [np.array([coord[0], coord[1], FILLED])],
                    axis=0,
                )
                if coord_id in pattern_coord_key
                else np.append(
                    origin_input_matrix, [np.array([coord[0], coord[1], EMPTY])], axis=0
                )
            )

        if full_grid_matrix.shape[0] != origin_input_matrix.shape[0]:
            raise ValueError(
                "The shape of the `full_grid_matrix` is not equal to the `origin_input`."
            )
        if origin_input_matrix.shape[1] != 3:
            raise ValueError("The shape of the `origin_input` is not equal to 3.")

        return (
            origin_input_matrix.reshape(1, -1)[0]
            if flat
            else origin_input_matrix.reshape(1, -1)
        )

    def _create_vent_hole_input_matrix(
        self,
        vent_hole_total_matrix_table: dict[int, np.ndarray[np.float64]],
        resolution: float,
        scale: float,
        bound: tuple[tuple[float, float], tuple[float, float]],
        flat: bool = False,
    ) -> dict[int, np.ndarray[np.float64]]:
        """
        Creates a input matrix for the vent holes.

        Args:
            vent_hole_whole_matrix_table (dict[int, np.ndarray[np.float64]]): Table of vent hole matrices.
            resolution (float): Resolution of the grid.
            scale (float): Scale factor.
            bound (tuple[tuple[float, float], tuple[float, float]]): Boundaries of the grid.
            flat (bool): Whether to flatten the resulting matrix.

        Returns:
            dict[int, np.ndarray[np.float64]]: GPR input matrix table for the vent holes.
        """
        vent_hole_input_matrix_table = {}
        full_grid_matrix = Grid(bound=bound, k=1 / resolution).generate_grid(scale)

        for key, value in vent_hole_total_matrix_table.items():
            trainable_input = self._to_trainable_input_matrix(
                full_grid_matrix=full_grid_matrix,
                pattern_matrix=value,
                flat=flat,
            )
            vent_hole_input_matrix_table[key] = trainable_input

        return vent_hole_input_matrix_table

    def _create_optimized_pca(
        self,
        input_matrix: np.ndarray[np.float64],
        output_matrix: np.ndarray[np.float64],
    ) -> None:
        """
        Creates an optimized PCA model.

        Args:
            gpr_input_matrix (np.ndarray[np.float64]): GPR input matrix.
            gpr_output_matrix (np.ndarray[np.float64]): GPR output matrix.

        Returns:
            PCA: Optimized PCA model.
        """
        feature_count: int = output_matrix.shape[0]

        pca_optimal_founding = PCA(n_components=feature_count)
        pca_optimal_founding.fit(input_matrix, output_matrix)

        self._box_title("PCA optimal founding initialized")
        cumulative_variance = np.cumsum(pca_optimal_founding.explained_variance_ratio_)

        optimal_feature_count: int = (
            np.argmax(cumulative_variance >= self.desired_variance) + 1
        )

        self._box_title(
            f"Optimal feature count founded: {optimal_feature_count} for desired variance: {self.desired_variance}"
        )

        self._pca = PCA(n_components=optimal_feature_count)
        self._pca.fit(input_matrix, output_matrix)

    def _create_training_set(
        self,
        vent_hole_matrix_table: dict[int, np.ndarray],
        result_table: dict[int, Tuple[float, float, float]],
        skip_pca_recalculation: bool = False,
        external_pca: Union[PCA, None] = None,
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """
        Creates a training set.

        Args:
            vent_hole_matrix_table (dict[int, np.ndarray]): Table of vent hole GPR matrices.
            result_table (dict[int, Tuple[float, float, float]]): Table of results.

        Returns:
            (gp_input_matrix, gpr_dim_reduction_input_matrix, gpr_output_matrix): GPR training set.
        """
        input_matrix = list(vent_hole_matrix_table.values())
        output_matrix = [result_table[key] for key in vent_hole_matrix_table.keys()]

        input_matrix = np.vstack(input_matrix)
        output_matrix = np.vstack(output_matrix)

        if not skip_pca_recalculation:
            self._create_optimized_pca(input_matrix, output_matrix)

        # Apply PCA -> dimension reduction
        if external_pca is not None:
            dim_reduction = external_pca.transform(input_matrix)
        else:
            dim_reduction = self._pca.fit_transform(input_matrix)

        return (
            input_matrix,
            dim_reduction,
            output_matrix,
        )

    def _parse_output(
        self,
    ) -> dict[int, Tuple[float, float, float]]:
        """
        Parses the result file to extract `(drag, average temperature, and maximum temperature)`.

        Returns:
            dict[int, Tuple[float, float, float]]: Parsed result data.
        """
        file_path = os.path.join(self.data_path, "result.csv")
        self._box_title(f"Parsing result from {file_path}")
        parsed_result: dict[int, Tuple[float, float, float]] = {}
        data = np.loadtxt(file_path, delimiter=",", dtype=str, encoding="utf-8")

        idx_ptr: int = 0
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

    def get_test_set(
        self,
        test_boundary: tuple[int, int],
        origin_pca: PCA,
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        """
        Retrieves the test set.

        Returns:
            (`test_origin, test_input_matrix, test_output_matrix`): Test set.

        Args:
            test_boundary (`Union[tuple[float, float], None]`): Boundary of the test set.
        """

        if self.test_set is None:
            self._get_trainset(test_boundary=test_boundary)

        test_origin, test_input_matrix, test_output_matrix = self._transform_data(
            point_data=self.test_set,
            point_range_info=("test", test_boundary),
            skip_pca_recalculation=True,
            external_pca=origin_pca,
        )
        return test_origin, test_input_matrix, test_output_matrix

    def _transform_data(
        self,
        point_data: np.ndarray[np.float64],
        point_range_info: tuple[Literal["test", "train"], Union[tuple[int, int], None]],
        skip_pca_recalculation: bool = False,
        external_pca: Union[PCA, None] = None,
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
        result_table: dict[int, Tuple[float, float, float]] = self._parse_output()
        result_entry: set[int] = set(result_table.keys())

        hole_origin_table: dict[int, list[tuple[float, float]]] = (
            self._extract_hole_BL_origin_coordinates(
                point_data=point_data,
                point_range_info=point_range_info,
                grid_w=self.grid_bound_width,
                scale=self.grid_scale,
                result_entry=result_entry,
            )
        )
        vent_hole_hole_matrix_table: dict[int, np.ndarray[np.float64]] = (
            self._create_total_vent_hole_matrix_from_origin_table(
                hold_origin_table=hole_origin_table,
                scale=self.grid_scale,
                resolution=self.grid_resolution,
                grid_w=self.grid_bound_width,
            )
        )
        vent_hole_input_matrix_table: dict[int, np.ndarray[np.float64]] = (
            self._create_vent_hole_input_matrix(
                vent_hole_total_matrix_table=vent_hole_hole_matrix_table,
                resolution=self.grid_resolution,
                scale=self.grid_scale,
                bound=self.grid_bound,
            )
        )
        x_original, x_reduction, y = self._create_training_set(
            vent_hole_matrix_table=vent_hole_input_matrix_table,
            result_table=result_table,
            skip_pca_recalculation=skip_pca_recalculation,
            external_pca=external_pca,
        )

        return x_original, x_reduction, y

    def get_train_set(
        self,
        use_original_input: bool = False,
        test_boundary: Union[tuple[int, int], None] = None,
    ) -> Union[
        Tuple[
            np.ndarray[np.float64],
            np.ndarray[np.float64],
            np.ndarray[np.float64],
        ],
        Tuple[
            np.ndarray[np.float64],
            np.ndarray[np.float64],
        ],
    ]:
        """
        Retrieves the training set.

        Args:
            use_original_input (bool): Whether to use the original input.
            test_boundary (Union[tuple[int, int], None]): Boundary of the test set.

        Returns:
        - (`train_x_original, train_x_reduction, train_y`): Training set with original input.
        - (`train_x_reduction, train_y`): Training set.

        Examples:
            >>> trainer.get_train_set()
            >>> trainer.get_train_set(use_original_input=True)
            >>> trainer.get_train_set(test_boundary=(0, 10))
        """
        model_train_path = os.path.join(self.train_path, self.train_id)

        if os.path.exists(model_train_path):
            self._box_title("Training set found, loading")
            train_x_original = np.load(
                os.path.join(model_train_path, "train_input_original.npy")
            )
            train_x_reduction = np.load(
                os.path.join(model_train_path, "train_input.npy")
            )
            train_y = np.load(os.path.join(model_train_path, "train_output.npy"))

            return (
                (
                    train_x_original,
                    train_x_reduction,
                    train_y,
                )
                if use_original_input
                else (
                    train_x_reduction,
                    train_y,
                )
            )

        self._box_title("Training set not found, creating a new one")
        os.makedirs(model_train_path, exist_ok=False)

        train_x_original, train_x_reduction, train_y = self._transform_data(
            point_data=self._get_trainset(test_boundary=test_boundary),
            point_range_info=("train", test_boundary),
        )

        with open(
            os.path.join(model_train_path, "train_config.json"),
            "w",
            encoding="utf-8",
            newline="\r",
        ) as f:
            json.dump(self.train_config, f, ensure_ascii=False, indent=4)

        np.save(
            os.path.join(model_train_path, "train_input_original.npy"),
            train_x_original,
        )
        np.save(
            os.path.join(model_train_path, "train_input.npy"),
            train_x_reduction,
        )
        self._box_title("Reduction input saved to train_input.npy")

        np.save(
            os.path.join(model_train_path, "train_output.npy"),
            train_y,
        )

        return (
            (
                train_x_original,
                train_x_reduction,
                train_y,
            )
            if use_original_input
            else (
                train_x_reduction,
                train_y,
            )
        )

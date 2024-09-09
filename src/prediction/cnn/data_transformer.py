from datetime import datetime
import os
from typing import Literal, Tuple, Union
import numpy as np

from src.grid.grid import Grid


class BinaryImageTransformer:
    """
    Transform raw grid data into `image` format.
    """

    @staticmethod
    def title(text: str) -> str:
        """
        Args:
            text (str): Text to display.

        Returns:
            str: Title text.
        """
        print(f"{'=' * 75}\n{text}\n{'=' * 75}")

    def __init__(
        self,
        data_dir: str,
        grid_width: int = 70,
        grid_resolution: float = 2.0,
    ) -> None:
        """
        Args:
            data_dir (str): Directory of the data.
            grid_size (int): Size of the grid.(default 70)
            grid_resolution (float): Resolution of the grid.(default 2.0)

        Note:
            - The `grid_size` must be an even number.
            - The `grid_resolution` must be a positive number.
            - Result image size will be `(grid_size * grid_resolution)^2`.
        """
        self.data_dir = data_dir
        self.data_info: dict[
            Literal[
                "result",
                "dpset",
            ],
            str,
        ] = {
            "result": "result.csv",
            "dpset": "dpset.csv",
        }
        self.data_result_path = os.path.join(data_dir, self.data_info["result"])
        self.data_dpset_path = os.path.join(data_dir, self.data_info["dpset"])

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"File not found: {data_dir}")
        if not os.path.exists(self.data_dpset_path):
            raise FileNotFoundError(f"Data Point set not found: {self.data_dpset_path}")
        if not os.path.exists(self.data_result_path):
            raise FileNotFoundError(f"Result file not found: {self.data_result_path}")

        self.grid_width = grid_width
        self.grid_resolution = grid_resolution
        self.grid_bound = (
            (-grid_width / 2, grid_width / 2),
            (-grid_width / 2, grid_width / 2),
        )

        if grid_width % 2 != 0:
            raise ValueError("Grid size must be an even number.")

        self.opt_id = f"grid_{grid_width}_res_{grid_resolution}"
        self.save_path = os.path.join(
            os.path.dirname(data_dir),
            "images",
            self.opt_id,
        )
        option_str = [
            f"data_path={data_dir}",
            f"grid_size={grid_width}",
            f"grid_resolution={grid_resolution}",
            f"grid_bound={self.grid_bound}",
            f"save_path={self.save_path}",
        ]
        self.title(
            f"Grid Transformer, option: \n\t{'\n\t'.join(option_str)} initialized."
        )
        # Create directories
        os.makedirs(self.save_path, exist_ok=True)
        # Modify readme file
        with open(f"{self.save_path}/readme.md", "w", encoding="utf-8") as f:
            f.write(
                f"Data transformed from {data_dir} with grid size {grid_width}.\n\nInitialized at {datetime.now()}."
            )
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"Directory not found: {self.save_path}")

    def _extract_dot_coordinates(
        self,
        point_data_list: np.ndarray[np.float64],
    ) -> list[tuple[float, float]]:
        """
        Pipeline1:
        Extracts dot coordinates from the point data list.

        Args:
            point_data_list (np.ndarray[np.float64]): List of point data.

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
        grid_scale = (self.grid_width / 2) / 10
        for dot in dot_coordinates:
            transformed_x = (-dot[0] + 10) * grid_scale
            transformed_y = (dot[1] - 10) * grid_scale
            transformed_dot_coordinates.append((transformed_x, transformed_y))

        return transformed_dot_coordinates

    def _extract_hole_BL_origin_coordinates(
        self,
        point_data: np.ndarray[np.float64],
        result_entry: set[int],
    ) -> dict[int, list[tuple[float, float]]]:
        """
        Extracts hole BL origin coordinates based on the grid width and scale.

        Args:
            point_data (np.ndarray[np.float64]): Point data.
            result_entry (set[int]): Set of result entries.

        Returns:
            coords (dict[int, list[tuple[float, float]]]): Hole BL origin coordinates
        """
        result_table: dict[int, list[tuple[float, float]]] = {}

        for i, point in enumerate(point_data):
            if i in result_entry:
                result = self._extract_dot_coordinates(point)
                result_table[i] = result

        return result_table

    def _create_vent_hole_matrix(
        self,
        hole_origin_coords: list[tuple[float, float]],
    ) -> np.ndarray[np.float64]:
        """
        Creates a vent hole matrix based on hole origin coordinates.

        Args:
            hole_origin_coords (list[tuple[float, float]]): List of hole origin coordinates.

        Returns:
            mat (np.ndarray[np.float64]): Vent hole matrix.
        """
        vent_hole_result_matrix = np.zeros((0, 2), dtype=np.float64)

        TEST_VENT_SIZE: int = self.grid_width / 20

        for hole_coord in hole_origin_coords:
            x, y = hole_coord
            k = 1 / self.grid_resolution
            vent_hole_matrix = Grid(
                k=k,
                bound=(
                    (x, x + TEST_VENT_SIZE),
                    (y, y + TEST_VENT_SIZE),
                ),
            ).generate_grid(scale=1)

            for point in vent_hole_matrix:
                p_x, p_y = point
                # Check if the point is inside the target vent hole design domain
                # 1) y <= x / 5 + 2 * grid_width / 5
                # 2) y >= -x / 5 - 2 * grid_width / 5
                is_inside_of_vent = (
                    p_y <= p_x / 5 + 2 * self.grid_width / 5
                    and p_y >= -1 * (p_x / 5) - 2 * self.grid_width / 5
                    # TODO: What is this problem? the left side is a little bit upward.
                    #   + self.grid_resolution
                )

                if is_inside_of_vent:
                    vent_hole_result_matrix = np.append(
                        vent_hole_result_matrix, [[p_x, p_y]], axis=0
                    )
        return vent_hole_result_matrix

    def _create_total_vent_hole_matrix_from_origin_table(
        self,
        hole_origin_table: dict[int, list[tuple[float, float]]],
    ) -> dict[int, np.ndarray[np.float64]]:
        """
        Creates a whole vent hole matrix for each hole origin coordinate.

        Args:
            hole_origin_table (dict[int, list[tuple[float, float]]]): Table of hole origin coordinates.

        Returns:
            dict[int, np.ndarray[np.float64]]: Whole vent hole matrix.
        """
        vent_hole_whole_matrix_table: dict[int, np.ndarray[np.float64]] = {}

        for key, value in hole_origin_table.items():
            res = self._create_vent_hole_matrix(
                hole_origin_coords=value,
            )
            vent_hole_whole_matrix_table[key] = res

        return vent_hole_whole_matrix_table

    def _to_image_matrix(
        self,
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
        image_width: np.int8 = int(self.grid_width * self.grid_resolution)
        binary_image_matrix: np.ndarray[np.int8] = np.zeros(
            (image_width, image_width), dtype=np.int8
        )

        pattern_coord_key = set(
            f"{pattern[0]}_{pattern[1]}" for pattern in pattern_matrix
        )

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

    def _create_vent_hole_image_matrix(
        self,
        vent_hole_total_matrix_table: dict[int, np.ndarray[np.float64]],
    ) -> np.ndarray[np.int8]:
        """
        Creates a vent hole image matrix for each vent hole matrix.

        Args:
            vent_hole_total_matrix_table (dict[int, np.ndarray[np.float64]]): Table of vent hole matrices.

        Returns:
            np.ndarray[np.int8]: Vent hole image matrix. Stacked into a 3D tensor.
        """
        image_matrices = []

        full_grid_matrix = Grid(
            bound=self.grid_bound, k=1 / self.grid_resolution
        ).generate_grid(scale=1, x_major_iteration=True)

        for pattern_matrix in vent_hole_total_matrix_table.values():
            image_matrix = self._to_image_matrix(
                full_grid_matrix,
                pattern_matrix,
            )
            image_matrices.append(image_matrix)  # Add each image matrix to the list

        # Stack images
        vent_hole_image_matrix = np.stack(image_matrices, axis=0)

        return vent_hole_image_matrix

    def _get_output_data_from_raw_csv(
        self,
    ) -> Tuple[dict[int, Tuple[float, float, float]], np.ndarray[np.float64]]:
        """
        Parses the result file to extract `(drag, average temperature, and maximum temperature)`.

        Returns:
            dict[int, Tuple[float, float, float]]: Parsed result data.
        """
        parsed_result: dict[int, Tuple[float, float, float]] = {}
        data = np.loadtxt(
            self.data_result_path, delimiter=",", dtype=str, encoding="utf-8"
        )

        results = []

        idx_ptr: int = 0
        for _, row in enumerate(data, start=0):
            if len(row) >= 3:
                last_three = row[-3:]
                if all(field.strip() for field in last_three):
                    try:
                        last_three_vec = np.array(last_three, dtype=np.float64)
                        results.append(last_three_vec)
                        parsed_result[idx_ptr] = tuple(map(float, last_three))
                        idx_ptr += 1
                    except ValueError:
                        pass

        results_matrix = np.vstack(results)

        return parsed_result, results_matrix

    def _transform(
        self,
        point_data: np.ndarray[np.float64],
        output_data: dict[int, Tuple[float, float, float]],
    ) -> np.ndarray[np.int8]:
        """
        Transforms the point data into an image matrix.

        Args:
            point_data (np.ndarray[np.float64]): basic point data from self.data_dpset_path.

        Returns:
            np.ndarray[np.int8]: Transformed image matrix.
        """
        result_entry: set[int] = set(output_data.keys())

        self.title("1. Extracting hole BL origin coordinates.")
        hole_origin_table = self._extract_hole_BL_origin_coordinates(
            point_data,
            result_entry,
        )
        self.title("2. Creating vent hole matrix.")
        vent_hole_hole_matrix_table = (
            self._create_total_vent_hole_matrix_from_origin_table(
                hole_origin_table,
            )
        )
        self.title("3. Creating vent hole image matrix.")
        vent_hole_image_matrix = self._create_vent_hole_image_matrix(
            vent_hole_hole_matrix_table,
        )
        self.title("Transformation completed.")
        return vent_hole_image_matrix

    def _get_point_data_from_raw_csv(self) -> np.ndarray[np.float64]:
        """
        Extracts point data from the raw CSV file.

        Returns:
            np.ndarray[np.float64]: Point data.
        """
        self.title(f"Extracting point data from {self.data_dpset_path}")

        raw_data = np.loadtxt(self.data_dpset_path, delimiter=",", encoding="utf-8")
        return raw_data

    @property
    def image_matrix_path(self) -> str:
        """
        Returns the path of the image matrix file.

        Returns:
            str: Image matrix file path.
        """
        return os.path.join(self.save_path, "image_matrix.npy")

    @property
    def output_matrix_path(self) -> str:
        """
        Returns the path of the output matrix file.

        Returns:
            str: Output matrix file path.
        """
        return os.path.join(self.data_dir, "output_matrix.npy")

    def _save_image_matrix(self, image_matrix: np.ndarray[np.int8]) -> None:
        """
        Saves the image matrix into a file.

        Args:
            image_matrix (np.ndarray[np.int8]): Image matrix.
        """
        np.save(
            self.image_matrix_path,
            image_matrix,
        )
        self.title(f"Image matrix saved at {self.image_matrix_path}")

    def _load_image_matrix(self) -> Union[np.ndarray[np.int8], None]:
        """
        Loads the image matrix from the file.

        Returns:
            np.ndarray[np.int8]: Image matrix.
        """
        self.title(f"Loading image matrix from {self.image_matrix_path}")

        if not os.path.exists(self.image_matrix_path):
            self.title(f"Image matrix not found at {self.image_matrix_path}")
            return None

        image_matrix = np.load(self.image_matrix_path)
        self.title(f"Image matrix loaded from {self.image_matrix_path}")
        return image_matrix

    def _save_output_matrix(self, output_matrix: np.ndarray[np.float64]) -> None:
        """
        Saves the output matrix into a file.

        Args:
            output_matrix (np.ndarray[np.float64]): Output matrix.
        """
        np.save(
            self.output_matrix_path,
            output_matrix,
        )
        self.title(f"Output matrix saved at {self.output_matrix_path}")

    def generate_images(self, reset: bool = False) -> np.ndarray[np.int8]:
        """
        Generates images from the raw data.

        - Image size will be `(grid_size * grid_resolution)^2`.
        - Image is binary square matrix(one-hot encoded).
        """
        output_data, output_matrix = self._get_output_data_from_raw_csv()

        self._save_output_matrix(output_matrix)

        loaded = self._load_image_matrix()
        if loaded is not None and not reset:
            return loaded

        point_data = self._get_point_data_from_raw_csv()

        transformed_image_matrix = self._transform(point_data, output_data)

        self._save_image_matrix(transformed_image_matrix)

        return transformed_image_matrix

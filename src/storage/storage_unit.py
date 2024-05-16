import os

from typing import Union
from file_manager import JSONFileManager


class Storage:
    """
    Storage class is a base class for all storage units.

    Attributes:
        root_path: str
            The root path of the storage unit.
    """

    def __init__(self, root_filename: str, label: str = "Storage"):

        root = os.path.join(os.getcwd(), root_filename + ".json")

        self.root_path = root
        self.label = label
        self.io = JSONFileManager(root_path=self.root_path, label=label)
        self.io.load_json()
        self.db: dict = self.io.read()

    def check_field_exist(self, field: str) -> bool:
        """
        Check if the field exists in the storage unit.

        Args:
            field: str
                The field to check.

        Returns:
            bool
                True if the field exists, False otherwise.
        """
        return field in self.keys

    def get_field(self, field: str) -> any:
        """
        Get the field from the storage unit.

        Args:
            field: str
                The field to get.

        Returns:
            any
                The field value.
        """
        return self.db[field]

    def save(self) -> None:
        """
        Save the storage unit to the file.
        """
        self.io.save_json(self.db)

    def insert(self, field: str, value: Union[str, int, float, dict, list]) -> None:
        """
        Insert a field to the storage unit.

        Args:
            field: str
                The field to insert.
            value: Union[str, int, float, dict, list]
                The value to insert.
        """
        if self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} already exists.")
            return

        self.db[field] = value

    def delete(self, field: str) -> None:
        """
        Delete a field from the storage unit.

        Args:
            field: str
                The field to delete.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return

        del self.db[field]

    def reset(self) -> None:
        """
        Reset the storage unit.
        """
        self.db = {}
        self.io.save_json(self.db)

    def update(self, field: str, value: Union[str, int, float, dict, list]) -> None:
        """
        Update a field in the storage unit.

        Args:
            field: str
                The field to update.
            value: Union[str, int, float, dict, list]
                The value to update.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return

        self.db[field] = value

    @property
    def keys(self) -> list[str]:
        """
        Get the keys of the storage unit.

        Returns:
            list[str]
                The list of keys.
        """
        return list(self.db.keys())

    def inquire(self, field: str) -> Union[str, int, float, dict, list, None]:
        """
        Inquire the storage unit.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return None
        return self.db[field]

    def __repr__(self) -> str:
        return f"[Storage Unit: {self.label}] (root_path={self.root_path})"

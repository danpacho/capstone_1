from src.ds.bst import BST
from src.storage.storage_unit import Storage


class VectorStorage(Storage[list[float]]):
    """
    VectorStorage for storing list[float] vector kinda data.
    """

    def __init__(self, root_filename: str, label: str = "VectorStorage"):
        super().__init__(root_filename, label)
        self._db: dict[str, BST] = {}
        loaded: dict[str, list[float]] = self._io.read()
        for key, value in loaded.items():
            self._db[key] = BST()
            self._db[key].insert_list(value)

    def save(self) -> None:
        """
        Save the storage unit to the file.
        """
        json = {key: value.list for key, value in self._db.items()}
        print(f"save {json}")
        self._io.save_json(json)

    def insert_field(self, field: str, value: list[float]) -> None:
        """
        Update a field

        Args:
            field: str
                The field to update.
            value: T
                The value to update.
        """
        if not self.check_field_exist(field):
            self._db[field] = BST()
        else:
            self._db[field].reset()

        self._db[field].insert_list(value)

    def insert_list(self, field: str, value: list[float]) -> None:
        """
        Insert a new value at existing field

        Args:
            field: str
                The field to update.
            value: list[float]
                The value to update.
        """
        if not self.check_field_exist(field):
            self._db[field] = BST()

        self._db[field].insert_list(value)

    def insert_single(self, field: str, value: float) -> None:
        """
        Insert a new value at existing field

        Args:
            field: str
                The field to update.
            value: float
                The value to update.
        """
        if not self.check_field_exist(field):
            self._db[field] = BST()
        self._db[field].insert(value)

    def inquire(self, field: str) -> list[float]:
        """
        Inquire field
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return None
        return self._db[field].list

    def delete_field(self, field: str) -> None:
        """
        Delete a field

        Args:
            field: str
                The field to delete.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return

        del self._db[field]

    def delete_list(self, field: str, value: list[float]) -> None:
        """
        Delete a list

        Args:
            field: str
                The field to delete.
            value: list[float]
                The list to delete.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return

        self._db[field].delete_list(value)

    def delete_single(self, field: str, value: float) -> None:
        """
        Delete a single value

        Args:
            field: str
                The field to delete.
            value: float
                The value to delete.
        """
        if not self.check_field_exist(field):
            print(f"[{self.label}]: Field {field} does not exist.")
            return

        self._db[field].delete(value)

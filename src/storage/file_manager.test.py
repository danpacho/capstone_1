import unittest
import os
import json
import numpy as np
from file_manager import JSONFileManager


class TestJSONFileManager(unittest.TestCase):
    def setUp(self):
        self.filename = os.path.join(os.getcwd(), "root.json")
        self.file_manager = JSONFileManager(self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_create(self):
        data = {"key": "value"}
        self.file_manager.create(data)
        self.assertTrue(os.path.exists(self.filename))
        with open(self.filename, "r") as file:
            saved_data = json.load(file)
        self.assertEqual(saved_data, data)

    def test_read(self):
        data = {"key": "value"}
        with open(self.filename, "w") as file:
            json.dump(data, file)
        loaded_data = self.file_manager.read()
        self.assertEqual(loaded_data, data)

    def test_update(self):
        initial_data = {"key": "value"}
        updated_data = {"new_key": "new_value"}
        with open(self.filename, "w") as file:
            json.dump(initial_data, file)
        self.file_manager.update(updated_data)
        with open(self.filename, "r") as file:
            saved_data = json.load(file)
        expected_data = initial_data.copy()
        expected_data.update(updated_data)
        self.assertEqual(saved_data, expected_data)

    def test_delete(self):
        with open(self.filename, "w") as file:
            file.write("test")
        self.file_manager.delete()
        self.assertFalse(os.path.exists(self.filename))

    def test_load_json(self):
        data = {"key": "value"}
        with open(self.filename, "w") as file:
            json.dump(data, file)
        loaded_data = self.file_manager.load_json()
        self.assertEqual(loaded_data, data)

    def test_save_json(self):
        data = {"key": "value"}
        self.file_manager.save_json(data)
        self.assertTrue(os.path.exists(self.filename))
        with open(self.filename, "r") as file:
            saved_data = json.load(file)
        self.assertEqual(saved_data, data)

    def test_update_root_path(self):
        root_path = os.path.join(os.getcwd(), "test.json")
        self.file_manager.update_root_path(root_path)
        self.assertEqual(self.file_manager.root_path, root_path)

    def test_save_and_load_numpy_array(self):
        data = np.array([1, 2, 3]).tolist()
        self.file_manager.save_json(data)
        loaded_data = self.file_manager.load_json()
        self.assertEqual(loaded_data, data)


if __name__ == "__main__":
    unittest.main()

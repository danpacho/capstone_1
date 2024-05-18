import os
import unittest
from src.storage.vector_storage import VectorStorage


class TestVectorStorage(unittest.TestCase):
    def setUp(self):
        self.storage = VectorStorage("test_vector_storage")

    def tearDown(self):
        os.remove(self.storage.root_path)
        self.storage = None

    def test_insert_field(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0])
        self.assertEqual(self.storage.inquire("test_field"), [1.0, 2.0, 3.0])
        self.storage.save()

    def test_insert_list(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0])
        self.storage.insert_list("test_field", [4.0, 5.0])
        self.assertEqual(self.storage.inquire("test_field"), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_insert_single(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0])
        self.storage.insert_single("test_field", 4.0)
        self.assertEqual(self.storage.inquire("test_field"), [1.0, 2.0, 3.0, 4.0])

    def test_delete_field(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0])
        self.storage.delete_field("test_field")
        self.assertIsNone(self.storage.inquire("test_field"))

    def test_delete_list(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0, 4.0, 5.0])
        self.storage.delete_list("test_field", [2.0, 3.0])
        self.assertEqual(self.storage.inquire("test_field"), [1.0, 4.0, 5.0])

    def test_delete_single(self):
        self.storage.insert_field("test_field", [1.0, 2.0, 3.0, 4.0, 5.0])
        self.storage.delete_single("test_field", 3.0)
        self.assertEqual(self.storage.inquire("test_field"), [1.0, 2.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()

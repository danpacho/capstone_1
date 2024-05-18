import unittest
import os

from src.storage.storage_unit import Storage


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.storage = Storage("test_base")

    def tearDown(self):
        self.storage.reset()

        os.remove(self.storage.root_path)
        self.storage = None
        # delete file

    def test_check_field_exist(self):
        self.assertFalse(self.storage.check_field_exist("field1"))
        self.storage.insert_field("field1", "value1")
        self.assertTrue(self.storage.check_field_exist("field1"))

    def test_inquire(self):
        self.storage.insert_field("field1", "value1")
        self.assertEqual(self.storage.inquire("field1"), "value1")

    def test_save(self):
        self.storage.insert_field("field1", "value1")
        self.storage.save()
        self.storage = Storage("test_base")
        self.assertTrue(self.storage.check_field_exist("field1"))

    def test_insert(self):
        self.storage.insert_field("field1", "value1")
        self.assertTrue(self.storage.check_field_exist("field1"))
        self.assertEqual(self.storage.inquire("field1"), "value1")

    def test_delete(self):
        self.storage.insert_field("field1", "value1")
        self.storage.delete_field("field1")
        self.assertFalse(self.storage.check_field_exist("field1"))

    def test_update(self):
        self.storage.insert_field("field1", "value1")
        self.storage.insert_field("field1", "new_value")
        self.assertEqual(self.storage.inquire("field1"), "new_value")

    def test_keys(self):
        self.storage.insert_field("field1", "value1")
        self.storage.insert_field("field2", "value2")
        self.assertEqual(self.storage.keys, ["field1", "field2"])


if __name__ == "__main__":
    unittest.main()

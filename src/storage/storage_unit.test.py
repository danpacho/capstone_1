import unittest
import os

from storage_unit import Storage


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.storage = Storage("test_storage")

    def tearDown(self):
        self.storage.reset()

        os.remove(self.storage.root_path)
        self.storage = None
        # delete file

    def test_check_field_exist(self):
        self.assertFalse(self.storage.check_field_exist("field1"))
        self.storage.insert("field1", "value1")
        self.assertTrue(self.storage.check_field_exist("field1"))

    def test_get_field(self):
        self.storage.insert("field1", "value1")
        self.assertEqual(self.storage.get_field("field1"), "value1")

    def test_save(self):
        self.storage.insert("field1", "value1")
        self.storage.save()
        self.storage = Storage("test_storage")
        self.assertTrue(self.storage.check_field_exist("field1"))

    def test_insert(self):
        self.storage.insert("field1", "value1")
        self.assertTrue(self.storage.check_field_exist("field1"))
        self.assertEqual(self.storage.get_field("field1"), "value1")

    def test_delete(self):
        self.storage.insert("field1", "value1")
        self.storage.delete("field1")
        self.assertFalse(self.storage.check_field_exist("field1"))

    def test_update(self):
        self.storage.insert("field1", "value1")
        self.storage.update("field1", "new_value")
        self.assertEqual(self.storage.get_field("field1"), "new_value")

    def test_keys(self):
        self.storage.insert("field1", "value1")
        self.storage.insert("field2", "value2")
        self.assertEqual(self.storage.keys, ["field1", "field2"])

    def test_inquire(self):
        self.storage.insert("field1", "value1")
        self.assertEqual(self.storage.inquire("field1"), "value1")
        self.assertIsNone(self.storage.inquire("field2"))


if __name__ == "__main__":
    unittest.main()

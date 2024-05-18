import os
from random import randrange
import unittest

from src.storage.stochastic_storage import StochasticStorage


class TestStochasticStorage(unittest.TestCase):
    def setUp(self):
        self.storage = StochasticStorage("test_storage")

    def tearDown(self):
        self.storage.reset()
        os.remove(self.storage.root_path)
        self.storage = None

    def test_pick_avg(self):
        # Test when field exists in the storage
        self.storage.insert_list("field1", [1, 2, 3, 4, 5])
        avg = self.storage.pick_avg("field1")
        self.assertEqual(avg, 3.0)

        # Test when field does not exist in the storage
        with self.assertRaises(KeyError):
            self.storage.pick_avg("field2")

    def test_pick_random(self):
        # Test when field exists in the storage
        self.storage.insert_list("field1", [1, 2, 3, 4, 5])
        random_value = self.storage.pick_random("field1")
        self.assertIn(random_value, [1, 2, 3, 4, 5])

        # Test when field does not exist in the storage
        with self.assertRaises(KeyError):
            self.storage.pick_random("field2")

    def test_pick_top5(self):
        # Test when field exists in the storage
        self.storage.insert_list("field1", [1, 2, 3, 4, 5])
        top5 = self.storage.pick_top5("field1")
        assert top5 == 5

        # Test when field does not exist in the storage
        with self.assertRaises(KeyError):
            self.storage.pick_top5("field2")

    def test_pick_bottom5(self):
        # Test when field exists in the storage
        self.storage.insert_list("field1", [1, 2, 3, 4, 5])
        bottom5 = self.storage.pick_bottom5("field1")
        assert bottom5 == 1

        # Test when field does not exist in the storage
        with self.assertRaises(KeyError):
            self.storage.pick_bottom5("field2")

    def test_huge_amount_of_data(self):
        # Test when inserting a large amount of data
        self.storage.insert_list("field1", list(randrange(100) for _ in range(100)))
        avg = self.storage.pick_avg("field1")
        print(avg)

        top5 = self.storage.pick_top5("field1")
        print(top5)

        bottom5 = self.storage.pick_bottom5("field1")
        print(bottom5)


if __name__ == "__main__":
    unittest.main()

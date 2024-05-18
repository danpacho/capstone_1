import numpy as np
import unittest
from src.pdf.gaussian import GaussianDistribution


class TestGaussianDistribution(unittest.TestCase):
    def setUp(self):
        self.sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.pdf = GaussianDistribution(self.sample)

    def test_pick_rand(self):
        rand_value = self.pdf.pick_rand()
        self.assertIn(rand_value, self.sample)

    def test_pick_avg(self):
        avg_value = self.pdf.pick_avg()
        self.assertEqual(avg_value, np.mean(self.sample))

    def test_pick_max(self):
        max_value = self.pdf.pick_max()
        self.assertEqual(max_value, np.max(self.sample))

    def test_pick_min(self):
        min_value = self.pdf.pick_min()
        self.assertEqual(min_value, np.min(self.sample))

    def test_pick_top5(self):
        top5_value = self.pdf.pick_top5()
        self.assertGreater(top5_value, self.pdf.pick_avg())

    def test_pick_bottom5(self):
        bottom5_value = self.pdf.pick_bottom5()
        self.assertLess(bottom5_value, self.pdf.pick_avg())

    def test_confidence_range(self):
        lower_bound, upper_bound = self.pdf.confidence_range()
        self.assertLessEqual(lower_bound, self.pdf.pick_avg())
        self.assertGreaterEqual(upper_bound, self.pdf.pick_avg())

    def test_update_distribution(self):
        new_sample = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        self.pdf.replace_sample(new_sample)
        self.assertEqual(self.pdf.sample.tolist(), new_sample.tolist())
        self.assertEqual(self.pdf.mean, np.mean(new_sample))
        self.assertEqual(self.pdf.std, np.std(new_sample))
        self.assertEqual(self.pdf.n, len(new_sample))


if __name__ == "__main__":
    unittest.main()

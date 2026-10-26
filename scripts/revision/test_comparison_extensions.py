import unittest
import numpy as np
from comparison_extensions import paired_system_difference


class AdditionalPairedContrastTests(unittest.TestCase):
    def test_all_reference_refits_and_common_draws_are_used(self):
        points = np.array([[.1], [.2], [.3], [.4], [.5], [.6]])
        samples = np.array([points, points+.01, points-.02])
        point, low, high = paired_system_difference(points, samples, [0, 1, 2], [3, 4, 5])
        np.testing.assert_allclose([point, low, high], .3)
        # Using only the first encoder seed would incorrectly give .4.
        self.assertNotAlmostEqual(point.item(), .4)

    def test_identical_systems_have_exactly_zero_contrast(self):
        points = np.array([[.1, .2], [.3, .4], [.5, .6]])
        samples = np.stack([points, points*.9, points*1.1])
        result = paired_system_difference(points, samples, [0, 1, 2], [0, 1, 2])
        for value in result:
            np.testing.assert_array_equal(value, np.zeros(2))


if __name__ == '__main__':
    unittest.main()

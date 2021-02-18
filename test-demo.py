import unittest
from demo import fill_polyline, sort_plants
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_fill_polyline(self):
        polyline = np.asarray([[1, 4],[5, 4]])
        expected_ret = np.asarray([[1,2,3,4],[5,5,4,4]])
        self.assertTrue(np.array_equal(fill_polyline(polyline), expected_ret))

        polyline = np.asarray([[1, 4, 6, 8], [5, 4, 4, 6]])
        expected_ret = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 4, 4, 4, 4, 5, 6]])
        self.assertTrue(np.array_equal(fill_polyline(polyline), expected_ret))

    def test_sort_plants(self):
        row = np.array([[0, 1],[1,1]])
        centroids = np.array([[5,4], [1,2], [4,4]])
        expected_ret = np.array([[1, 2], [4, 4], [5, 4]])
        ret, dist = sort_plants(row, centroids)
        self.assertTrue(np.array_equal(ret, expected_ret))


if __name__ == '__main__':
    unittest.main()

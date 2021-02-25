import unittest
from demo import fill_polyline, sort_plants, check_space, remove_parallel_cnt
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
        expected_ret = np.array([1, 2, 0])
        ret, dist = sort_plants(row, centroids)
        self.assertTrue(np.array_equal(ret, expected_ret))

    def test_check_space(self):
        start_r = ((2, 2), (4.0, 4.0), 0)
        end_r = ((7,7), (4.0, 4.0), 0)
        expect_start = np.asarray([4,4])
        expect_end = np.asarray([5, 5])
        ret_start, ret_end, min_pdist = check_space(start_r, end_r)
        self.assertTrue(np.array_equal(ret_start, expect_start))
        self.assertTrue(np.array_equal(ret_end, expect_end))
        self.assertAlmostEqual(min_pdist, np.sqrt(2))

    def test_remove_parallel_cnt(self):
        pass


if __name__ == '__main__':
    unittest.main()

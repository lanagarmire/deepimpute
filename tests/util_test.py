import unittest

import pandas as pd
import numpy as np

from deepimpute import util


class TestUtil(unittest.TestCase):
    """ """

    def test_log1x(self):
        self.assertEqual(util.log1x(0), 0)

    def test_exp1x(self):
        self.assertEqual(util.exp1x(0), 0)

    def test_get_maxes(self):
        dataframe = pd.DataFrame({"c1": [10, 2, 18, 5, 3]}, index=range(5))
        res = sorted(util.get_maxes(dataframe, 2))
        self.assertEqual(res, [0, 2])

    def test_get_input_genes(self):
        data = pd.DataFrame(np.identity(10))
        res = util.get_input_genes(data, [2, 5], targets=[[0, 1]])

    def test_get_target_genes(self):
        genes_quantiles = pd.Series([1, 15, 9, 10, 12])
        res = util.get_target_genes(genes_quantiles, NN_lim=2)
        self.assertEqual(sorted(res), [1, 4])


if __name__ == "__main__":
    unittest.main()

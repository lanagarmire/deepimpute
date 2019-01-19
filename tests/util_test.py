import unittest

# import pandas as pd
# import numpy as np

from deepimpute import util


class TestUtil(unittest.TestCase):
    """ """

    def test_log1x(self):
        self.assertEqual(util.log1x(0), 0)

    def test_exp1x(self):
        self.assertEqual(util.exp1x(0), 0)

    # def test_get_input_genes(self):
    #     data = pd.DataFrame(np.identity(10))
    #     res = util.get_input_genes(data, [2, 5], targets=[[0, 1]])

    # def test_get_target_genes(self):
    #     genes_quantiles = pd.Series([1, 15, 9, 10, 12])
    #     res = util._get_target_genes(genes_quantiles, minExpressionLevel=5, maxNumOfGenes=2)
    #     self.assertEqual(sorted(res), [1, 4])


if __name__ == "__main__":
    unittest.main()

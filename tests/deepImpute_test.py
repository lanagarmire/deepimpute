import unittest

import test_data
from deepimpute.deepImpute import deepImpute


# test sending data transposed


class TestDeepImpute(unittest.TestCase):
    """ """

    def test_all(self):
        _ = deepImpute(test_data.rawData, ncores=4, NN_lim=2000)

    def test_minExpressionLevel(self):
    	_ = deepImpute(test_data.rawData, ncores=4, noise=1, minPct=0.01)


if __name__ == "__main__":
    unittest.main()

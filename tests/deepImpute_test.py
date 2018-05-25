import unittest

import test_data
from deepimpute.deepImpute import deepImpute


# test sending data transposed


class TestDeepImpute(unittest.TestCase):
    """ """

    def test_all(self):
        rawData = test_data.rawData
        _ = deepImpute(rawData, ncores=4, NN_lim=2000)


if __name__ == "__main__":
    unittest.main()

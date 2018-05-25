import unittest

import numpy as np
import pandas as pd

import test_data
from deepimpute.maskedArrays import MaskedArray


class TestNet(unittest.TestCase):
    """ """

    def test_generate(self):
        rawData = test_data.rawData
        m_df = MaskedArray(data=rawData, dropout=0.1)
        m_df.generate()

    def test_getValues(self):
        data = pd.DataFrame(
            {"r1": np.arange(5), "r2": np.arange(5) ** 2, "r3": np.arange(5) ** 3}
        ).T
        binMask = [
            [True, True, False, False, True],
            [True, True, False, False, False],
            [False, True, True, True, True],
        ]
        maskedData = MaskedArray(data=data, mask=binMask)

        res_rows = [masked for masked in maskedData.getMasked(rows=True)]
        self.assertEqual(res_rows, [[2, 3], [4, 9, 16], [0]])

        res_cols = [masked for masked in maskedData.getMasked(rows=False)]
        self.assertEqual(res_cols, [[0], [], [2, 4], [3, 9], [16]])


if __name__ == "__main__":
    unittest.main()

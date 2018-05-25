import unittest
import numpy as np
import pandas as pd

from deepimpute.normalizer import Normalizer


class TestNormalizer(unittest.TestCase):
    """ """

    def test_normalizer(self):
        data = np.ones([3, 5])
        data[0, 2] = 9
        Norm = Normalizer(factorFn=np.sum, activations=[np.exp, np.log])
        Norm.fit(data)
        # print(data)
        data_df = pd.DataFrame(
            data,
            index=["r" + str(ii) for ii in range(data.shape[0])],
            columns=["c" + str(ii) for ii in range(data.shape[1])],
        )
        data_norm = Norm.transform(data_df)
        # print(data_norm)
        # print(Norm.transform(data_norm,rev=True))


if __name__ == "__main__":
    unittest.main()

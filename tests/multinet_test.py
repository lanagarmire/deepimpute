import unittest

import test_data
from deepimpute.multinet import MultiNet

# from utils_plot import train_test_scatter


class TestMultinet(unittest.TestCase):
    """ """

    def test_all(self):
        rawData = test_data.rawData
        idx = rawData.quantile(.99).sort_values(ascending=False).index[0:1300]
        rawData = rawData[idx]

        hyperparams = {
            "architecture": [
                {"type": "dense", "activation": "relu", "neurons": 150},
                {"type": "dropout", "activation": "dropout", "rate": 0.2},
            ],
            "loss": "wMSE",
            "sub_outputdim": 512,
            "seed": 123,
            "ncores": 2,
            "verbose": 1
        }

        model = MultiNet(**hyperparams)
        model.fit(rawData)
        _ = model.predict(rawData,policy="restore")

        print(model.test_metrics)


if __name__ == "__main__":
    unittest.main()

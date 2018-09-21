import unittest

import test_data
from deepimpute.multinet import MultiNet

# from utils_plot import train_test_scatter


class TestMultinet(unittest.TestCase):
    """ """

    def test_all(self):
        rawData = test_data.rawData
        idx = rawData.quantile(.99).sort_values(ascending=False).index[0:900]
        rawData = rawData[idx]

        hyperparams = {
            "layers": [
                {"label": "dense", "activation": "relu", "nb_neurons": 150},
                {"label": "dropout", "activation": "dropout", "rate": 0.2},
                {"label": "dense", "activation": "relu"},
            ],
            "loss": "wMSE",
            "optimizer": "Adam",
            "dims": [20, 500],
            "preproc": "log_or_exp",
            "seed": 1,
            "ncores": 4,
        }

        model = MultiNet(**hyperparams)
        model.fit(rawData)
        _ = model.predict(rawData)

        print(model.score(rawData))


if __name__ == "__main__":
    unittest.main()

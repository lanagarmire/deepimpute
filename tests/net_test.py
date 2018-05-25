import unittest
import numpy as np

import test_data
from deepimpute.net import Net


class TestNet(unittest.TestCase):
    """ """

    def test_init_works(self):
        net = Net()
        net.set_params(
            max_epochs=50,
            learning_rate=1e-3,
            batch_size=50,
            layer2={"label": "dense", "activation": "relu", "nb_neurons": 50},
            ncores=4,
        )

    def test_preprocess(self):
        rawData = test_data.rawData

        idx = rawData.quantile(.99).sort_values(ascending=False).index[0:2000]
        rawData = np.log10(1 + rawData[idx])

        hyperparams = {
            "layers": [
                {"label": "dense", "activation": "relu", "nb_neurons": 100},
                {"label": "dropout", "activation": "dropout", "rate": 0.15},
                {"label": "dense", "activation": "relu"},
            ],
            "n_cores": 6,
        }

        model = Net(**hyperparams)
        model.fit(rawData)
        _ = model.predict(rawData)
        print(model.score(rawData))


if __name__ == "__main__":
    unittest.main()

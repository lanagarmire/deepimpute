import unittest
import numpy as np

import test_data
from deepimpute.net import Net

class TestNet(unittest.TestCase):
    """ """

    def test_init_works(self):
        net = Net(dims=[400,400],
                  max_epochs=50,
                  learning_rate=1e-3,
                  batch_size=50,
                  architecture = [{"type": "dense", "activation": "relu", "neurons": 50}],
                  ncores=4,
        )
        return net

    def test_preprocess(self):
        rawData = test_data.rawData

        idx = rawData.quantile(.99).sort_values(ascending=False).index[0:2000]
        rawData = np.log10(1 + rawData[idx])

        hyperparams = {
            "architecture": [
                {"type": "dense", "activation": "relu", "neurons": 10},
                {"type": "dropout", "activation": "dropout", "rate": 0.15},
            ],
            "ncores": 3,
            "max_epochs": 50
        }

        X,Y = rawData.iloc[:,:50],rawData.iloc[:,:20],

        model = Net(dims=[X.shape[1],Y.shape[1]],**hyperparams)
        model.fit(X,Y,verbose=1)
        print(model.score(X,Y))


if __name__ == "__main__":
    unittest.main()

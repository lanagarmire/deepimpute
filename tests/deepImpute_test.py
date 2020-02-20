import unittest
from unittest import mock
import argparse

import test_data
from deepimpute.deepImpute import deepImpute

args = {
    'inputFile': test_data.PATH+"csv",
    'cell_axis': "rows",
    'cores': 1,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'max_epochs': 300,
    'output_neurons': 512,
    'hidden_neurons': 300,
    "dropout_rate": 0.2,
    "subset": 1,
    "limit": 1000,
    "minVMR": 0.5,
    "n_pred": None,
    "policy": "restore",
    "output": None
}

class TestDeepImpute(unittest.TestCase):
    """ """

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(**args))
    def test_all(self, mock_args):
        _ = deepImpute()

if __name__ == "__main__":
    unittest.main()

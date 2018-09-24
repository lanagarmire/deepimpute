import os

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense,Dropout
import tensorflow as tf

from util import get_int, set_int, get_input_genes

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.INFO)


class Net:

    def __init__(
        self,
        dims=[20, 500],
        NN_id=None,
        n_cores=1,
        seed=1234,
        layers=None,
        **kwargs
    ):
        self.model = None
        self.seed = int(seed)
        self.n_cores = n_cores
        self.NNid = NN_id

        # Some Network Parameters
        self.step = 0
        self.max_epochs = 300
        self.loss = "mean_squared_error"
        self.optimizer = "AdamOptimizer"
        self.learning_rate = 5e-4
        self._dims = dims
        self.batch_size = 64

        # Default layers
        if layers is None:
            layers = [
                {"activation": "relu", "nb_neurons": 300},
                {"activation": "dropout", "rate": 0.5},
                {"activation": None},
            ]
        self.layers = layers

    """Define some properties to control attributes value and assignment"""

    dims = property(get_int("_dims"), set_int("_dims"))
    # max_epochs = property(get_int("_max_epochs"), set_int("_max_epochs"))
    # batch_size = property(get_int("_batch_size"), set_int("_batch_size"))

    def _build(self):
        input_dim,output_dim = self.dims
        self.layers[-1]['nb_neurons'] = output_dim
        
        self.model = Sequential()
        for i,layer in enumerate(self.layers):
            if 'rate' in layer.keys():
                self.model.add(Dropout(layer['rate']))
            else:
                self.model.add(Dense(layer['nb_neurons'],
                                     input_dim=input_dim,
                                     activation=layer['activation']))
            input_dim = None

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _fit(self,X,Y):
        self.dims = [X.shape[1],Y.shape[1]]
        if self.model is None:
            self._build()
        self.model.fit(X,Y,batch_size=self.batch_size, nb_epoch=self.max_epochs)

    def fit(
        self,
        X,
        targetGenes=None,
        predictorGenes=None,
        dists=None,
        cell_thresh=0.1,
        labels=None,
        retrieve_training=False,
        **params
    ):  # Extract features and start training

        if labels is not None:
            data = pd.DataFrame(
                np.reshape(X, list(map(len, labels))),
                index=labels[0],
                columns=labels[1],
            )
        else:
            data = X

        if targetGenes is not None:
            if len(targetGenes) < self.dims[1]:
                self._dims[1] = len(targetGenes)

        if predictorGenes is None:
            self.predictorGenes, self.targetGenes = get_input_genes(
                data,
                self.dims,
                distanceMatrix=dists,
                targets=targetGenes,
                predictorLimit=None,
            )[0]
        else:
            self.predictorGenes, self.targetGenes = predictorGenes, targetGenes

        filt = (data[self.targetGenes] > 0).sum(axis=1) >= self.dims[1] * cell_thresh

        features, targets = (
            data.loc[filt, self.predictorGenes].values,
            data.loc[filt, self.targetGenes].values,
        )

        self._fit(features, targets)

        
    def predict(self,data):
        predictions = self.model.predict(data[self.predictorGenes].values)

        return pd.DataFrame(predictions,
                            columns=data.columns,
                            index=data.index)

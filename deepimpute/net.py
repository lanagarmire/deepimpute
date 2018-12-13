import os
import numpy as np
import pandas as pd

from keras.models import Sequential,model_from_json
from keras.layers import Dense,Dropout,Activation
from keras import optimizers
from keras.callbacks import EarlyStopping

import binascii
from sklearn.metrics import r2_score
from multiprocessing import cpu_count
import tempfile

from deepimpute.util import get_int, set_int, get_input_genes
from deepimpute.util import wMSE, poisson_loss
from deepimpute.util import score_model

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Net(object):

    def __init__(
        self,
        dims=[20, 500],
        NN_id=None,
        n_cores=1,
        seed=1234,
        layers=None,
        runDir=os.path.join(tempfile.gettempdir(), "run"),
        **kwargs
    ):
        self.seed = int(seed)
        self.runDir = runDir
        self.n_cores = n_cores
        self.NNid = NN_id
        self._sessionDir = None

        # Some Network Parameters
        self.step = 0
        self._max_epochs = 500
        self.loss = "wMSE"
        self.optimizer = "Adam"
        self.learning_rate = 1e-4
        self._dims = dims
        self._batch_size = 64

        # Default layers
        if layers is None:
            layers = [
                {"label": "dense", "activation": "relu", "nb_neurons": 128},
                {"label": "dropout", "activation": "dropout", "rate": 0.2},
                {"label": "dense", "activation": "softplus"},
            ]
        self.layers = layers

        """ Builds the whole neural network model based on the configuration file (if any)"""
        self.set_params(**kwargs)

    """Define some properties to control attributes value and assignment"""

    dims = property(get_int("_dims"), set_int("_dims"))
    max_epochs = property(get_int("_max_epochs"), set_int("_max_epochs"))
    batch_size = property(get_int("_batch_size"), set_int("_batch_size"))

    @property
    def sessionDir(self):
        if self._sessionDir is None:
            self.sessionDir = None
        return self._sessionDir

    @sessionDir.setter
    def sessionDir(self, value):
        if value is None:
            self._sessionDir = os.path.join(self.runDir, self.NNid)
        else:
            self._sessionDir = value
        if not os.path.exists(self.sessionDir):
            os.makedirs(self.sessionDir)

    def get_params(self, deep=False):
        return self.__dict__

    def display_params(self):
        for nb,layer in enumerate(self.layers):
            pretty_display = ['{}: {}'.format(k,v) for (k,v) in layer.items()]
            print('Layer-{}'.format(nb),pretty_display)
        print('Batch size',self.batch_size)
        print('Learning rate',self.learning_rate)
        print('Loss function',self.loss)        

    # Load parameters from the configuration file
    def set_params(self, **params):
        if self.n_cores == "all":
            self.n_cores = cpu_count()

        for key, par in params.items():
            if type(par) is dict:
                setattr(self, key, par.copy())
            else:
                setattr(self, key, par)
        self._check_layer_params()

        try:
            self.loss = eval(self.loss)
        except:
            pass

        if self.NNid == "auto":
            rand = binascii.b2a_hex(os.urandom(3))

            if type(rand) is bytes:
                rand = rand.decode()
            
            self.NNid = "lr={}_bs={}_dims={}-{}_nodes={}_dp={}_{}".format(
                self.learning_rate,
                self.batch_size,
                self.dims[0],
                self.dims[1],
                self.layers[0]["nb_neurons"],
                self.layers[1]["rate"],
                rand,
            )
        return self

    def _check_layer_params(self):
        for layer in self.layers:
            if "nb_neurons" in layer.keys() and type(layer["nb_neurons"]) is not int:
                print(
                    "Warning: nb_neurons must be an integer ({})".format(
                        layer["nb_neurons"]
                    )
                )
                layer["nb_neurons"] = int(layer["nb_neurons"])
        if layer["label"] == "dropout":
            self.layers += [{"label": "dense", "activation": "relu"}]
        if "nb_neurons" in layer.keys():
            if layer["nb_neurons"] != self.dims[1]:
                print("Addind a dense layer to connect to output neurons")
                self.layers += [{"label": "dense", "activation": "relu"}]

    def _build(self, inputDim, outputDim, trainable=True):  # build network
        """ Build the whole network """

        model = Sequential()

        for layer in self.layers:
            if layer["label"] == "dense" and "nb_neurons" not in layer.keys():
                layer["nb_neurons"] = outputDim

            if layer["label"] == "dense":
                model.add(Dense(layer["nb_neurons"]))
                if layer["activation"] is not None:
                    model.add(Activation(layer["activation"]))
            elif layer["label"] == "dropout":
                model.add(Dropout(layer["rate"]))
            else:
                print("Unknown layer type. Aborting.")
                exit(1)

        optimizer = getattr(optimizers,self.optimizer)
        model.compile(optimizer=optimizer(lr=self.learning_rate),
                      loss=self.loss)
        return model
        

    def _fit(self, features, targets, retrieve_training=False):
        """ Network training """

        test_samples = np.random.choice(targets.shape[0],
                                        int(targets.shape[0]/10.),
                                        replace=False)
        train_samples = np.setdiff1d(range(targets.shape[0]),
                                     test_samples)
        
        model = self._build(features.shape[1],targets.shape[1])
        model.fit(features[train_samples,:],
                  targets[train_samples,:],
                  epochs=self.max_epochs,
                  batch_size=self.batch_size,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                  validation_data=(features[test_samples,:],targets[test_samples,:]),
                  verbose=0
        )

        # serialize model to JSON
        model_json = model.to_json()
        
        with open("{}/model.json".format(self.sessionDir), "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        model.save_weights("{}/model.h5".format(self.sessionDir))
        print("Saved model to disk")
        
        return model

    def fit(
        self,
        X,
        targetGenes=None,
        predictorGenes=None,
        dists=None,
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

        if not retrieve_training:
            self.set_params(NNid="auto", **params)

        if targetGenes is not None:
            if len(targetGenes) < self.dims[1]:
                self._dims[1] = len(targetGenes)

        if predictorGenes is None:
            self.predictorGenes, self.targetGenes = get_input_genes(
                data,
                self.dims[1],
                nbest=self.dims[0],
                distanceMatrix=dists,
                targets=targetGenes
            )[0]
        else:
            self.predictorGenes, self.targetGenes = predictorGenes, targetGenes
        
        features, targets = (
            data.loc[:, self.predictorGenes].values,
            data.loc[:, self.targetGenes].values,
        )

        model = self._fit(features, targets, retrieve_training=retrieve_training)

        return model
        

    def predict(
        self, X_test, saver_path="", checkpoint=None
    ):  # Prediction on (new) dataset

        # load json and create model
        json_file = open('{}/model.json'.format(self.sessionDir), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('{}/model.h5'.format(self.sessionDir))
        
        Y_impute = model.predict(X_test[self.predictorGenes])
        
        if np.sum(np.isnan(Y_impute)) > 0:
            print("Removing NaN values")
            Y_impute = Y_impute.fillna(0)
        return pd.DataFrame(Y_impute, index=X_test.index, columns=self.targetGenes)

    def score(self, X, metric=r2_score):
        print("Scoring model by masking the matrix.")
        return score_model(self, X, metric=metric)

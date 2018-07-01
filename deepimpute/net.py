import os
import numpy as np
import pandas as pd
import tensorflow as tf
import binascii
from sklearn.metrics import r2_score
from multiprocessing import cpu_count
from math import ceil
from collections import deque
import tempfile

from deepimpute.util import get_int, set_int, get_input_genes
from deepimpute.util import score_model

DISPLAY_STEPS = np.arange(0, 1000, 100)
PRETRAINING = True
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.INFO)


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
        self._max_epochs = 300
        self.loss = "mean_squared_error"
        self.optimizer = "AdamOptimizer"
        self.learning_rate = 5e-4
        self._dims = dims
        self._batch_size = 64

        # Default layers
        if layers is None:
            layers = [
                {"label": "dense", "activation": "relu", "nb_neurons": 300},
                {"label": "dropout", "activation": "dropout", "rate": 0.5},
                {"label": "dense", "activation": "relu"},
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

    def _build_next_layer(self, current_layer, layer, trainable):
        """ Build one layer on top of the last one"""

        if layer["label"] == "dropout":
            current_layer = tf.layers.dropout(
                current_layer,
                rate=layer["rate"],
                training=tf.get_collection("placeholders")[2],
                name="dropout",
            )
        else:
            current_layer = tf.layers.dense(
                current_layer,
                layer["nb_neurons"],
                activation=getattr(tf.nn, layer["activation"]),
                use_bias=True,
                trainable=trainable,
                name="dense",
            )
        tf.summary.histogram("activations", current_layer)
        return current_layer

    def _build(self, inputDim, outputDim, trainable=True):  # build network
        """ Build the whole network """
        input_placeholder = tf.placeholder(
            tf.float32, shape=[None, inputDim], name="rawInput"
        )
        output_placeholder = tf.placeholder(
            tf.float32, shape=[None, outputDim], name="rawOutput"
        )
        phase_placeholder = tf.placeholder(tf.bool, name="dropoutActivation")
        current_layer = input_placeholder
        tf.add_to_collection("placeholders", input_placeholder)
        tf.add_to_collection("placeholders", output_placeholder)
        tf.add_to_collection("placeholders", phase_placeholder)

        with tf.variable_scope("Epoch", reuse=tf.AUTO_REUSE):
            step = tf.get_variable(
                "Epoch", initializer=self.step, trainable=False, dtype=tf.int32
            )
        tf.add_to_collection("step", step)

        # print('Input layer: {}'.format(inputDim))

        for layer_nb, layer in enumerate(self.layers):
            with tf.variable_scope("Layer-{}".format(layer_nb), reuse=tf.AUTO_REUSE):
                if layer["label"] == "dense" and "nb_neurons" not in layer.keys():
                    trainable = True
                    layer["nb_neurons"] = outputDim
                current_layer = self._build_next_layer(current_layer, layer, trainable)
            # print('Layer-{}'.format(layer_nb),
            #      ', '.join(['{}: {}'.format(k, v) for k, v in layer.items()]))

        tf.add_to_collection("outputLayer", current_layer)

        with tf.variable_scope("ops", reuse=tf.AUTO_REUSE):
            self.loss_op = getattr(tf.losses, self.loss)(
                output_placeholder,
                tf.get_collection("outputLayer")[0],
                weights=output_placeholder,
            )
            self.train_op = getattr(tf.train, self.optimizer)(
                learning_rate=self.learning_rate
            ).minimize(self.loss_op)
        tf.summary.scalar("Loss", self.loss_op)

        tf.add_to_collection("ops", self.train_op)
        tf.add_to_collection("ops", self.loss_op)

    def _fit(self, features, targets, retrieve_training=False):
        """ Network training """

        train_graph = tf.Graph()

        with train_graph.as_default():
            tf.set_random_seed(self.seed)

            if retrieve_training:
                save_path = tf.train.latest_checkpoint(self.sessionDir)
                self.saver = tf.train.import_meta_graph(save_path+'.meta')
            else:
                self._build(
                    trainable=True, inputDim=features.shape[1], outputDim=targets.shape[1]
                )
                self.saver = tf.train.Saver(max_to_keep=1)

            summary_writer = tf.summary.FileWriter(
                self.sessionDir, tf.get_default_graph()
            )
            
            merged_summaries = tf.summary.merge_all()
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                device_count={"CPU": self.n_cores},
                intra_op_parallelism_threads=self.n_cores,
                inter_op_parallelism_threads=self.n_cores,
            )

            with tf.Session(config=session_conf, graph=train_graph) as sess:
                if retrieve_training:
                    self.saver.restore(sess,save_path)
                else:
                    tfGlobals = tf.global_variables_initializer()
                    sess.run(tfGlobals)
                    
                step_init = self.step

                error_buffer = deque([1e5] * 10)

                n_runs_per_epoch = int(ceil(features.shape[0] / self.batch_size))

                while self.step - step_init < self.max_epochs and error_buffer[0] <= np.mean(list(error_buffer)[1:]):
                    epoch_error = 0

                    # Run through each mini-batch
                    shuff = np.random.choice(
                        targets.shape[0], targets.shape[0], replace=False
                    )

                    for i in range(n_runs_per_epoch):
                        indices = range(
                            self.batch_size * i,
                            min(self.batch_size * (i + 1), targets.shape[0]),
                        )
                        feedDict = {
                            tf.get_collection("placeholders")[0]: features[
                                shuff[indices], :
                            ],
                            tf.get_collection("placeholders")[1]: targets[
                                shuff[indices], :
                            ],
                            tf.get_collection("placeholders")[2]: True,
                        }

                        # Optimization and extract metrics
                        ops, summary = sess.run(
                            [tf.get_collection("ops"), merged_summaries],
                            feed_dict=feedDict,
                        )
                        epoch_error += ops[1]

                    epoch_error /= n_runs_per_epoch

                    error_buffer.appendleft(epoch_error)
                    error_buffer.pop()

                    # Save session every epoch except the last one
                    if error_buffer[0] <= np.mean(list(error_buffer)[1:]):
                        self.step += 1
                        summary_writer.add_summary(summary, self.step)
                        self.saver.save(
                            sess,
                            "/".join([self.sessionDir, "Checkpoint"]),
                            global_step=self.step,
                        )

                print("Training finished after {} epochs.".format(self.step-step_init))

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

        if not retrieve_training:
            self.set_params(NNid="auto", **params)

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

        self._fit(features, targets, retrieve_training=retrieve_training)

    def predict(
        self, X_test, saver_path="", checkpoint=None
    ):  # Prediction on (new) dataset
        """ Test network on X """
        if checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(self.sessionDir)
        else:
            checkpoint = self.sessionDir + "/Checkpoint-" + str(checkpoint)

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            device_count={"CPU": self.n_cores},
            intra_op_parallelism_threads=self.n_cores,
            inter_op_parallelism_threads=self.n_cores,
        )
        test_graph = tf.Graph()
        with test_graph.as_default():
            self.saver = tf.train.import_meta_graph(checkpoint + ".meta")

            feed_dict = {
                tf.get_collection("placeholders")[0]: X_test[self.predictorGenes],
                tf.get_collection("placeholders")[2]: False,
            }

        # Start session
        with tf.Session(config=session_conf, graph=test_graph) as sess:
            self.saver.restore(sess, checkpoint)

            Y_impute = sess.run(
                tf.get_collection("outputLayer")[0], feed_dict=feed_dict
            )

        if np.sum(np.isnan(Y_impute)) > 0:
            print("Removing NaN values")
            Y_impute = Y_impute.fillna(0)
        return pd.DataFrame(Y_impute, index=X_test.index, columns=self.targetGenes)

    def score(self, X, metric=r2_score):
        print("Scoring model by masking the matrix.")
        return score_model(self, X, metric=metric)

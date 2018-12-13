import os
import numpy as np
import pandas as pd
import binascii
import warnings
import tempfile
from math import ceil
from multiprocessing import cpu_count, sharedctypes
from multiprocessing.pool import Pool
from sklearn.metrics import r2_score

from deepimpute.net import Net
from deepimpute.normalizer import Normalizer
from deepimpute.util import get_input_genes, _get_target_genes
from deepimpute.util import score_model


def _newCoreInitializer(arr_to_populate):
    global sharedArray
    sharedArray = arr_to_populate


def _trainNet(in_out, NN_param_i, data_i, labels, retrieve_training=False):
    features, targets = in_out

    net = Net(**NN_param_i)
    net.fit(data_i, targetGenes=targets, predictorGenes=features, labels=labels, retrieve_training=retrieve_training)

    # retrieve the array
    params = list(NN_param_i.keys()) + ["targetGenes", "NNid", "predictorGenes"]
    args2return = [(attr, getattr(net, attr)) for attr in params]
    return {k: v if k[0] != "_" else (k[1:], v) for k, v in args2return}


def _predictNet(data_i, NN_param_i, labels):
    net = Net(**NN_param_i)
    data_i_ok = pd.DataFrame(
        np.reshape(data_i, list(map(len, labels))), index=labels[0], columns=labels[1]
    )
    return net.predict(data_i_ok)

def trainOrPredict(args):
    if len(args) == 5:
        in_out, NN_param_i, labels, mode, retrieve_training = args
    else:
        in_out, NN_param_i, labels, mode = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data_i = np.ctypeslib.as_array(sharedArray)
    if mode == "predict":
        return _predictNet(data_i, NN_param_i, labels)
    return _trainNet(in_out, NN_param_i, data_i, labels, retrieve_training=retrieve_training)

class MultiNet(object):

    def __init__(
        self,
        n_cores=4,
        minExpressionLevel=0.05,
        normalization="log_or_exp",
        runDir=os.path.join(tempfile.gettempdir(), "run"),
        seed=0,
        **NN_params
    ):
        self.maxcores = min(n_cores, cpu_count())
        self.inOutGenes = None
        self.norm = normalization
        self.runDir = runDir
        self.seed = seed

        NN_params["seed"] = seed
        if "dims" not in NN_params.keys():
            NN_params["dims"] = [20, 512]
        self.NN_params = NN_params
        self.trainingParams = None

    def _setIDandRundir(self, data):
        # set runID
        runID = binascii.b2a_hex(os.urandom(5))
        if type(runID) is bytes:
            runID = runID.decode()
        self.NN_params["runDir"] = os.path.join(self.runDir, str(runID))

    def _getRunsAndCores(self, geneCount):
        n_runs = int(ceil(1. * geneCount / self.NN_params["dims"][1]))
        n_cores = min(self.maxcores, n_runs)
        self.NN_params["n_cores"] = max(1, int(self.maxcores / n_cores))
        return n_runs, n_cores

    def fit(self, data, NN_lim="auto", minExpressionLevel=.05, cell_subset=None, targetGeneNames=None, retrieve_training=False, predictorLimit=2000):
        np.random.seed(seed=self.seed)
        
        inputExpressionMatrixDF = pd.DataFrame(data)
        print("Input dataset is {} genes (columns) and {} cells (rows)".format(inputExpressionMatrixDF.shape[1], inputExpressionMatrixDF.shape[0]))
        print("First 3 rows and columns:")
        print(inputExpressionMatrixDF.iloc[0:3, 0:3])

        self._setIDandRundir(inputExpressionMatrixDF)

        # Change the output dimension if the data has too few genes
        if inputExpressionMatrixDF.shape[1] < self.NN_params["dims"][1]:
            self.NN_params["dims"][1] = inputExpressionMatrixDF.shape[1]

        subnetOutputColumns = self.NN_params["dims"][1]
        
        # Choose genes to impute
        if targetGeneNames is None:
            geneMetric = (inputExpressionMatrixDF > 0).mean()
            targetGeneNames = _get_target_genes(geneMetric, minExpressionLevel=minExpressionLevel, maxNumOfGenes = NN_lim)

        df_to_impute = inputExpressionMatrixDF[targetGeneNames]

        numberOfTargetGenes = len(targetGeneNames)
        if (numberOfTargetGenes == 0):
            raise Exception("Unable to compute any target genes. Is your data log transformed? Perhaps try with a lower minExpressionLevel.")

        n_runs, n_cores = self._getRunsAndCores(numberOfTargetGenes)

        # ------------------------# Subnetworks #------------------------#

        n_choose = int(numberOfTargetGenes / subnetOutputColumns)

        subGenelists = np.random.choice(
            targetGeneNames, [n_choose, subnetOutputColumns], replace=False
        ).tolist()
        
        if n_choose < n_runs:
            # Special case: for the last run, the output layer will have previous targets
            selectedGenes = np.reshape(subGenelists, -1)
            leftOutGenes = np.setdiff1d(targetGeneNames, selectedGenes)

            fill_genes = np.random.choice(targetGeneNames,
                                          subnetOutputColumns-len(leftOutGenes),
                                          replace=False)

            subGenelists.append(np.concatenate([leftOutGenes,fill_genes]).tolist())

        # ------------------------# Extracting input genes #------------------------#

        corrMatrix = 1 - np.abs(
            pd.DataFrame(np.corrcoef(df_to_impute.T), index=targetGeneNames, columns=targetGeneNames)
        )
        
        if self.inOutGenes is None:
            self.inOutGenes = get_input_genes(
                df_to_impute,
                self.NN_params["dims"][1],
                nbest=self.NN_params["dims"][0],
                distanceMatrix=corrMatrix,
                targets=subGenelists,
                predictorLimit=predictorLimit
            )

        # ------------------------# Subsets for fitting #------------------------#

        n_cells = df_to_impute.shape[0]

        if type(cell_subset) is float or cell_subset == 1:
            n_cells = int(cell_subset * n_cells)

        elif type(cell_subset) is int:
            n_cells = cell_subset

        self.trainCells = df_to_impute.sample(n_cells, replace=False).index

        print(
            "Starting training with {} cells ({:.1%}) on {} threads ({} cores/thread).".format(
                n_cells,
                1. * n_cells / df_to_impute.shape[0],
                n_cores,
                self.NN_params["n_cores"],
            )
        )

        if self.trainingParams is None:
            self.trainingParams = [self.NN_params]*len(self.inOutGenes)

        # -------------------# Preprocessing (if any) #--------------------#

        normalizer = Normalizer.fromName(self.norm)

        df_to_impute = normalizer.fit(df_to_impute).transform(df_to_impute)

        # -------------------# Share matrix between subprocesses #--------------------#

        """ Create memory chunk and put the matrix in it """
        idx, cols = self.trainCells, df_to_impute.columns
        trainData = df_to_impute.loc[self.trainCells, :].values

        """ Parallelize process with shared array """
        childJobs = [
            (in_out, trainingParams, (idx, cols), "train", retrieve_training)
            for in_out,trainingParams in zip(self.inOutGenes,self.trainingParams)
        ]

        self.trainingParams = self._runOnMultipleCores(n_cores, trainData.flatten(), childJobs)

        self.networks = []
        for dictionnary in self.trainingParams:
            self.networks.append(Net(**dictionnary))

        print('---- Hyperparameters summary ----')
        self.networks[0].display_params()

        return self

    def _runOnMultipleCores(self, cores, data, childJobs):
        sharedArray = sharedctypes.RawArray("d", data)

        pool = Pool(
            processes=cores, initializer=_newCoreInitializer, initargs=(sharedArray,)
        )
        
        output_dicts = pool.map(trainOrPredict, childJobs)
        pool.close()
        pool.join()
        return output_dicts

    def predict(self, data, imputed_only=False, policy="restore"):
        print("Starting prediction")
        df = pd.DataFrame(data)
        normalizer = Normalizer.fromName(self.norm)

        """ Create memory chunk and put the matrix in it """
        idx, cols = df.index, df.columns
        df_norm = normalizer.fit(df).transform(df)

        """ Parallelize process with shared array """
        childJobs = [
            ((12, 15), net.__dict__, (idx, cols), "predict") for net in self.networks
        ]

        output_dicts = self._runOnMultipleCores(self.maxcores, df_norm.values.flatten(), childJobs)

        Y_imputed = pd.concat(output_dicts, axis=1)
        Y_imputed = Y_imputed.groupby(by=Y_imputed.columns,axis=1).mean()

        Y_imputed = Y_imputed.mask(Y_imputed > df_norm.values.max(), df_norm[Y_imputed.columns])
        
        Y_imputed = normalizer.transform(Y_imputed,rev=True)
        
        Y_not_imputed = df.drop(Y_imputed.columns,axis=1)
        
        Y_total = pd.concat([Y_imputed, Y_not_imputed], axis=1)[df.columns]
        
        if policy == "restore":
            Y_total = Y_total.mask(df > 0, df)
        elif policy == "max":
            Y_total = pd.concat([Y_total,df]).max(level=0)
        else:
            Y_total = Y_total.mask(Y_total==0,df)
            
        if imputed_only:
            Y_total = Y_total[Y_imputed.columns]

        if type(data) == type(pd.DataFrame()):
            return Y_total
        else:
            return Y_total.values

    def score(self, data, metric=r2_score):
        imputedGenes = list(zip(*[net.targetGenes for net in self.networks]))
        return score_model(self, pd.DataFrame(data), metric=r2_score, cols=imputedGenes)

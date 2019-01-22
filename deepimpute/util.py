import numpy as np
import pandas as pd

from deepimpute.maskedArrays import MaskedArray

""" Preprocessing functions """


def log1x(x):
    return np.log(1 + x)


def exp1x(x):
    return np.exp(x) - 1


def libNorm(scale=10000):
    def _libNorm(x):
        return scale / np.sum(x)
    return _libNorm

def set_int(name):

    def setter_wrapper(self, value):
        if type(np.prod(value)) is not np.int64:
            print("Wrong value for {}={}. Converting to integer.".format(name, value))
            if np.array(value).size == 1:
                setattr(self, name, int(value))
            else:
                setattr(self, name, [el for el in map(int, list(value))])
        else:
            setattr(self, name, value)

    return setter_wrapper


def get_int(name):

    def getter_wrapper(self):
        out = getattr(self, name)
        if np.array(out).size == 1:
            return int(out)
        else:
            return [el for el in map(int, out)]

    return getter_wrapper

# def get_input_genes(
#         dataframeToImpute, dims, distanceMatrix=None, targets=None, seed=1234 #predictorDropoutLimit=.99,seed=1234
# ):
#     #geneDropoutRate = (dataframeToImpute==0).mean()
#     potential_predictors = dataframeToImpute.columns #geneDropoutRate.index[geneDropoutRate < predictorDropoutLimit]

#     print("Keeping {} potential predictors.".format(len(potential_predictors)))
    
#     if targets is None:
#         np.random.seed(seed)
#         targets = [np.random.choice(dataframeToImpute.columns, dims[1], replace=False)]

#     if distanceMatrix is None:
#         distanceMatrix = pd.DataFrame(
#             np.abs(np.corrcoef(dataframeToImpute.T)),
#             index=dataframeToImpute.columns, columns=dataframeToImpute.columns
#         )[potential_predictors]
#     in_out_genes = []

#     max_limit = dims[0]
#     for genes in targets:
#         pred_to_rmv = np.setdiff1d(potential_predictors,targets)
#         subMatrix = distanceMatrix.loc[genes].drop(pred_to_rmv,axis=1)
#         sorted_idx = np.argsort(-subMatrix.values,axis=1)
#         predictorGenes = subMatrix.columns[sorted_idx[:,:max_limit]].values.flatten()        
#         in_out_genes.append((predictorGenes, genes))
#     return in_out_genes


# def _get_target_genes(geneQuantiles, minExpressionLevel, maxNumOfGenes):
#     if maxNumOfGenes == "auto":
#         targetGenes = geneQuantiles[geneQuantiles > minExpressionLevel].index
#         print("Minimum gene count for imputation set to {}, leaving {} genes for imputation."
#               .format(minExpressionLevel,len(targetGenes)))

#     else:
#         if maxNumOfGenes is None:
#             maxNumOfGenes = len(geneQuantiles)
#         maxNumOfGenes = min(maxNumOfGenes, len(geneQuantiles))
#         targetGenes = geneQuantiles.sort_values(ascending=False).index[:maxNumOfGenes]
#         print("Gene prediction limit set to {} genes".format(len(targetGenes)))

#     return targetGenes.tolist()


def score_model(model, data, metric, cols=None):
    # Create masked array
    if cols is None:
        cols = data.columns

    maskedData = MaskedArray(data=data)
    maskedData.generate()
    maskedDf = pd.DataFrame(
        maskedData.getMaskedMatrix(), index=data.index, columns=data.columns
    )
    # Predict
    # model.fit(maskedDf)
    imputed = model.predict(maskedDf)

    imputedGenes = np.intersect1d(cols, imputed.columns)

    # Compare imputed masked array and input
    maskedIdx = maskedDf[imputedGenes].values != data[imputedGenes].values
    score_res = metric(
        data[imputedGenes].values[maskedIdx], imputed[imputedGenes].values[maskedIdx]
    )
    return score_res

# def wMSE(y_true,y_pred):
#     return tf.reduce_mean(y_true*tf.square(y_true-y_pred))

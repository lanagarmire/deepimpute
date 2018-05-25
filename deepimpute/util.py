import numpy as np
import pandas as pd

from deepimpute.maskedArrays import MaskedArray

""" Preprocessing functions """


def log1x(x):
    return np.log(1 + x)


def exp1x(x):
    return np.exp(x) - 1


def libNorm(x):
    return 1.e6 / np.sum(x + 1)


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


def get_maxes(dataframe, limit):
    genes = dataframe.values
    topValues = genes[:limit].copy()
    topIndices = np.arange(limit)
    currentMinIdxInTopValues = np.argmin(topValues)
    currentMinVal = topValues[currentMinIdxInTopValues]
    for idx, val in enumerate(genes[limit:]):
        if val <= currentMinVal:
            continue
        topIndices[currentMinIdxInTopValues] = idx + limit
        topValues[currentMinIdxInTopValues] = val
        currentMinIdxInTopValues = np.argmin(topValues)
        currentMinVal = topValues[currentMinIdxInTopValues]
    return dataframe.index[topIndices]


def get_input_genes(
    data, dims, distanceMatrix=None, targets=None, predictorLimit=None, seed=1234
):
    if predictorLimit is None:
        predictorLimit = data.shape[1]
    predictorLimit = min(predictorLimit, data.shape[1])
    predictors = data.quantile(.99).sort_values(ascending=False).index[0:predictorLimit]

    if targets is None:
        np.random.seed(seed)
        targets = [np.random.choice(data.columns, dims[1], replace=False)]

    if distanceMatrix is None:
        distanceMatrix = np.abs(
            pd.DataFrame(np.corrcoef(data.T), index=data.columns, columns=data.columns)[
                predictors
            ]
        )
    in_out_genes = []
    for genes in targets:
        predictorGenes = np.unique(
            [get_maxes(distanceMatrix.loc[gene], dims[0]) for gene in genes]
        )
        predictors_noTarget = [gene for gene in predictorGenes if gene not in genes]
        if len(predictors_noTarget) > 0.01 * dims[1]:
            predictorGenes = predictors_noTarget
        in_out_genes.append((predictorGenes, genes))
    return in_out_genes


def get_target_genes(gene_quantiles, NN_lim="auto"):
    if NN_lim == "auto":
        NN_genes = gene_quantiles[gene_quantiles > 5].index
    else:
        if NN_lim is None:
            NN_lim = len(gene_quantiles)
        NN_lim = min(NN_lim, len(gene_quantiles))
        NN_genes = gene_quantiles.sort_values(ascending=False).index[:NN_lim]
    print("Gene prediction limit set to {} genes".format(len(NN_genes)))

    return NN_genes.tolist()


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

import os,binascii

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

# import multiprocessing
from multiprocessing.pool import Pool

from deepimpute.net import Net
from deepimpute.normalizer import Normalizer

def fit_parallel(args):
    NN_parameters,output,X,Y = args
    net = Net(dims=[X.shape[1],Y.shape[1]],
              outputdir=output,
              **NN_parameters,
    )
    net.fit(X,Y)
    print("Training finished. Writing to {}".format(output))

def generate_random_id():
    rd_number = binascii.b2a_hex(os.urandom(3))

    if type(rd_number) is bytes:
        rd_number = rd_number.decode()
    return rd_number

def get_distance_matrix(raw,npred):
    potential_pred = ( (raw.std() / (1+raw.mean()))
                       .sort_values(ascending=False)
                       .index[:npred])
    
    covariance_matrix = pd.DataFrame(np.abs(np.corrcoef(raw.T)),
                                     index=raw.columns,
                                     columns=raw.columns).fillna(0)[potential_pred]
    return covariance_matrix

class MultiNet:

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=64,
                 max_epochs=500,
                 ncores=20,
                 loss="wMSE",
                 normalization="log_or_exp",
                 output_prefix="/tmp/multinet",
                 sub_outputdim=512,
                 seed=1234,
                 architecture=None
    ):
        self.NN_parameters = {"learning_rate": learning_rate,
                              "batch_size": batch_size,
                              "loss": loss,
                              "architecture": architecture,
                              "max_epochs": max_epochs,
                              "seed": seed
                              }
        self.normalization = normalization
        self.sub_outputdim = sub_outputdim
        self.output_prefix = "{}-{}".format(output_prefix,generate_random_id())
        self.outputdirs = []
        self.ncores = ncores
        self.seed = seed

    def initSubNet(self,dims,output):
        net = Net(dims=dims,
                  learning_rate=self.lr,
                  batch_size=self.bs,
                  loss=self.loss,
                  outputdir=output
        )
        return net
        
    def fit(self,
            raw,
            cell_subset=1,
            NN_lim=None,
            genes_to_impute=None,
            npred=5000,
            ntop=5,
            minVMR=0.5,
            mode='random',
    ):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if cell_subset != 1:
            if cell_subset < 1:
                raw = raw.sample(frac=cell_subset)
            else:
                raw = raw.sample(cell_subset)

        gene_metric = (raw.var()/(1+raw.mean())).sort_values(ascending=False)

        if genes_to_impute is None:
            genes_to_impute = self.filter_genes(gene_metric, minVMR, NN_lim=NN_lim)
        else:
            n_genes = len(genes_to_impute)
            if n_genes % self.sub_outputdim != 0:
                print("The number of input genes is not a multiple of {}. Filling with other genes.".format(n_genes))
                fill_genes = gene_metric[:(self.sub_outputdim-n_genes)]
                genes_to_impute = np.concatenate((genes_to_impute,fill_genes))

        covariance_matrix = get_distance_matrix(raw,npred)
        
        self.setTargets(raw.reindex(columns=genes_to_impute), mode=mode)
        self.setPredictors(covariance_matrix,ntop=ntop)

        normalizer = Normalizer.fromName(self.normalization).fit(raw)
        norm_data = normalizer.transform(raw)

        self.outputdirs = ["{}/NN_{}".format(self.output_prefix,i)
                           for i in range(len(self.targets))]

        self.NN_parameters['ncores'] = max(1,int( self.ncores / len(self.targets) ) )

        print("Using {} cores / thread ({} threads)".format(self.NN_parameters['ncores'], len(self.targets)))

        inputs = [(self.NN_parameters,output,
                   norm_data[predictors].values.astype(np.float32),
                   norm_data[targets].values.astype(np.float32))
                  for (output,predictors,targets) in zip(self.outputdirs,
                                                         self.predictors,
                                                         self.targets)]        
        pool = Pool(self.ncores)
        pool.map(fit_parallel,inputs)
        pool.close()
        pool.join()

        return self

    def predict(self,
                raw,
                imputed_only=False,
                policy="restore"):

        normalizer = Normalizer.fromName(self.normalization).fit(raw)
        norm_raw = normalizer.transform(raw)
        
        predicted = []

        for model_dir,predictors,targets in zip(self.outputdirs,self.predictors,self.targets):
            net = Net(dims=[len(predictors),len(targets)],
                      outputdir=model_dir,
                      **self.NN_parameters)
            predicted.append(net.predict(norm_raw.loc[:,predictors].values.astype(np.float32)))

        predicted = pd.DataFrame(np.hstack(predicted),
                                 index=raw.index,
                                 columns=self.targets.flatten())

        predicted = predicted.groupby(by=predicted.columns,axis=1).mean()
        
        not_predicted = norm_raw.drop(self.targets.flatten(),axis=1)

        imputed = pd.concat([predicted,not_predicted],axis=1).loc[raw.index,raw.columns]

        # To prevent overflow
        imputed = imputed.mask( (imputed>2*norm_raw.values.max()) | imputed.isnull(), norm_raw)
        # Convert back to counts
        imputed = normalizer.transform(imputed,rev=True)        
        
        if policy == "restore":
            print("Filling zeros")
            imputed = imputed.mask(raw>0,raw)
        elif policy == "max":
            print("Imputing data with 'max' policy")
            imputed = imputed.mask(raw>imputed,raw)

        if imputed_only:
            return imputed.loc[:,predicted.columns]
        else:
            return imputed
        
    def filter_genes(self,
                    gene_metric, # assumes gene_metric is sorted
                    threshold,
                    NN_lim=None
    ):
        if not str(NN_lim).isdigit():
            NN_lim = (gene_metric > threshold).sum()

        n_subsets = int(np.ceil(NN_lim / self.sub_outputdim))
        genes_to_impute = gene_metric.index[:n_subsets*self.sub_outputdim]

        rest = (self.sub_outputdim*n_subsets) % len(genes_to_impute)

        if rest > 0:
            fill_genes = np.random.choice(gene_metric.index, rest)
            genes_to_impute = np.concatenate([genes_to_impute,fill_genes])

        print("{} genes selected for imputation".format(len(genes_to_impute)))

        return genes_to_impute

    def setTargets(self,data, mode='random'):
        
        n_subsets = int(data.shape[1]/self.sub_outputdim)

        if mode == 'progressive':
            self.targets = data.columns.values.reshape([n_subsets,self.sub_outputdim])
        else:
            self.targets = np.random.choice(data.columns,
                                            [n_subsets,self.sub_outputdim],
                                            replace=False)
        
    def setPredictors(self,covariance_matrix,ntop=5):
        
        self.predictors = []
        for i,targets in enumerate(self.targets):
            subMatrix = ( covariance_matrix
                          .loc[targets]
                          # .drop( np.intersect1d(targets,potential_pred),axis=1 )
                          )
            sorted_idx = np.argsort(-subMatrix.values,axis=1)
            predictors = subMatrix.columns[sorted_idx[:,:ntop]].values.flatten()

            self.predictors.append(np.unique(predictors))

            print("Net {}: {} predictors, {} targets"
                  .format(i,len(np.unique(predictors)),len(targets)))

    def score(self,data,policy=None):
        Y_hat = self.predict(data,policy=policy)
        Y = data.loc[Y_hat.index,Y_hat.columns]

        return pearsonr(Y_hat.values.reshape(-1), Y.values.reshape(-1))
        

if __name__ == '__main__':

    from deepimpute.multinet import MultiNet as MN
    np.random.seed(1234)

    # Prepare data
    raw = pd.read_csv('jurkat_dp0.05_2k-genes.csv',index_col=0).sample(frac=1).T

    # New implementation of DeepImpute
    model = MultiNet()
    print("Starting multinet")
    model.fit(raw)
    targets = np.unique(model.targets.flatten())
    pred = model.predict(raw)[targets].values

    raw = raw.loc[:,targets]
    
    # Deepimpute
    if not os.path.exists('dp_prediction.csv'):
        model_DI = MN(ncores=20,loss='wMSE')
        model_DI.fit(raw,NN_lim=2000)
        pred_DI = model_DI.predict(raw,restore_pos_values=False)
        pred_DI.to_csv('dp_prediction.csv')
    else:
        pred_DI = pd.read_csv('dp_prediction.csv',index_col=0)

    pred_DI = pred_DI.loc[raw.index,raw.columns].values
    
    # Assessment
    
    truth = pd.read_csv('jurkat_filt-95.csv',index_col=0).loc[raw.index,raw.columns]
    mask = truth != raw

    y_true = np.log1p(truth.values.flatten())
    y_hat = np.log1p(pred.flatten())
    y_hat_DI = np.log1p(pred_DI.flatten())

    indices = np.random.choice(len(y_true),100000,replace=False)
    y_true = [y_true[i] for i in indices]
    y_hat = [y_hat[i] for i in indices]
    y_hat_DI = [y_hat_DI[i] for i in indices]

    y_true_masked = np.log1p(truth.values[mask.values])
    y_hat_masked = np.log1p(pred[mask.values])
    y_hat_DI_masked = np.log1p(pred_DI[mask.values])

    lims = [0,7]
    
    fig,ax = plt.subplots(2,2)
    ax[0,0].scatter(y_true,y_hat,s=1)
    ax[0,0].plot(lims,lims,'r-.')
    ax[0,1].scatter(y_true_masked,y_hat_masked,s=1)    
    ax[0,1].plot(lims,lims,'r-.')

    ax[1,0].scatter(y_true,y_hat_DI,s=1)
    ax[1,0].plot(lims,lims,'r-.')
    ax[1,1].scatter(y_true_masked,y_hat_DI_masked,s=1)    
    ax[1,1].plot(lims,lims,'r-.')

    plt.show()




    

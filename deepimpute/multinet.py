import os
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing.pool import Pool

from deepimpute.net import Net
from deepimpute.normalizer import Normalizer

multiprocessing.set_start_method('spawn', force=True)

def fit_parallel(args):
    NN_parameters,output,X,Y = args
    net = Net(dims=[X.shape[1],Y.shape[1]],
              outputdir=output,
              **NN_parameters
    )
    net.fit(X,Y)
    print("Training finished. Writing to {}".format(output))

class MultiNet:

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=64,
                 n_cores=20,
                 loss="wMSE",
                 normalization="log_or_exp",
                 output_prefix="/tmp/multinet",
                 sub_outputdim=512,
                 seed=1234
    ):
        self.NN_parameters = {"learning_rate": learning_rate,
                              "batch_size": batch_size,
                              "loss": loss,
                              }
        self.normalization = normalization
        self.sub_outputdim = sub_outputdim
        self.output_prefix = output_prefix
        self.outputdirs = []
        self.cores = n_cores

        if seed is not None:
            np.random.seed(seed)

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
            npred=1000,
            ntop=20,
            gene_metrics=None
    ):
        
        if cell_subset != 1:
            if cell_subset < 1:
                raw = raw.sample(frac=cell_subset)
            else:
                raw = raw.sample(cell_subset)

        if gene_metrics is None:
            # gene_metrics = raw.quantile(.99) ; threshold = 5 # Gene criteria for filtering
            gene_metrics = raw[raw>0].std()

        threshold = 1
     
        genesToImpute = self.filter_genes(gene_metrics, threshold, NN_lim=NN_lim)
        
        self.setTargets(raw.loc[:,genesToImpute])
        self.setPredictors(raw.loc[:,genesToImpute],npred=npred,ntop=ntop)

        normalizer = Normalizer.fromName(self.normalization).fit(raw)
        norm_data = normalizer.transform(raw)

        self.outputdirs = ["{}/NN_{}".format(self.output_prefix,i)
                           for i in range(len(self.targets))]

        inputs = [(self.NN_parameters,output,norm_data[predictors],norm_data[targets])
                  for (output,predictors,targets) in zip(self.outputdirs,
                                                         self.predictors,
                                                         self.targets)]

        pool = Pool(self.cores)
        pool.map(fit_parallel,inputs)
        pool.close()

        return self

    def predict(self,
                raw,
                imputed_only=False,
                restore_pos_values=False,
                policy="restore"):

        normalizer = Normalizer.fromName(self.normalization).fit(raw)
        norm_raw = normalizer.transform(raw)
        
        predicted = []

        for model_dir,predictors,targets in zip(self.outputdirs,self.predictors,self.targets):
            net = Net(dims=[len(predictors),len(targets)],
                      outputdir=model_dir,
                      **self.NN_parameters)
            predicted.append(net.predict(norm_raw.loc[:,predictors].values))

        predicted = pd.DataFrame(np.hstack(predicted),
                                 index=raw.index,
                                 columns=list(chain(*self.targets)))

        predicted = predicted.groupby(by=predicted.columns,axis=1).mean()
        
        not_predicted = norm_raw.drop(list(chain(*self.targets)),axis=1)

        imputed = pd.concat([predicted,not_predicted],axis=1).loc[raw.index,raw.columns]

        # To prevent overflow
        imputed = imputed.mask( (imputed>norm_raw.values.max()) | imputed.isnull(), norm_raw)
        # Convert back to counts
        imputed = normalizer.transform(imputed,rev=True)        
        
        if policy == "restore" or restore_pos_values:
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
                    gene_metric,
                    threshold,
                    NN_lim=None
    ):
        if NN_lim is None:
            gene_filtered = gene_metric.index[gene_metric > threshold]
        else:
            gene_filtered = gene_metric.sort_values(ascending=False).index[:NN_lim]

        # if len(gene_filtered)>10000:
        #     gene_filtered = gene_filtered[:10000]
        print("{} genes selected for imputation".format(len(gene_filtered)))

        return gene_filtered


    def setTargets(self,data):
        
        n_subsets = int(data.shape[1]/self.sub_outputdim)
        
        self.targets = np.random.choice(data.columns,
                                        [n_subsets,self.sub_outputdim],
                                        replace=False)
        
        leftout_genes = np.setdiff1d(data.columns,self.targets.flatten())

        if len(leftout_genes) > 0:
            fill_genes = np.random.choice(data.columns,
                                          self.sub_outputdim-len(leftout_genes),
                                          replace=False)
            last_batch = np.concatenate([leftout_genes,fill_genes])
            self.targets = np.vstack([self.targets,last_batch])
            # self.targets = self.targets.tolist() + [leftout_genes.tolist()]

    # def _setPredictors(self,data,ntop=20):

    #     mask = (data==0).astype(float).T
    #     dists = 1 / (1+np.dot(mask.values,data.values))
        
    #     dist_matrix = pd.DataFrame(dists,
    #                                index=data.columns,
    #                                columns=data.columns)
    #     self.predictors = []
    #     for i,targets in enumerate(self.targets):
    #         subMatrix = dist_matrix.loc[targets]
    #         sorted_idx = np.argsort(-subMatrix.values,axis=1)
    #         predictors = subMatrix.columns[sorted_idx[:,:ntop]].values.flatten()

    #         print("{} predictors selected for model {}"
    #               .format(len(np.unique(predictors)),i))
    #         self.predictors.append(np.unique(predictors))

    def setPredictors(self,raw,ntop=20,npred=2000):
        potential_pred = ((raw.std() / (1+raw.mean()))
                          .sort_values(ascending=False)
                          .index[:npred]
        )
        covariance_matrix = pd.DataFrame(np.abs(np.corrcoef(raw.T)),
                                         index=raw.columns,
                                         columns=raw.columns).fillna(0)[potential_pred]
        self.predictors = []
        for i,targets in enumerate(self.targets):
            subMatrix = covariance_matrix.loc[targets].drop(
                np.intersect1d(targets,potential_pred),axis=1)
            sorted_idx = np.argsort(-subMatrix.values,axis=1)
            predictors = subMatrix.columns[sorted_idx[:,:ntop]].values.flatten()

            self.predictors.append(np.unique(predictors))

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
        model_DI = MN(n_cores=20,loss='wMSE')
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




    

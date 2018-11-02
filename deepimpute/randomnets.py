from itertools import chain
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Model,model_from_json
from keras.layers import Dense,Dropout,Input,concatenate
from keras.callbacks import EarlyStopping

from deepimpute.util import wMSE,poisson_loss
from deepimpute.normalizer import Normalizer

from deepimpute.multinet import MultiNet

class RandomNets:

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=64,
                 loss=wMSE,
                 normalization="log_or_exp",
                 outputdir="/tmp/test1234"
    ):
        self.architecture = None
        self.lr = learning_rate
        self.bs = batch_size
        self.loss = loss
        self.max_epochs = 300
        self.normalization=normalization
        self.sub_outputdim = 500
        self.outputdir = outputdir

    def loadDefaultArchitecture(self):
        self.architecture = [
            {"type": "dense", "neurons": 256, "activation": "relu"},
            {"type": "dropout", "rate": 0.3} ]

    def save(self,model):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        
        model_json = model.to_json()
                
        with open("{}/model.json".format(self.outputdir), "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        model.save_weights("{}/model.h5".format(self.outputdir))
        print("Saved model to disk")

    def load(self):
        json_file = open('{}/model.json'.format(self.outputdir), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('{}/model.h5'.format(self.outputdir))

        return model
        
    def build(self):

        if self.architecture is None:
            self.loadDefaultArchitecture()
        
        inputs = [Input(shape=(len(genes),))
                  for genes in self.predictors]

        outputs = inputs

        for layer in self.architecture:
            if layer['type'].lower() == 'dense':
                outputs = [Dense(layer['neurons'],activation=layer['activation'])(xi)
                           for xi in outputs]
            elif layer['type'].lower() == 'dropout':
                outputs = [Dropout(layer['rate'])(xi)
                           for xi in outputs]
            else:
                print("Unknown layer type.")

        outputs = [Dense(self.sub_outputdim,activation="softplus")(xi)
                   for xi in outputs ]
        
        global_output = concatenate(outputs)

        # n_out = len(np.unique(self.targets.flatten()))
        
        # global_output = Dense(n_out,activation='softplus')(global_output)
        # global_output = Dropout(.3)(global_output)        
    
        model = Model(inputs=inputs,
                      outputs=global_output)

        try:
            loss = eval(self.loss)
        except:
            loss = self.loss
    
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=loss)
        print(model.summary())
        
        return model

    
    def fit(self,raw):
        
        self.setTargets(raw)
        self.setPredictors(raw)

        # Filter out low-quality cells
        data = raw[self.targets.flatten()]

        filt = data.index[(data==0).sum(axis=1)>0.05*data.shape[1]]
        data = data.loc[filt]
        
        # Normalize data
        normalizer = Normalizer.fromName(self.normalization)
        data = normalizer.fit(data).transform(data)

        # Build network
        model = self.build()
        
        # Train / Test split
        test_samples = data.sample(frac=0.05).index
        train_samples = data.drop(test_samples).index

        X_train = [data.loc[train_samples,genes].values
                   for genes in self.predictors]
        Y_train = [data.loc[train_samples,genes].values
                   for genes in self.targets]
        
        X_test = [data.loc[test_samples,genes].values
                  for genes in self.predictors]
        Y_test = [data.loc[test_samples,genes].values
                  for genes in self.targets]
        # Fitting
        model.fit(X_train,
                  np.hstack(Y_train), # Y_train
                  epochs=self.max_epochs,
                  batch_size=self.bs,
                  callbacks=[EarlyStopping(monitor='val_loss',patience=5)],
                  validation_data=(X_test,np.hstack(Y_test)))

        self.save(model)

        return self

    def predict(self,raw):

        model = self.load()
        
        normalizer = Normalizer.fromName(self.normalization)
        data = normalizer.fit(raw).transform(raw)

        X_in = [data.loc[:,genes].values
                for genes in self.predictors]

        predicted = model.predict(X_in)

        # Aggregate data
        predicted = pd.DataFrame(predicted,# np.hstack(predicted),
                                 columns=list(chain(*self.targets)),
                                 index=data.index)

        return normalizer.transform(predicted[data.columns],rev=True)
        

    def setTargets(self,data):
        n_subsets = int(data.shape[1]/self.sub_outputdim)
        
        self.targets = np.random.choice(data.columns,
                                        [n_subsets,self.sub_outputdim],
                                        replace=False)
        if n_subsets*self.sub_outputdim < data.shape[1]:
            leftout_genes = np.setdiff1d(data.columns,self.targets.flatten())
            fill_genes = np.random.choice(self.targets.flatten(),
                                          data.shape[1]-n_subsets*self.sub_outputdim,
                                          replace=False)
            self.targets = np.vstack([self.targets,
                                      leftout_genes.reshape(1,-1),
                                      fill_genes.reshape(1,-1)])
            
    def setPredictors(self,data,ntop=20):
        potential_predictors = (data.quantile(.99)>10).index

        covariance_matrix = pd.DataFrame(np.corrcoef(data.T),
                                         index=data.columns,
                                         columns=data.columns)[potential_predictors]

        self.predictors = []
        for targets in self.targets:
            subMatrix = covariance_matrix.loc[targets]
            sorted_idx = np.argsort(-subMatrix.values,axis=1)
            predictors = subMatrix.columns[sorted_idx[:,:ntop]]
            self.predictors.append(np.unique(predictors))

if __name__ == '__main__':

    np.random.seed(1234)

    # Prepare data
    raw = pd.read_csv('jurkat_dp0.05_2k-genes.csv',index_col=0).sample(frac=1).T

    # Deepimpute
    if not os.path.exists('dp_prediction.csv'):
        model_DI = MultiNet(n_cores=20,loss='wMSE')
        model_DI.fit(raw,NN_lim=2000)
        pred_DI = model_DI.predict(raw,restore_pos_values=False)
        pred_DI.to_csv('dp_prediction.csv')
    else:
        pred_DI = pd.read_csv('dp_prediction.csv',index_col=0)

    pred_DI = pred_DI.loc[raw.index,raw.columns].values

    # New implementation of DeepImpute
    model = RandomNets()
    model.fit(raw)
    pred = model.predict(raw).values
    
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




    

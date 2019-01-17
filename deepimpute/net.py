import os

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import numpy as np

import keras
from keras import backend as K
from keras.models import Model,model_from_json
from keras.layers import Dense,Dropout,Input
from keras.callbacks import EarlyStopping
import keras.losses

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def wMSE(y_true,y_pred):
    # weights = tf.cast(y_true>0,tf.float32)
    weights = y_true
    return tf.reduce_mean(weights*tf.square(y_true-y_pred))

class Net:
    def __init__(self,
                 dims=None,
                 learning_rate=1e-4,
                 batch_size=64,
                 loss="wMSE",
                 architecture=None,
                 ncores=3,
                 max_epochs=500,
                 outputdir="/tmp/test1234",
                 seed=1234
    ):
        self.architecture = architecture
        self.inputdim, self.outputdim = dims
        self.lr = learning_rate
        self.bs = batch_size
        self.loss = loss
        self.max_epochs = max_epochs
        self.outputdir = outputdir
        self.ncores = ncores
        self.seed=seed

    def loadDefaultArchitecture(self):
        if self.architecture is None:
            self.architecture = [
                {"type": "dense", "neurons": 256, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
            ]

    def save(self,model):
        os.system("mkdir -p {}".format(self.outputdir))
        
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
        
        inputs = Input(shape=(self.inputdim,))
        outputs = inputs

        for layer in self.architecture:
            
            if layer['type'].lower() == 'dense':
                outputs = Dense(layer['neurons'],activation=layer['activation'])(outputs)

            elif layer['type'].lower() == 'dropout':
                outputs = Dropout(layer['rate'], seed=self.seed)(outputs)
                
            else:
                print("Unknown layer type.")

        outputs = Dense(self.outputdim,activation="softplus")(outputs)
                
        model = Model(inputs=inputs,outputs=outputs)

        try:
            loss = eval(self.loss)
        except:
            loss = getattr(keras.losses,self.loss)
    
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),loss=loss)
        
        return model
    
    def fit(self,X,Y,verbose=0):

        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        
        config = tf.ConfigProto(intra_op_parallelism_threads=self.ncores,
                                inter_op_parallelism_threads=self.ncores,
                                allow_soft_placement=True, device_count = {'CPU': self.ncores})
        session = tf.Session(config=config)
        K.set_session(session)

        # cell_filt = Y.index 

        # Build network
        model = self.build()

        # Train / Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.05)

        # Fitting
        model.fit(X_train,Y_train,
                  validation_data=(X_test,Y_test),
                  epochs=self.max_epochs,
                  batch_size=self.bs,
                  callbacks=[EarlyStopping(monitor='val_loss',patience=5)],
                  verbose=verbose
        )
        self.save(model)

        return self

    def predict(self, X):

        model = self.load()

        return model.predict(X)

    def score(self,X,Y):

        model = self.load()
        Y_hat = model.predict(X)

        return pearsonr(Y_hat.reshape(-1), Y.values.reshape(-1))

    
if __name__ == '__main__':

    net = Net()
    print("Need to implement a test...")

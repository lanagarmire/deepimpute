import os

from sklearn.model_selection import train_test_split

import keras
from keras.models import Model,model_from_json
from keras.layers import Dense,Dropout,Input,GaussianNoise
from keras.callbacks import EarlyStopping
import keras.losses

import tensorflow as tf

def wMSE(y_true,y_pred):
    weights = tf.cast(y_true>0,tf.float32)
    weights = y_true
    return tf.reduce_mean(weights*tf.square(y_true-y_pred))

class Net:
    def __init__(self,
                 dims=None,
                 learning_rate=1e-4,
                 batch_size=64,
                 loss="wMSE",
                 outputdir="/tmp/test1234"
    ):
        self.architecture = None
        self.inputdim, self.outputdim = dims
        self.lr = learning_rate
        self.bs = batch_size
        self.loss = loss
        self.max_epochs = 300
        self.outputdir = outputdir

    def loadDefaultArchitecture(self):
        self.architecture = [
            {"type": "dense", "neurons": 256, "activation": "relu"},
            {"type": "dropout", "rate": 0.3},
        ]

        # self.architecture = [
        #     {"type": "dense", "neurons": 128, "activation": "relu"},
        #     {"type": "dropout", "rate": 0.3},
        #     {"type": "dense", "neurons": self.outputdim, "activation": "relu"},
        #     {"type": "dense", "neurons": 128, "activation": "relu"},
        #     {"type": "dense", "neurons": self.outputdim, "activation": "sigmoid"}
        #     ]

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

    def _build(self):
        
        inputs = Input(shape=(self.inputdim,))

        z1 = Dropout(0.2)(Dense(128,activation='relu')(inputs))
        y = Dense(self.outputdim,activation='softplus')(z1)
        z2 = Dense(128,activation='relu')(y)
        pi = Dense(self.outputdim,activation='sigmoid')(z2)

        outputs = y

        for layer in self.architecture:
            
            if layer['type'].lower() == 'dense':
                outputs = Dense(layer['neurons'],activation=layer['activation'])(outputs)
            elif layer['type'].lower() == 'dropout':
                outputs = Dropout(layer['rate'])(outputs)
            elif layer['type'].lower() == 'noise':
                outputs = GaussianNoise(layer['stddev'])(outputs)
            else:
                print("Unknown layer type.")

        
        model = Model(inputs=inputs,outputs=[y,pi])

        if callable(self.loss):
            loss = self.loss
        else:
            try:
                loss = eval(self.loss)
            except:
                loss = getattr(keras.losses,self.loss)
    
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),loss=loss)
        
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
                outputs = Dropout(layer['rate'])(outputs)
                
            elif layer['type'].lower() == 'noise':
                outputs = GaussianNoise(layer['stddev'])(outputs)
                
            else:
                print("Unknown layer type.")

        outputs = Dense(self.outputdim,activation="softplus")(outputs)
                
        model = Model(inputs=inputs,outputs=outputs)

        if callable(self.loss):
            loss = self.loss
        else:
            try:
                loss = eval(self.loss)
            except:
                loss = getattr(keras.losses,self.loss)
    
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),loss=loss)
        
        return model
    
    def fit(self,X,Y):

        cell_filt = Y.index[ (Y>0).sum(axis=1) > 0.01*Y.shape[0] ]

        if len(cell_filt) < X.shape[0]:
            print("Using {}/{} cells for training".format(len(cell_filt),X.shape[0]))
        
        # Build network
        model = self.build()
        
        # Train / Test split
        X_train, X_test, Y_train, Y_test = train_test_split(X.loc[cell_filt],Y.loc[cell_filt],test_size=0.05)
        
        # Fitting
        model.fit(X_train,Y_train,
                  validation_data=(X_test,Y_test),
                  epochs=self.max_epochs,
                  batch_size=self.bs,
                  callbacks=[EarlyStopping(monitor='val_loss',patience=10)],
                  verbose=0
        )
        self.save(model)

        return self

    def predict(self, X):

        model = self.load()        
        prediction = model.predict(X)

        return prediction

    
if __name__ == '__main__':

    net = Net()
    print("Need to implement a test...")

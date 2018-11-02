import numpy as np
import tensorflow as tf

""" Preprocessing functions """

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

""" Custom loss functions """

def wMSE(y_true,y_pred):
    return tf.reduce_mean(y_true*tf.square(y_true-y_pred))

def poisson_loss(y_true,y_pred):
    # mask = tf.cast(y_true>0,tf.float32)
    y_true = y_true + 0.0001
    NLL = tf.lgamma(y_pred+1)-y_pred*tf.log(y_true)+y_true
    return tf.reduce_mean(NLL)


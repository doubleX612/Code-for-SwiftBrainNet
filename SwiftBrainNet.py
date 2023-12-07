import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import math
import h5py
import random
import numpy as np
import scipy.io as scio
from keras import metrics
from keras import regularizers
import os
import tensorflow as tf
from scipy import signal
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True 
import dill
from keras import backend as K


# Data Augmentation
def gen_data(xin,yout,rate): # data with the same label
    
    bigsize = int(xin.shape[0]*rate)
    new_data = np.zeros((bigsize,xin.shape[1]))
    new_label =yout[:,0:bigsize]
    lambd = 0.75
    for i in range(bigsize):
        data_sel1 = xin[random.randint(0,xin.shape[0]-1),:]
        data_sel2 = xin[random.randint(0,xin.shape[0]-1),:]
        new_data[i,:] = lambd*data_sel1+(1-lambd)*data_sel2

    return new_data.T,new_label

# Build SwiftBrainNet Model
def SwiftBrainNet(rate,hlayer_num,relayer_num,nb_input):
    ee_in =  Input(shape=(nb_input,),name='ee_in')
    ee_hid = Dense(units=int(nb_input*(1+rate)),kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='relu')(ee_in)
    ee_hid = Dropout(0.3)(ee_hid)
    
    nb_hidden = int(nb_input*(0.9))
    for i in range(hlayer_num):  # Main Classifier
        if(i!=0): nb_hidden = nb_hidden*rate
        ee_hid = Dense(units=int(nb_hidden),kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='relu')(ee_hid)
        if(i==1):
            nb_decode = nb_hidden/rate/2
            ee_decode = Dense(units=int(nb_decode),kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='relu',name='de0'+str(i))(ee_hid)
            for j in range(relayer_num-1,1,-1): # Auxiliary Decoder
                nb_decode =  nb_decode/rate/2
                ee_decode = Dense(units=int(nb_decode),kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='relu',name='de'+str(j))(ee_decode)
            ee_rebuildout = Dense(units=nb_input,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='tanh',name='ee_rebuildout')(ee_decode)

    ee_classout = Dense(2, activation='softmax',name='ee_classout')(ee_hid)
    model=Model(inputs=ee_in,outputs=[ee_rebuildout,ee_classout])
    return model   


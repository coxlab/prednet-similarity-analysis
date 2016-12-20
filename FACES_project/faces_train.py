'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

import hickle as hkl

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from faces_settings import *


save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_faces_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_faces_model.json')

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
nt = 10 #make prediction after 10 channels
input_shape = (1, 64, 64) #(numChannels, ypixels, xpixels)
stack_sizes = (input_shape[0], 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(input=inputs, output=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')


lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))


X_train=hkl.load(train_file)
X_val = hkl.load(val_file)
Y_train = np.zeros((X_train.shape[0],1))
Y_val = np.zeros((X_val.shape[0],1))

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=callbacks,validation_data=(X_val, Y_val))

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

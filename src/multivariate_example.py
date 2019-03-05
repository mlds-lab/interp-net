import pickle
import random
import keras
import argparse
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auc_score
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, Masking, GRU, Dropout, Lambda, Permute
from keras.models import load_model, Model, Sequential
from interpolation_layer import single_channel_interp, cross_channel_interp

np.random.seed(10)
tf.set_random_seed(10)

# Loading dataset
"""
y : (N,) discrete for classification, real values for regression
x : (N, D, tn) input multivariate time series data with dimension 
    where N is number of data cases, D is the dimension of 
    sparse and irregularly sampled time series and tn is the union
    of observed time stamps in all the dimension for a data case n.
    Since each tn is of variable length, we pad them with zeros to 
    have an array representation. 
m : (N, D, tn) where m[i,j,k] = 0 means that x[i,j,k] is not observed.
T : (N, D, tn) represents the actual time stamps of observation; 
"""

"""To implement the autoencoder component of the loss, we introduce a set 
of masking variables mr (and mr1) for each data point. If drop_mask = 0, then we remove 
the data point as an input to the interpolation network, and include 
the predicted value at this time point when assessing
the autoencoder loss. In practice, we randomly select 20% of the 
observed data points to hold out from
every input time series."""


def drop_mask(mask, perc=0.2):
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask


x = np.concatenate((x, m, T, drop_mask(m)), axis=1)  # input format
print(x.shape, y.shape)

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=4,
                help="# of GPUs to use for training")
ap.add_argument("-batch", "--batch_size", type=int, default=256,
                help="# batch size to use for training")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs for training")
ap.add_argument("-ref", "--reference_points", type=int,
                default=192, help="# of reference points")
ap.add_argument("-units", "--hidden_units", type=int,
                default=100, help="# of hidden units")
ap.add_argument("-hfadm", "--hours_from_adm", type=int,
                default=48, help="Hours of record to look at")

args = vars(ap.parse_args())
gpu_num = args["gpus"]
iter = args["epochs"]
hid = args["hidden_units"]
timestamp = x.shape[2]
num_features = x.shape[1]/4
ref_points = args["reference_points"]
hours_look_ahead = args["hours_from_adm"]
if gpu_num > 0:
    batch = args["batch_size"]*gpu_num
else:
    batch = args["batch_size"]

# Autoencoder loss


def customloss(ytrue, ypred):
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
                   3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
    wc.shape = (1, num_features)
    y = ytrue[:, :num_features, :]
    m2 = ytrue[:, 3*num_features:4*num_features, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_features:2*num_features, :]
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    x = x/(wc**2)  # dividing by standard deviation
    x = tf.reduce_sum(x, axis=1)/num_features
    return tf.reduce_mean(x)


seed = 0
results = {}
results['loss'] = []
results['auc'] = []
results['acc'] = []
results['auprc'] = []

# interpolation-prediction network


def interp_net():
    if gpu_num > 1:
        dev = "/cpu:0"
    else:
        dev = "/gpu:0"
    with tf.device(dev):
        main_input = Input(shape=(4*num_features, timestamp), name='input')
        sci = single_channel_interp(ref_points, hours_look_ahead)
        cci = cross_channel_interp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True),
                      reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh', recurrent_dropout=0.2, dropout=0.2)(z)
        main_output = Dense(1, activation='sigmoid', name='main_output')(z)
        orig_model = Model([main_input], [main_output, aux_output])
    if gpu_num > 1:
        model = multi_gpu_model(orig_model, gpus=gpu_num)
    else:
        model = orig_model
    print(orig_model.summary())
    return model


earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0000, patience=20, verbose=0)
callbacks_list = [earlystop]

# 5-fold cross-validation

i = 0
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(np.zeros(len(y)), y):
    print("Running Fold:", i+1)
    model = interp_net()  # re-initializing every time
    model.compile(optimizer='adam', loss={'main_output': 'binary_crossentropy', 'aux_output': customloss},
                  loss_weights={'main_output': 1., 'aux_output': 1.}, metrics={'main_output': 'accuracy'})
    model.fit({'input': x[train]}, {'main_output': y[train], 'aux_output': x[train]},
              batch_size=batch, callbacks=callbacks_list, nb_epoch=iter, validation_split=0.20, verbose=2)
    y_pred = model.predict(x[test], batch_size=batch)
    y_pred = y_pred[0]
    total_loss, score, reconst_loss,  acc = model.evaluate(
        {'input': x[test]}, {'main_output': y[test], 'aux_output': x[test]}, batch_size=batch, verbose=0)
    results['loss'].append(score)
    results['acc'].append(acc)
    results['auc'].append(auc_score(y[test], y_pred))
    results['auprc'].append(auprc(y[test], y_pred))
    print(results)
    i += 1

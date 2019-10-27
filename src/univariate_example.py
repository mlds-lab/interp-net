import argparse
import pickle
import numpy as np
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU, Lambda, Permute
from sklearn.preprocessing import MultiLabelBinarizer
from interpolation_layer import single_channel_interp
import warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-batch", "--batch_size", type=int, default=256,
                help="# batch size to use for training")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="# of epochs for training")
ap.add_argument("-units", "--hidden_units", type=int,
                default=128, help="# of hidden units")
ap.add_argument("-ref", "--ref_points", type=int,
                default=128, help="# of refpoints")
args = vars(ap.parse_args())
batch = args["batch_size"]
epoch = args["epochs"]
hid = args["hidden_units"]
ref_points = args["ref_points"]
hours_look_ahead = 100  # same as the input time stamps range
timestamp = 94
num_features = 1


# Loading Dataset
with open('Dataset/UWaveGestureLibraryAll-10.pkl', 'rb') as f:
    x_train, y_train, x_test, y_test, l_train, l_test = pickle.load(f, encoding='latin1')

x_train = np.array(x_train)
l_train = np.array(l_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
l_test = np.array(l_test)
mlb = MultiLabelBinarizer()

l_train = [(int(val),) for val in l_train]
l_train = mlb.fit_transform(l_train)
l_test = [(int(val),) for val in l_test]
l_test = mlb.fit_transform(l_test)
print(x_train.shape, y_train.shape, x_test.shape,
      y_test.shape, l_train.shape, l_test.shape)

# To implement the autoencoder component of the loss, we introduce a set 
# of masking variables mr (and mr1) for each data point. If mr = 0, then we remove 
# the data point as an input to the interpolation network, and include 
# the predicted value at this time point when assessing
# the autoencoder loss. In practice, we randomly select 20% of the 
# observed data points to hold out from
# every input time series.

m = np.ones_like(x_train) # for one dimensional time series m is all ones
mr = np.ones_like(x_train)
for i in range(m.shape[0]):
    r = np.random.choice(m.shape[1], int(0.2*m.shape[1]), replace=False)
    mr[i, r] = 0

x_train = np.stack([y_train, m, 100*x_train, mr], axis=1)
m1 = np.ones_like(x_test)
mr1 = np.ones_like(x_test)
for i in range(m.shape[0]):
    r = np.random.choice(m1.shape[1], int(0.2*m1.shape[1]), replace=False)
    mr1[i, r] = 0

x_test = np.stack([y_test, m1, 100*x_test, mr1], axis=1)
print(x_train.shape, l_train.shape)
print(x_test.shape, l_test.shape)

# Autoencoder Loss
def customloss(ytrue, ypred):
    wc = 1
    y = ytrue[:, :num_features, :]
    m2 = ytrue[:, 3*num_features:4*num_features, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_features:2*num_features, :]
    m = m1 * m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred) * (y - ypred)
    x = x * m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    x = x/(wc**2)  # divide by std in case of multivariate time-series
    x = tf.reduce_sum(x, axis=1)/num_features
    return tf.reduce_mean(x)


# Interpolation-Prediction Model
main_input = Input(shape=(4*num_features, timestamp), name='input')
sci = single_channel_interp(
    ref_points, hours_look_ahead, weights=[np.array([-3.0])])
interp = sci(main_input)
reconst = sci(main_input, reconstruction=True)
aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
z = Permute((2, 1))(interp)
z = GRU(hid, activation='tanh')(z)
main_output = Dense(8, activation='softmax', name='main_output')(z)
model = Model([main_input], [main_output, aux_output])
model.compile(optimizer='adam', loss={'main_output': 'categorical_crossentropy', 'aux_output': customloss},
              loss_weights={'main_output': 1., 'aux_output': 1.}, metrics={'main_output': 'accuracy'})
print(model.summary())
earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.000, patience=30, verbose=0)
callbacks_list = [earlystop]

print('Train...')
model.fit({'input': x_test}, {'main_output': l_test, 'aux_output': x_test}, batch_size=batch,
          callbacks=callbacks_list, epochs=epoch, validation_split=0.3, shuffle=True)
print(model.evaluate(x_train, [l_train, x_train], batch_size=batch))

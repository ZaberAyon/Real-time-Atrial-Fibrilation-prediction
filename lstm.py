from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks


traindata = pd.read_csv('test_set.csv', header=None,index_col=0)
testdata = pd.read_csv('train_set.csv', header=None, index_col=0)

traindata.fillna(method='ffill')
testdata.fillna(method = 'ffill')

traindata.fillna(-99999, inplace=True)
testdata.fillna(-99999, inplace = True)




X = traindata.iloc[:,1:15001]
Y = traindata.iloc[:,-1]
C = testdata.iloc[:,-1]
T = testdata.iloc[:,1:15001]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 4

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=14999))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="F:\Projects\Hackathon\log_file\checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=50, callbacks=[checkpointer])
model.save("F:\Projects\Hackathon\log_file\lstm1layer_model.hdf5")

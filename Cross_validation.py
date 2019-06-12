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
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

traindata = pd.read_csv('test_set.csv', header=None,index_col=0)
testdata = pd.read_csv('train_set.csv', header=None, index_col=0)



traindata.fillna(method='ffill')
testdata.fillna(method = 'ffill')

traindata.fillna(0, inplace=True)
testdata.fillna(0, inplace = True)

print(traindata)
print(testdata)


label_encoder = LabelEncoder()

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

y_train = label_encoder.fit_transform(y_train)
y_test =label_encoder.fit_transform(y_test)
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


batch_size = 4

# 1. define the network
def create_model():
  model = Sequential()
  model.add(LSTM(32,input_dim=14999))  # try using a GRU instead, for fun
  model.add(Dropout(0.1))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

seed = 7
np.random.seed(seed)


model = KerasClassifier(build_fn=create_model, nb_epoch=1000, batch_size=4, verbose=0)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

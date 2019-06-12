from __future__ import print_function
from sklearn.model_selection import train_test_split
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

import serial
import csv
import re
#import matplotlib.pyplot as plt

portPath = "COM4"       # Must match value shown on Arduino IDE
baud = 9600                     # Must match Arduino baud rate
timeout = 5                       # Seconds
filename = "tttt.csv"
max_num_readings = 15000
num_signals = 1

def create_serial_obj(portPath, baud_rate, tout):
    """
    Given the port path, baud rate, and timeout value, creates
    and returns a pyserial object.
    """
    return serial.Serial(portPath, baud_rate, timeout = tout)
    
def read_serial_data(serial):
    """
    Given a pyserial object (serial). Outputs a list of lines read in
    from the serial port
    """
    serial.flushInput()
    
    serial_data = []
    readings_left = True
    timeout_reached = False
    
    while readings_left and not timeout_reached:
        serial_line = serial.readline()
        if serial_line == '':
            timeout_reached = True
        else:
            serial_data.append(serial_line)
            if len(serial_data) == max_num_readings:
                readings_left = False
        
    return serial_data
 
def is_number(string):
    """
    Given a string returns True if the string represents a number.
    Returns False otherwise.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False



 
def clean_serial_data(data):
    """
    Given a list of serial lines (data). Removes all characters.
    Returns the cleaned list of lists of digits.
    Given something like: ['0.5000,33\r\n', '1.0000,283\r\n']
    Returns: [[0.5,33.0], [1.0,283.0]]
    """
    clean_data = []
    line_data = []
    for line in data:
        print (line)
        #line = float(line)
        clean_data.append(int(line)/1000)
        
    return clean_data

print ("Creating serial object...")
serial_obj = create_serial_obj(portPath, baud, timeout)

 
print ("Reading serial data...")
serial_data = read_serial_data(serial_obj)
print (len(serial_data))
print ("Cleaning data...")
clean_data =  clean_serial_data(serial_data)
print (clean_data)


df = pd.DataFrame(np.array(clean_data).reshape(-1,len(clean_data)))

print (df)


testdata = df
#testdata = pd.read_csv('tttt.csv', header=None)
#traindata.fillna(method='ffill')
testdata.fillna(method = 'ffill')

#traindata.fillna(-99999, inplace=True)
testdata.fillna(-99999, inplace = True)


C = testdata.iloc[:,-1]
print(C)
T = testdata.iloc[:,1:15000]

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_test = np.array(C)
#print(y_test)

X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))

#print(X_test)

batch_size = 2

# 1. define the network
model = Sequential()
model.add(LSTM(32,input_dim=14999))  
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.metrics import confusion_matrix

model.load_weights("F:\Projects\Hackathon\log_file\lstm1layer_model.hdf5")
y_pred = model.predict_classes(X_test)
print(y_pred)
print(confusion_matrix(y_test, y_pred))
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
y_prob = model.predict_proba(X_test)
np.savetxt("gru.txt", y_prob)


import os
for file in os.listdir("F:\Projects\Hackathon\log_file\\"):
  model.load_weights("F:\Projects\Hackathon\log_file\\"+file)
  y_pred = model.predict_classes(X_test)

  print("Prediction")

  

  print (y_pred)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  loss, accuracy = model.evaluate(X_test, y_test)
  print(file)
  print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
  print("---------------------------------------------------------------------------------")
  accuracy = accuracy_score(y_test, y_pred.round(),normalize=False)
  recall = recall_score(y_test, y_pred , average="binary")
  precision = precision_score(y_test, y_pred , average="binary")
  f1 = f1_score(y_test, y_pred, average="binary")

  print("accuracy")
  print("%.3f" %accuracy)
  print("precision")
  print("%.3f" %precision)
  print("recall")
  print("%.3f" %recall)
  print("f1score")
  print("%.3f" %f1)

  if(y_pred==[[0]]):
    print("Normal Synas Rythm")
  if(y_pred==[[1]]):
    print("Atrial Fibrilation")
  

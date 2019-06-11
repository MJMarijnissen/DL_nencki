# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:43:49 2019

@author: Kubus
"""

import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(60000, 28*28)

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10) #encoding hot-shot
y_test = keras.utils.to_categorical(y_train, 10) #encoding hot-shot

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(28*28,)))
model.add(Dropout(0.2)) #20% szansa na wylaczenie losowego neurona
model.add(Dense(128, activation="relu")
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='sqd', metrics=['accuracy'])
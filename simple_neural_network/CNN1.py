# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:49:33 2019

@author: Kubus
"""

import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.reshape[0], 28, 28, 1) # 1 bechause grayscale
x_test = x_test.reshape(x_test.reshape[0], 28, 28, 1) # 1 bechause grayscale

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1))) #padding does not delete the pixels after convolution (retians size of image)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
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

x_train.reshape(60000, 28*28)
x_test.reshape(60000, 28*28)

x_train /= 255
x_test /= 255

keras.utils.to_categorical(y_train, 10)
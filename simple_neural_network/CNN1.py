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

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

x_train /= 255
x_test /= 255

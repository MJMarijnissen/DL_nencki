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

x_train.reshape(60000)
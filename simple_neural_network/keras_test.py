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
x_test = x_test.reshape(10000, 28*28)
x_test = x_test.astype('float32')
x_train = x_train.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10) #encoding hot-shot
y_test = keras.utils.to_categorical(y_train, 10) #encoding hot-shot

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dropout(0.2)) #20% szansa na wylaczenie losowego neurona
model.add(Dense(128, activation="relu")
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', 
              optimizer='sqd', 
              metrics=['accuracy']
              )

model.fit(x_train, y_train, 
          batch_size=32, 
          epochs = 128, 
          verbose=True, # verbose = get info back
          validation_data = (x_test,y_test)
          ) 

score = model.evaluate(x_test, y_test, 
                       verbose = False
                       )

print('Test loss', score[0])
print('Test accuracy', score[1])
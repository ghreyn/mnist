#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:43:48 2017

@author: reynold
"""

import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_diagram import ascii

VAL_SIZE = 0.20
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
SEED=100
logfile ="kresults.txt"

def create_model():
    """
    """
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
     
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def generate_submission(predictions):
    """
    """
    submission = pd.DataFrame({
        "ImageId": np.arange(1,28001),
        "Label": predictions.astype(int)
    })
  
    print("output dataframe: {}".format(submission.shape))

    now = datetime.datetime.now()

    csvfile = "mnist_keras.csv" 
    submission.to_csv(csvfile, index=False)
    
    with open(logfile, "a") as myfile:
        print("{0} Wrote output file: {1} {2}".format(now, csvfile, submission.shape), file=myfile)           


def main():
    """
    """
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train, X_val, y_train, y_val = train_test_split(train.drop(['label'],axis=1),train['label'],test_size=VAL_SIZE,stratify=train['label'], random_state=SEED)
    print (X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    train_dataset = X_train.values
    train_labels = y_train

    valid_dataset = X_val.values
    valid_labels = y_val

    train_dataset = train_dataset.reshape(X_train.shape[0], 28, 28, 1)
    valid_dataset = valid_dataset.reshape(X_val.shape[0], 28, 28, 1)

    train_dataset = train_dataset.astype('float32')
    train_dataset/= 255
    valid_dataset = valid_dataset.astype('float32')
    valid_dataset /= 255

    train_labels = np_utils.to_categorical(train_labels, 10)
    valid_labels = np_utils.to_categorical(valid_labels, 10)

    run_model = create_model()
    run_model.summary()

    start_time = datetime.datetime.now()

    run_model.fit(train_dataset, train_labels,
         batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, verbose=1)

    names = run_model.metrics_names
    score = run_model.evaluate(valid_dataset, valid_labels, verbose=0)
    print (names, score)
    
    with open(logfile, "a") as myfile:
        now = datetime.datetime.now()   
        results = zip(names, score)          
        summary = ascii(run_model)
        print("\n{0} START\n{1} END".format(start_time, now), file=myfile)
        print("learning rate: {} batch size: {} epochs: {} optimizer: adam".format(LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS), file=myfile)
        for name, score in results:
            print("{0}: {1:.5f}".format(name,score), file=myfile)
        print(summary, file=myfile)        
        
    test_dataset = test.values
    test_dataset = test_dataset.reshape(test.shape[0], 28, 28, 1)
    test_dataset = test_dataset.astype('float32')
    test_dataset /= 255
    
    predicted_labels = run_model.predict_classes(test_dataset)
    generate_submission(predicted_labels)
    
    
if __name__ == '__main__':
    main()
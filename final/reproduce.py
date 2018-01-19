# -*- coding: utf-8 -*-
import os
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate, PReLU, LeakyReLU, RepeatVector, Masking, add, multiply
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Dot, Permute
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
from keras import optimizers
from keras import initializers
from keras import regularizers, constraints
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer, InputSpec
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from math import log, floor
from sklearn.metrics import log_loss
import codecs
import pandas as pd
import numpy as np
import pickle
import random
import sys
import io
import re

train_data = sys.argv[1]
test_data = sys.argv[2]
train_caption = sys.argv[3]
test_csv = sys.argv[4]
model_path = sys.argv[5]
predict_path = sys.argv[6]


# Read audio file
with open(train_data, "rb") as fp:
    X = pickle.load(fp, encoding='latin1')
with open(test_data, "rb") as fp:
    X_test = pickle.load(fp, encoding='latin1')

# Steps for normalizing mfcc values to z score
X_temp = np.concatenate(X, axis=0)
X_test_temp = np.concatenate(X_test, axis=0)
X_mfcc = np.concatenate([X_temp, X_test_temp], axis=0)
mean = np.mean(X_mfcc, axis=0)
std = np.std(X_mfcc, axis=0)

for i, sample in enumerate(X):
    X[i] = (sample - mean) / std

for i, sample in enumerate(X_test):
    X_test[i] = (sample - mean) / std

# Pad audio sequences into equal length by adding zero rows for input shorter than maxlen
X1_maxlen = max([len(sentence) for sentence in X])
X1 = pad_sequences(X, maxlen=X1_maxlen, dtype='float32')
X1_test = pad_sequences(X_test, maxlen=X1_maxlen, dtype='float32')
X1 = np.concatenate((X1, X1), axis=0)
X1_test = np.repeat(X1_test, 4, axis=0)

# Read translated chinese file and store the chinese sequences for training
X2 = []
X2_test = []
data = []

with codecs.open(train_caption, encoding='utf-8') as f:
    for line in f.readlines():
        X2.append(line)
        data.append(line.split())

with codecs.open(test_csv, encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split(',')
        X2_test.extend(line)

# Tokenize Chinese words into unique ids
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X2)
train_sequences = tokenizer.texts_to_sequences(X2)
test_sequences = tokenizer.texts_to_sequences(X2_test)

word_index = tokenizer.word_index
vocab = len(word_index) + 1
print('Found %s unique tokens.' % len(word_index))

# Pad Chinese sequences into equal length by adding zero ids for input shorter than maxlen
X2_maxlen = max([len(sentence) for sentence in train_sequences])
X2 = pad_sequences(train_sequences, maxlen=X2_maxlen)
X2_test = pad_sequences(test_sequences, maxlen=X2_maxlen)

model = load_model(model_path)

# Predict the option with the value most closest to 1
predict = model.predict([X1_test, X2_test])
predict = predict.reshape(-1, 4)
predict = np.argmax(predict, axis=1)

with open(predict_path, 'w') as f:
    f.write('id,answer\n')
    for i, v in  enumerate(predict):
        f.write('%d,%d\n' %(i+1, v))

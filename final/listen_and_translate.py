# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
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
import io
import re

def random_permute(lst):
    """
    Generate negative samples for each audio input 
    """
    randomized_list = []
    iters = len(lst)

    for i in range(iters):
        index = np.delete(np.arange(iters), i)
        negative_sample = np.random.choice(index)
        intersect = np.intersect1d(lst[i,:], lst[negative_sample,:])
        while len(intersect) > 1:
            negative_sample = np.random.choice(index)
            intersect = np.intersect1d(lst[i,:], lst[negative_sample,:])
        randomized_list.append(lst[negative_sample,:])
    randomized_list = np.array(randomized_list)
    randomized_list = np.concatenate((lst, randomized_list), axis=0)
    return randomized_list

# Read audio file
with open("data/train.data", "rb") as fp:
    X = pickle.load(fp, encoding='latin1')
with open("data/test.data", "rb") as fp:
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

with codecs.open("data/train.caption", encoding='utf-8') as f:
    for line in f.readlines():
        X2.append(line)
        data.append(line.split())

with codecs.open("data/test.csv", encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split(',')
        X2_test.extend(line)

# Tokenize Chinese words into unique ids
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X2 + X2_test)
train_sequences = tokenizer.texts_to_sequences(X2)
test_sequences = tokenizer.texts_to_sequences(X2_test)

word_index = tokenizer.word_index
vocab = len(word_index) + 1
print('Found %s unique tokens.' % len(word_index))

# Pad Chinese sequences into equal length by adding zero ids for input shorter than maxlen
X2_maxlen = max([len(sentence) for sentence in train_sequences])
X2 = pad_sequences(train_sequences, maxlen=X2_maxlen)
X2_test = pad_sequences(test_sequences, maxlen=X2_maxlen)

# Prepare labels for two input sequences
# 1 means the audio sequence is related to the Chinese sequence
# 0 means the audio sequence is not related to the Chinese sequence
y = np.concatenate((np.ones(X1.shape[0]), np.zeros(X1.shape[0])), axis=0)

# Create new Gensim Word2Vec model
# Pretrain word vectors by Chinese sequences
w2v_model = Word2Vec(data, size=200, min_count=1, window=3, workers=10, iter=5)

embedding_matrix = np.random.random((len(word_index) + 1, 200))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i,:] = embedding_vector
        
    except KeyError:
        print("not in vocabulary")


taiwanese = Input(shape=(X1_maxlen, 39))
chinese = Input(shape=(X2_maxlen, ))

# Creating audio encoder network
l_taiwanese = Conv1D(200, 3, padding='valid', activation='relu')(taiwanese)
l_taiwanese = Bidirectional(LSTM(200, return_sequences=True, dropout=0.4))(l_taiwanese)

# attention
attention1 = TimeDistributed(Dense(1, activation='tanh', use_bias=False))(l_taiwanese) 
attention1 = Flatten()(attention1)
attention1 = Activation('softmax')(attention1)
attention1 = RepeatVector(400)(attention1)
attention1 = Permute([2, 1])(attention1)

l_taiwanese = multiply([attention1, l_taiwanese])
l_taiwanese = Lambda(lambda xin: K.sum(xin, axis=1))(l_taiwanese)
l_taiwanese = Dense(400, activation='linear', use_bias=False)(l_taiwanese)

# Creating Chinese encoder network
embedding_layer = Embedding(len(word_index) + 1,
                            200,
                            weights=[embedding_matrix],
                            input_length=X2_maxlen,
                            trainable=True, mask_zero=False)
l_chinese = embedding_layer(chinese)
l_chinese = Bidirectional(LSTM(200, return_sequences=True, dropout=0.4))(l_chinese)

# attention
attention2 = TimeDistributed(Dense(1, activation='tanh', use_bias=False))(l_chinese) 
attention2 = Flatten()(attention2)
attention2 = Activation('softmax')(attention2)
attention2 = RepeatVector(400)(attention2)
attention2 = Permute([2, 1])(attention2)

l_chinese = multiply([attention2, l_chinese])
l_chinese = Lambda(lambda xin: K.sum(xin, axis=1))(l_chinese)

# The dot product of the two tensors indicate their correlation
l_dot = Dot(axes=-1)([l_taiwanese, l_chinese])

preds = Dense(1, activation='sigmoid')(l_dot)
model = Model([taiwanese, chinese], preds)

# Compile the network by binary_crossentropy and rmsprop optimizer
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# The key part is to random sample new negative samples in every epoch
for k in range(300):
    print("{} th iteration:".format(k+1))
    x1, x1_valid, x2, x2_valid, y1, y1_valid = train_test_split(X1, random_permute(X2), y, test_size=0.05)
    model.fit([x1, x2], y1, epochs=1, batch_size=256, validation_data=([x1_valid, x2_valid], y1_valid))

model.save("modify_multimodel_permute_300.h5")

# Predict the option with the value most closest to 1
predict = model.predict([X1_test, X2_test])
predict = predict.reshape(-1, 4)
predict = np.argmax(predict, axis=1)

with open("modify_multimodel_permute_300.csv", 'w') as f:
    f.write('id,answer\n')
    for i, v in  enumerate(predict):
        f.write('%d,%d\n' %(i+1, v))

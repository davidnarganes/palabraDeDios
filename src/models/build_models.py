import sys
import os

sys.path.append(os.path.join("..","utils"))

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, Dense, Dropout, TimeDistributed, Activation, Flatten
from keras.optimizers import RMSprop, Adam

from utils import *
from model_utils import *

def multiCuDNNLSTM(embedding_vectors, input_length, batch_size, hidden_units=[512, 512, 512], dropout=0.2, training_mode=True):
    vocab_size = embedding_vectors.shape[0]
    embedd_dim = embedding_vectors.shape[1]

    # input_length and batch_size needs to be 1 for prediction

    model = Sequential() 
    model.add(Embedding(vocab_size, embedd_dim, weights=[embedding_vectors], input_length=input_length, batch_input_shape=(batch_size, input_length), trainable=False))
    
    for idx, hu in enumerate(hidden_units):
        model.add(Bidirectional(CuDNNLSTM(units=hu, stateful=True, return_sequences=(idx != len(hidden_units)-1))))
        model.add(Dropout(dropout))
    
    model.add(Dense(vocab_size, activation='softmax'))


    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # sparse categorical xentropy because we are using indices instad of one-hot encoding
    print(model.summary())

    return model



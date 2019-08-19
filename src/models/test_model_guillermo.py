import numpy as np
import sys
import os
import json

from gensim.models import Word2Vec

from keras.models import Sequential, load_model
from keras.layers import LSTM, CuDNNLSTM, Dense, Bidirectional, Embedding, Activation, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization, Dropout
from keras.callbacks import LambdaCallback, TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from hashlib import sha256

from tensorflow.keras.callbacks import TensorBoard

sys.path.append(os.path.join("..","utils"))
sys.path.append(os.path.join("..","preprocess"))
from utils import *
from preprocess import preprocess_func

def idx2word(idx, word2idx):
  keys = list(word2idx.keys())
  word = keys[idx]
  if word == '<white_space>':
    word = ' '
  elif word == '<tilde>':
    word = '´'
  return word

class batch_generator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=1):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step


    def generate(self):
      x = np.zeros((self.batch_size, self.num_steps))
      y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
      while True:
          for i in range(self.batch_size):
              if self.current_idx + self.num_steps >= len(self.data):
                  # reset the index back to the start of the data set
                  self.current_idx = 0
              x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
              temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
              # convert all of temp_y into a one hot representation
              y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
              self.current_idx += self.skip_step
          yield x, y
          
def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, word2idx, num_generated=200, T=0.5):
  sentence = [word2idx[word] for word in preprocess_func([text])]
  word_idxs = np.array(sentence, ndmin=2)
  for i in range(num_generated):
    prediction = model.predict(word_idxs)
    idx = sample(prediction[-1], temperature=T)
    sentence.append(idx)
    word_idxs = np.array(sentence[i+1:], ndmin=2)
  return ''.join(idx2word(idx, word2idx) for idx in sentence)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  text = 'y dios dijo'
  # model.layers[0].batch_input_shape = (1, window_width-1)
  # model.layers[1].stateful = False
  # model.layers[2].stateful = False
  # model.layers[3].stateful = False
  for idx in range(2):
    sample = generate_next(text, word2idx)
    print('%s... -> %s' % (idx, sample))
  
  # model.layers[0].batch_input_shape = (batch_size, window_width-1)
  # model.layers[1].stateful = True
  # model.layers[2].stateful = True
  # model.layers[3].stateful = True

w2v_modelpath = os.path.join("..","..","data","models","w2v","model_size_16_window_10_min_count_0_workers_8_seed_19_iter_5_.w2v")
w2v_model = Word2Vec.load(w2v_modelpath)
vectors=np.vstack([w2v_model[word] for word in w2v_model.wv.vocab.keys()])
words = list(w2v_model.wv.vocab)

word2idx = {word:idx for idx, word in enumerate(words)}

# GET DATA
in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
log_directory = os.path.join("..","..","data","models","log")
w2v_modelpath = os.path.join("..","..","data","models","w2v","model_size_16_window_10_min_count_1_workers_4_seed_123_iter_100_.w2v")
out_directory = os.path.join("..","..","data","models","keras")

window_width = 500

# GET DATA
biblia_preprocessed = read_bible(in_filepath)
biblia_inds = [word2idx[c] for c in biblia_preprocessed]
biblia_windows = np.array(list(make_windows(biblia_inds, window_width=window_width)))
# biblia_windows = list(chunks(biblia_inds, window_width))
# biblia_windows = np.array(biblia_windows[:-1], dtype=np.int32)

# to train in stateful mode it is necessary to match the number of training examples with the batch size
batch_size = 20 # minibatch size
num_epochs = 40 # number of epochs

X = biblia_windows[:,:-1]
Y = np.array(biblia_windows[:,-1],ndmin=2).T

# validation_split = 0 # training and validadtion splits need to be divisible by the batch size
# total_lines = biblia_windows.shape[0]
# train_lines = int(np.floor(total_lines*(1-validation_split)/batch_size)*batch_size)
# valid_lines = int(np.floor(total_lines*(validation_split)/batch_size)*batch_size)
# X = biblia_windows[0:train_lines,:-1]
# Y = np.array(biblia_windows[0:train_lines,-1],ndmin=2).T

# Xval = biblia_windows[train_lines:train_lines+valid_lines,:-1]
# Yval = np.array(biblia_windows[train_lines:train_lines+valid_lines,-1],ndmin=2).T

print(vectors.shape[0])
learning_rate = 0.0001
hidden_units = 512

model = Sequential() 
model.add(Embedding(vectors.shape[0], vectors.shape[1], weights=[vectors], input_length=window_width-1, trainable=False))
model.add(Bidirectional(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1), return_sequences=True)))
model.add(Dropout(0.4))
model.add(Bidirectional(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1), return_sequences=True)))
model.add(Dropout(0.4))
model.add(Bidirectional(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1))))
model.add(Dropout(0.4))
# model.add(LSTM(20, activation="tanh", input_shape=(vectors.shape[1], window_width-1), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(vectors.shape[0]))
model.add(Activation('softmax'))
optimizer = Adam(lr=learning_rate)
# optimizer = Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # SOLO FUNCIONA CON SPARSE_CATEGRICAL_XENTROPY XQ MI INPUT SON INTEGERS
print(model.summary()) 

# model = Sequential() 
# model.add(Embedding(vectors.shape[0], vectors.shape[1], weights=[vectors], input_length=window_width-1, trainable=False, batch_input_shape=(batch_size, window_width-1)))
# model.add(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1), stateful=True, return_sequences=True))
# model.add(Dropout(0.4))
# model.add(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1), stateful=True, return_sequences=True))
# model.add(Dropout(0.4))
# model.add(CuDNNLSTM(units=hidden_units, input_shape=(vectors.shape[1], window_width-1), stateful=True))
# model.add(Dropout(0.4))
# # model.add(LSTM(20, activation="tanh", input_shape=(vectors.shape[1], window_width-1), dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(vectors.shape[0]))
# model.add(Activation('softmax'))
# optimizer = Adam(lr=learning_rate)
# # optimizer = Adam()
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # SOLO FUNCIONA CON SPARSE_CATEGRICAL_XENTROPY XQ MI INPUT SON INTEGERS
# print(model.summary()) 

# path = 'C://Users//Sísifo//Documents//GitHub//palabraDeDios//data//models//log_guillermo'
logpath = r'D:\datos\datos_generacion_texto\Biblia\logs'
tbCallBack = TensorBoard(log_dir=logpath, histogram_freq=0, write_graph=True, write_images=True)
# python -m tensorboard.main --logdir=C:\Users\Sísifo\Documents\GitHub\palabraDeDios\src\models

out_directory = r'D:\datos\datos_generacion_texto\Biblia\char_models'
config = model.get_config()
sha_code = sha256(str(config).encode()).hexdigest()
modelpath = os.path.join(out_directory, "%s.keras" % sha_code)
config_filepath = modelpath.replace(".keras",".json")
with open(config_filepath, "w") as outfile:
    json.dump(config, outfile, indent=3)

# Calling TensorBard produces an obscure error with the optimizer. Only with this script
callbacks = [LambdaCallback(on_epoch_end=on_epoch_end), \
            EarlyStopping(patience=10, monitor='loss'), \
            ModelCheckpoint(filepath=modelpath+'.epochs_{epoch:02d}_loss_{loss:.2f}', monitor='loss', verbose=0, mode='auto', period=1)]

# from keras.models import load_model
# model = load_model("C://Users//Sísifo//Documents//GitHub//palabraDeDios//data//models//keras_guillermo//deep_model_biblia.05-1.55.hdf5")

model.fit(X, Y,
  batch_size=batch_size,
  shuffle=True,
  epochs=num_epochs,
  callbacks=callbacks,
  validation_split = 0.2)
  # validation_data=(Xval,Yval))




import numpy as np
import sys
import os

from gensim.models import Word2Vec

from keras.models import Sequential, load_model
from keras.layers import LSTM, CuDNNLSTM, Dense, Bidirectional, Embedding, Activation, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from keras.callbacks import LambdaCallback, TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

from tensorflow.keras.callbacks import TensorBoard

sys.path.append(os.path.join("..","utils"))
sys.path.append(os.path.join("..","preprocess"))
from utils import *
from preprocess import preprocess_func

def get_onehot_enccoding(words):
    char2int = dict((c, i) for i, c in enumerate(words))
    int2char = dict((i, c) for i, c in enumerate(words))
    integer_encoded = [char2int[c] for c in words]
    # one hot encode
    onehot = dict()
    for v,c in zip(integer_encoded,words):
        letter = [0 for _ in range(len(words))]
        letter[v] = 1
        onehot[c] = letter
    return onehot

def idx2word(idx, word2idx):
  # if type(indices) == int:
  #   indices = [indices]
  # keys = list(vocab.keys())
  # words = [keys[idx] for idx in indices]
  keys = list(word2idx.keys())
  word = keys[idx]
  if word == '<white_space>':
    word = ' '
  elif word == '<tilde>':
    word = '´'
  return word

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, word2idx, num_generated=10, T=0.2):
  sentence = [word2idx[word] for word in preprocess_func(text)]
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
  for idx in range(2):
    # sample = generate_next(random.choice(text, word2idx)[0:-1])
    # sample = generate_next(text[idx][0])
    sample = generate_next(text, word2idx)
    print('%s... -> %s' % (idx, sample))

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield np.array(l[i:i+n])

w2v_modelpath = os.path.join("..","..","data","models","w2v","model_size_16_window_10_min_count_0_workers_8_seed_19_iter_5_.w2v")
w2v_model = Word2Vec.load(w2v_modelpath)
vectors=np.vstack([w2v_model[word] for word in w2v_model.wv.vocab.keys()])
words = list(w2v_model.wv.vocab)
global word2idx1
word2idx = {word:idx for idx, word in enumerate(words)}

# GET DATA
in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
log_directory = os.path.join("..","..","data","models","log")
w2v_modelpath = os.path.join("..","..","data","models","w2v","model_size_16_window_10_min_count_1_workers_4_seed_123_iter_100_.w2v")
out_directory = os.path.join("..","..","data","models","keras")

window_width = 18

# GET DATA
biblia_preprocessed = read_bible(in_filepath)
biblia_inds = [word2idx[c] for c in biblia_preprocessed]
# biblia_windows = np.array(list(make_windows(biblia_inds, window_width=window_width)))
biblia_windows = list(chunks(biblia_inds, window_width))
biblia_windows = np.array(biblia_windows[:-1], dtype=np.int32)

X = biblia_windows[:,:-1]
Y = np.array(biblia_windows[:,-1],ndmin=2).T

print(vectors.shape[0])
learning_rate = 0.0001
model = Sequential() 
model.add(Embedding(vectors.shape[0], vectors.shape[1], weights=[vectors], input_length=window_width-1, trainable=False)) 
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(CuDNNLSTM(units=20, input_shape=(vectors.shape[1], window_width-1))))
# model.add(LSTM(20, activation="tanh", input_shape=(vectors.shape[1], window_width-1), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(vectors.shape[0]))
model.add(Activation('softmax'))
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # SOLO FUNCIONA CON SPARSE_CATEGRICAL_XENTROPY XQ MI INPUT SON INTEGERS
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
print(model.summary()) 

# path = 'C://Users//Sísifo//Documents//GitHub//palabraDeDios//data//models//log_guillermo'
path = os.getcwd()
tbCallBack = TensorBoard(log_dir=path, histogram_freq=0, write_graph=True, write_images=True)
# python -m tensorboard.main --logdir=C:\Users\Sísifo\Documents\GitHub\palabraDeDios\src\models

save_dir = "C://Users//Sísifo//Documents//GitHub//palabraDeDios//data//models//keras_guillermo//"
# os.mkdir(save_dir)

callbacks = [LambdaCallback(on_epoch_end=on_epoch_end), \
            EarlyStopping(patience=10, monitor='loss'), \
            ModelCheckpoint(filepath=save_dir + "/" + 'modelo_biblia.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0, mode='auto', period=10)]
# callbacks = [LambdaCallback(on_epoch_end=on_epoch_end), tbCallBack, EarlyStopping(patience=10, monitor='loss'), \
#   ModelCheckpoint(filepath=save_dir + "/" + 'modelo_biblia.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0, mode='auto', period=2)]
# callbacks=[EarlyStopping(patience=4, monitor='val_loss')]

batch_size = 10 # minibatch size
num_epochs = 100 # number of epochs

model.fit(X, Y,
  batch_size=batch_size,
  shuffle=True,
  epochs=num_epochs,
  callbacks=callbacks,
  validation_split=0.3)



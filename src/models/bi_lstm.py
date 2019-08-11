import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import time
import json
import numpy as np
import multiprocessing
from hashlib import sha256

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from gensim.models import Word2Vec

sys.path.append(os.path.join("..","utils"))
sys.path.append(os.path.join("..","preprocess"))
from utils import *
from preprocess import preprocess_func

def get_onehot_enc(w2v_model):
    chars = list(w2v_model.wv.vocab)
    char2int = dict((c, i) for i, c in enumerate(chars))
    int2char = dict((i, c) for i, c in enumerate(chars))
    integer_encoded = [char2int[c] for c in chars]
    # one hot encode
    onehot = dict()
    for v,c in zip(integer_encoded,chars):
        letter = [0 for _ in range(len(chars))]
        letter[v] = 1
        onehot[c] = letter
    return onehot

def stackemb(SENT, DICT):
    return [DICT[c] for c in SENT]

def embed_window_parallel(window, w2v_model):
    start = time.time()
    emb_dict = {c:w2v_model.wv[c] for c in w2v_model.wv.vocab.keys()}
    n = multiprocessing.cpu_count()
    with multiprocessing.Pool(n) as p:
        args = zip(window, [emb_dict]*len(window))
        emb = p.starmap(stackemb, args)
        p.close()
    emb = np.asarray(emb, dtype=float)
    print('Time for embedding: %.2f sec' % (time.time() - start))
    return emb

def embed_sent(sent, w2v_model):
    emb_dict = {c:w2v_model.wv[c] for c in w2v_model.wv.vocab.keys()}
    emb = [emb_dict[c] for c in sent]
    emb = np.asarray(emb)
    emb = emb.reshape(1,emb.shape[0],emb.shape[1])
    return emb

def make_verse(sentence, w2v_model, model, length=40):
    if type(sentence)!=str:
        raise ValueError("'bait' must be str")
    if len(sentence) != 11:
        raise ValueError("len of sentence must be 11")
    toreturn = sentence
    bait = preprocess_func([sentence])
    chars = list(w2v_model.wv.vocab)

    remplace_dict = {
        '<unknown>':'|',
        '<tilde>':'`',
        '<end_line>':'\n',
        '<white_space>':' ',
        '<dieresis>':'¨'
        }

    count = len(bait)
    while count < length:
        # Deprocess
        e = embed_sent(bait, w2v_model)
        p = model.predict(e)
        nextchar = np.random.choice(chars, size=1, p = p.ravel())[0]

        # Get nextchar
        bait.append(nextchar)
        if len(nextchar) > 1:
            toreturn += remplace_dict[nextchar]
        else:
            toreturn += nextchar
        bait = bait[1:]
        count +=1

    return toreturn 

if __name__ == "__main__":

    # GET DATA
    in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
    log_directory = os.path.join("..","..","data","models","log")
    w2v_modelpath = os.path.join("..","..","data","models","w2v","model_.w2v")
    out_directory = os.path.join("..","..","data","models","keras")
    mknewdir(log_directory)
    mknewdir(out_directory)

    # GET DATA
    biblia_preprocessed = read_bible(in_filepath)
    biblia_windows = make_windows(biblia_preprocessed, window_width=12)
    biblia_windows = list(biblia_windows)
    w2v_model = Word2Vec.load(w2v_modelpath)

    biblia_embedding = embed_window_parallel(biblia_windows, w2v_model)
    X = biblia_embedding[:,:11,:]
    onehot = get_onehot_enc(w2v_model)
    y = [c[-1] for c in biblia_windows]
    y = [onehot[c] for c in y]
    y = np.asarray(y).reshape(len(biblia_windows), len(w2v_model.wv.index2word))

    # DEFINE MODEL
    tensorboard = TensorBoard(log_dir=log_directory)
    model = Sequential()
    model.add(Bidirectional(LSTM(4, dropout=.2), input_shape=X.shape[1:]))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    config = model.get_config()
    sha_code = sha256(str(config).encode()).hexdigest()
    modelpath = os.path.join(out_directory, "%s.keras" % sha_code)
    config_filepath = model_name.replace(".keras",".json")
    with open(config_filepath, "w") as outfile:
        json.dump(config, outfile, indent=3)

    # TRAIN
    earlystop = EarlyStopping(monitor='val_acc', patience=10)
    for iteration in range(5):

        print('iter %d' % iteration)
        s = [make_verse('y dios dijo', w2v_model, model) for x in range(3)]
        print('\n%s\n\n' % '\n'.join(s))

        model.fit(X,y, batch_size=10000, epochs = 1, validation_split=.3, shuffle=True, verbose=1, callbacks=[earlystop,tensorboard])
    # Visualise: http://localhost:6006/
    model.save(modelpath)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import time
import json
import numpy as np
import multiprocessing
from hashlib import sha256, blake2b

from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
# from keras.layers import BatchNormalization
# from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks.tensorboard_v1 import TensorBoard
from gensim.models import Word2Vec

sys.path.append(os.path.join("/Users/dnarganes/repos/palabraDeDios/src/utils"))
sys.path.append(os.path.join("/Users/dnarganes/repos/palabraDeDios/src/preprocess"))
from utils import read_bible, make_windows
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
    print("Time for embedding: %.2f sec" % (time.time() - start))
    return emb

def embed_sent(sent, w2v_model):
    emb_dict = {c:w2v_model.wv[c] for c in w2v_model.wv.vocab.keys()}
    emb = [emb_dict[c] for c in sent]
    emb = np.asarray(emb)
    emb = emb.reshape(1,emb.shape[0],emb.shape[1])
    return emb

def make_verse(sentence, w2v_model, model, length=100):
    if type(sentence)!=str:
        raise ValueError("`bait` must be str")
    if len(sentence) != 11:
        raise ValueError("len of sentence must be 11")
    toreturn = sentence
    bait = preprocess_func([sentence])
    chars = list(w2v_model.wv.vocab)

    remplace_dict = {
        "<unknown>":"|",
        "<tilde>":"`",
        "<end_line>":"\n",
        "<white_space>":" ",
        "<dieresis>":"¨"
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

def get_modelname(config, digest_size=10):

    # encode
    s = str(config).encode()

    # hash
    h = blake2b(digest_size=digest_size)
    h.update(s)

    return h.hexdigest()


if __name__ == "__main__":

    # GET DATA
    in_directory = "/Users/dnarganes/repos/palabraDeDios"
    in_filepath = os.path.join(in_directory, "data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
    log_directory = os.path.join(in_directory,"data","models","log")
    w2v_modelpath = os.path.join(in_directory,"data","models","w2v","model_size_16_window_10_min_count_1_workers_4_seed_123_iter_100_.w2v")
    out_directory = os.path.join(in_directory,"data","models","keras")

    # Make dirs
    dirs = [log_directory, out_directory]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

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
    model = Sequential()
    model.add(Bidirectional(LSTM(64, dropout=0.3), input_shape=X.shape[1:]))
    # model.add(Bidirectional(LSTM(64, dropout=0.3)))
    model.add(Dense(y.shape[1], activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    # Naming
    config = model.get_config()
    model_name = get_modelname(config)
    model_dir = os.path.join(out_directory, model_name)

    # Paths
    model_path = os.path.join(model_dir, "{epoch:02d}.hdf5")
    config_path = os.path.join(model_dir, "config.json")
    arch_path = os.path.join(model_dir, "architecture.png")

    # Save
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(config_path, "w") as outfile:
        json.dump(config, outfile, indent=3)

    plot_model(
        model=model,
        to_file=arch_path,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False, 
        dpi=256
        )

    # TRAIN
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5
        )
    tensorboard_callback = TensorBoard(
        log_dir=log_directory
        )
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        save_weights_only=False, 
        monitor="val_loss",
        mode="min"
        )

    for iteration in range(30):

        print("iter %d" % iteration)
        s = [make_verse("and god said", w2v_model, model) for x in range(3)]
        print("\n%s\n\n" % "\n".join(s))

        # Train
        model.fit(
            x=X,
            y=y,
            epochs=1,
            batch_size=X.shape[0] // 1000,
            verbose=1,
            validation_split=0.3,
            shuffle=True,
            callbacks=[tensorboard_callback, early_stopping, checkpoint],
            )

    # Visualise: http://localhost:6006/



import os
import sys
import json
import functools
import numpy as np
from hashlib import blake2b

from keras.utils import plot_model
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Bidirectional, Embedding, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks.tensorboard_v1 import TensorBoard
from keras.callbacks import Callback
from gensim.models import Word2Vec

sys.path.append("/Users/dnarganes/repos/palabraDeDios/src/preprocess")

from english import get_bible, window, SaveModel, PlotChars

def make_lstm(model, X, y, n_layers=2):

    # Embedding
    Embedding_Layer = Embedding(
        y.shape[1],
        model["w2v"].wv.vector_size,
        weights=[model["w2v"].wv.vectors],
        input_length=X.shape[1:],
        trainable=False,
        )

    MAX_SEQUENCE_LENGTH = X.shape[1]
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int16")
    embedded_sequences = Embedding_Layer(sequence_input)
    x = LSTM(128, dropout=0.2, return_sequences=True)(embedded_sequences)
    x = LSTM(128, dropout=0.2)(x)
    # x = LSTM(32, dropout=0.3)(x)
    preds = Dense(y.shape[1], activation="softmax")(x)

    m = Model(sequence_input, preds)
    m.compile(loss="categorical_crossentropy", optimizer="adam")

    return m

def idx2hot(i, n):
    x = np.zeros(n)
    x[i] += 1
    return x

def get_modelname(config, digest_size=10):

    # encode
    s = str(config).encode()

    # hash
    h = blake2b(digest_size=digest_size)
    h.update(s)

    return h.hexdigest()

class Recite(Callback):

    def __init__(self, bait, char2idx, out_directory,
        n_versicles=20, max_length=400):

        assert len(bait) == 12

        self.epoch = 0
        self.bait = bait

        self.seed_length = len(bait)
        self.char2idx = char2idx
        self.idx2char = {v:k for k,v in d.items()}
        self.idxlist = list(map(self.char2idx.get, self.bait))

        self.max_length = max_length
        self.n_versicles = n_versicles
        self.out_directory = out_directory

    def recite(self):


        # idxlist = list(map(d.get, bait))
        idxlist = self.idxlist.copy()
        nextidx = None
        while nextidx != d["%"] and (len(idxlist) < self.max_length):

            # asarray
            s = np.array(idxlist[-self.seed_length:]).reshape(1,-1)

            # Next char
            p = self.model.predict(s).ravel()
            nextidx = np.random.choice(list(self.idx2char), p=p)

            # Append
            idxlist.append(nextidx)

        versicle = "".join(map(self.idx2char.get, idxlist))

        return versicle

    def on_train_begin(self, logs={}):
        
        filename = os.path.join(self.out_directory, "recite_epoch-%02d.txt" % self.epoch)

        with open(filename, "w") as outfile:

            for _ in range(self.n_versicles):

                versicle = self.recite()

                outfile.write(versicle + "\n")

                print(versicle)

        self.epoch +=1

    def on_epoch_end(self, epoch, logs={}):

        filename = os.path.join(self.out_directory, "recite_epoch-%02d.txt" % self.epoch)

        with open(filename, "w") as outfile:

            for _ in range(self.n_versicles):

                versicle = self.recite()

                outfile.write(versicle + "\n")

                print(versicle)

        self.epoch +=1


if __name__ == "__main__":

    # Get bible
    out_directory = "/Users/dnarganes/repos/palabraDeDios/data/models/keras"
    log_directory = "/Users/dnarganes/repos/palabraDeDios/data/models/log"
    bait = "and god said"
    bible = get_bible()

    # Load model
    f = "/Users/dnarganes/repos/palabraDeDios/data/models/w2v/62473ee7f1aedbfd5a9f/epoch-10.w2v"
    model = dict()
    model["w2v"] = Word2Vec.load(f)

    # Index
    d = {v:k for k,v in enumerate(model["w2v"].wv.index2word)}
    bible = map(d.get, bible)
    bible = window(bible, len(bait) + 1)
    bible = list(bible)
    bible = np.array(bible)

    # Split
    X,y = bible[:,:-1], bible[:,-1]
    f = functools.partial(idx2hot, n=len(d))
    y = map(f, y)
    y = np.array(list(y))

    # Model
    model["lstm"] = make_lstm(model, X, y)

    # Naming
    config = model["lstm"].get_config()
    model_name = get_modelname(config)
    model_dir = os.path.join(out_directory, model_name)

    # Paths
    model_path = os.path.join(model_dir, "epoch-{epoch:02d}.hdf5")
    config_path = os.path.join(model_dir, "config.json")
    arch_path = os.path.join(model_dir, "architecture.png")

    # Save
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(config_path, "w") as outfile:
        json.dump(config, outfile, indent=3)

    plot_model(
        model=model["lstm"],
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

    recite = Recite(bait, d, model_dir, n_versicles=30, max_length=400)

    # Train
    model["lstm"].fit(
        x=X,
        y=y,
        epochs=50,
        batch_size=X.shape[0] // 5000,
        verbose=1,
        validation_split=0.2,
        shuffle=True,
        callbacks=[tensorboard_callback, early_stopping, checkpoint, recite],
        )

    
    


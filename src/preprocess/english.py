import os
import json
import string
import logging
import numpy as np
from itertools import islice
from hashlib import sha256, blake2b

import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from nltk.corpus import gutenberg


"""
Remember that:
    - `%` is end of sentence
    - ` ` is whitespace
"""


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield list(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield list(result)

def get_modelname(config, digest_size=10):

    # encode
    s = str(config).encode()

    # hash
    h = blake2b(digest_size=digest_size)
    h.update(s)

    return h.hexdigest()

def save_config(out_directory, d):
    config_path = os.path.join(out_directory, "config.json")
    with open(config_path, "w") as fp:
        json.dump(d, fp, indent=3)

def get_bible():
    """
    Function to get Bible as a list of chars
    """

    sentences = gutenberg.sents("bible-kjv.txt")

    # Lowercase
    eos = "%"
    sentences = [" ".join(s).lower() + eos for s in sentences]

    # Join
    bible_str = " ".join(sentences)

    # Replace chars
    chars = set("".join(sentences))

    # list
    bible_lst = list(bible_str)

    return bible_lst

class SaveModel(CallbackAny2Vec):
    """
    Callback to save model after each epoch
    """
    
    def __init__(self, out_directory):

        self.out_directory = out_directory
        self.epoch = 0
        
    def on_epoch_end(self, model):
        
        self.epoch += 1

        # Create directory for epoch
        filename = os.path.join(self.out_directory, "epoch-%02d.w2v" % self.epoch)

        # Save model
        model.save(filename)

class PlotChars(CallbackAny2Vec):
    """
    Callback to plot chars after each epoch
    """
    
    def __init__(self, out_directory):

        self.out_directory = out_directory
        self.epoch = 0

    def _get_color(self, s):
        
        vowels = set("aeiou")
        if s in vowels:
            return "orange"

        if s in list(string.digits):
            return "purple"

        if s in set(string.ascii_lowercase) - set(vowels):
            return "turquoise"

        return "gray"
        
    def on_epoch_begin(self, model):

        # matrix
        X = np.array([model.wv[x] for x in model.wv.index2word])

        # models
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2, init="pca", random_state=123)

        coords = dict()
        coords["pca"] = pca.fit_transform(X)
        coords["tsne"] = tsne.fit_transform(X)

        plt.close("all")
        fig, axes = plt.subplots(figsize=(8,5), nrows=1, ncols=2, constrained_layout=True)
        axes = {k:v for k,v in enumerate(axes.ravel())}

        X = coords["pca"][:,0]
        Y = coords["pca"][:,1]
        axes[0].scatter(x=X, y=Y, color=[self._get_color(x) for x in model.wv.index2word])
        axes[0].set_title("PCA", fontsize=20)
        axes[0].set_xlabel("PC1", fontsize=16)
        axes[0].set_ylabel("PC2", fontsize=16)
        bbox = dict(boxstyle="round", facecolor="wheat", alpha=0.15)

        for x, y, s in zip(X, Y, model.wv.index2word):

            # place a text box in upper left in axes coords
            axes[0].text(x, y, s, fontsize=14, bbox=bbox)

        X = coords["tsne"][:,0]
        Y = coords["tsne"][:,1]
        axes[1].scatter(x=X, y=Y, color=[self._get_color(x) for x in model.wv.index2word])
        axes[1].set_title("tSNE", fontsize=20)
        axes[1].set_xlabel("dim1", fontsize=16)
        axes[1].set_ylabel("dim2", fontsize=16)
        bbox = dict(boxstyle="round", facecolor="wheat", alpha=0.15)
        
        for x, y, s in zip(X, Y, model.wv.index2word):

            # place a text box in upper left in axes coords
            axes[1].text(x, y, s, fontsize=14, bbox=bbox)

        pic_path = os.path.join(self.out_directory, "plot_epoch-%02d.png" % self.epoch)

        plt.savefig(pic_path, dpi=200)

        self.epoch += 1


if __name__ == "__main__":

    # Paths
    out_directory = "/Users/dnarganes/repos/palabraDeDios/data/models/w2v"

    # Logging
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    
    # Load Bible
    bible = get_bible()
    bible = list(window(bible, 20))

    # Make embeddings
    args = {
        "min_count":1,
        "window":4,
        "sg":1,
        "cbow_mean":1,
        "workers":4,
        "size":8,
        "hs":0,
        # "negative":0,
        # "alpha":2e-2,
        "max_final_vocab":100,
        "sorted_vocab":True,
        "compute_loss":True,
        }
    model = Word2Vec(**args)
    model.build_vocab(bible)

    # Paths
    d = {k:str(v) for k,v in model.__dict__.items()}
    model_name = get_modelname(d)
    outdir = os.path.join(out_directory, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Save config
    save_config(outdir, d)

    # Train
    model.train(
        sentences=bible,
        total_examples=model.corpus_count,
        epochs=10,
        callbacks=[
            SaveModel(out_directory=outdir),
            PlotChars(out_directory=outdir)
            ]
        )
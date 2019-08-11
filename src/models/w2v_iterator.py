import os
import sys
import time
import logging
import multiprocessing
from itertools import islice
from gensim.models import Word2Vec

sys.path.append(os.path.join("..","utils"))
from utils import *

def make_windows(list_, window_width=30):
    """
    Function to take a list and yield sublists of a given window size

    Args:
        - list_
        - window_width
    
    Returns:
        - yield generator of a sliding window from list_
            s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    
    it = iter(list_)
    result = tuple(islice(it, window_width))
    if len(result) == window_width:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def read_bible(in_filepath):
    with open(in_filepath, "r") as infile:
        lines = infile.read()
    return lines.split("\t")

if __name__ == "__main__":
    in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
    out_directory = os.path.join("..","..","data","models","w2v")
    mknewdir(out_directory)

    w2v_args = {
        "size":32,
        "window":8,
        "min_count":1,
        "workers":multiprocessing.cpu_count(),
        "seed":123,
        "iter":70,
    }

    # GET DATA
    biblia_preprocessed = read_bible(in_filepath)
    biblia_windows = make_windows(biblia_preprocessed)
    biblia_windows = list(biblia_windows)

    w2v_model = Word2Vec(**w2v_args)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    modelname = "model%s.w2v" % dict2str(w2v_args)
    out_modelpath = os.path.join(out_directory, modelname)   

    # TRAIN
    w2v_model.build_vocab(biblia_windows)
    w2v_model.train(biblia_windows, total_examples=w2v_model.corpus_count, epochs=1000)
    w2v_model.save(out_modelpath)
import os
import sys
import time
import logging
import multiprocessing
from gensim.models import Word2Vec

sys.path.append(os.path.join("..","utils"))
from utils import *

if __name__ == "__main__":
    in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
    out_directory = os.path.join("..","..","data","models","w2v")
    mknewdir(out_directory)

    epochs = 50
    params = {
        "size":16,
        "window":10,
        "min_count":1,
        "workers":multiprocessing.cpu_count(),
        "seed":123,
        "iter":100,
    }

    # GET DATA
    biblia_preprocessed = read_bible(in_filepath)
    biblia_windows = make_windows(biblia_preprocessed)
    biblia_windows = list(biblia_windows)

    w2v_model = Word2Vec(**params)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    modelname = "model%s.w2v" % dict2str(params)
    out_modelpath = os.path.join(out_directory, modelname)   

    # TRAIN
    w2v_model.build_vocab(biblia_windows)
    w2v_model.train(biblia_windows, total_examples=w2v_model.corpus_count, epochs=epochs)
    w2v_model.save(out_modelpath)
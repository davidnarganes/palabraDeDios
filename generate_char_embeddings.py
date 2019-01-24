import string
import re
from gensim.models import Word2Vec
from itertools import islice
from timeit import default_timer as time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np


# def tokenise_char(sentences, remplace_list):
#     return_ = list()
#     for s in sentences:
#         for tup in remplace_list:
#             s = s.replace(tup[0],tup[1])
#         return_.append(s.split())
#     return return_

def tokenise_char(s, remplace_list):
    for tup in remplace_list:
        s = s.replace(tup[0],tup[1])
    return s

def flat_list(nested_list):
    while any([isinstance(x,list) for x in nested_list]):
        nested_list = [item for sublist in nested_list for item in sublist]
    return nested_list

def is_int(string):
    try:
        int(string)
        return 'b'
    except:
        return 'r'

def get_colordict(chars):
    red_chars = ['<dieresis>','<tilde>','<end_line>','<white_space>']
    green_chars = list(string.punctuation)
    blue_chars = list(string.digits)
    orange_chars = list(string.ascii_letters)
    colordict = dict()
    for k in chars:
        if k in green_chars:
            colordict[k] = [.16,.64,.16]
            continue
        if k in blue_chars:
            colordict[k] = [.12,.56,1.0]
            continue
        if k in red_chars:
            colordict[k] = [.9,.0,.0]
            continue
        if k in orange_chars:
            colordict[k] = [1.0,.33,.12]
            continue
        else:
            colordict[k] = [1.0,1.0,.0]
    return colordict

def preprocess1(biblia):
    # 1. Join in one string
    biblia = ''.join(biblia)
    # 2. Simplify \n\n to \n
    biblia = biblia.replace('\n\n','$*$')
    # 3. Replace '\n' while keeping '.\n'
    biblia = biblia.replace('.\n','*$*')
    biblia = biblia.replace('\n','')
    biblia = biblia.replace('$*$','\n')
    biblia = biblia.replace('*$*','.\n')
    # 4. Delete several spaces
    biblia = re.sub(' {2,}', ' ', biblia)
    # 5. Lowercase
    biblia = biblia.lower()
    # 6. Replace chars
    remplace_list = [
        ('á','a <tilde> '),
        ('é','e <tilde> '),
        ('í','i <tilde> '),
        ('ó','o <tilde> '),
        ('ú','u <tilde> '),
        ('ü','u <dieresis> '),
        ('\n','<end_line> ')
        ]
    chars = string.ascii_letters+string.digits
    chars = chars + string.punctuation
    chars = chars + '¡¿ñ'
    remplace_list = [(char,'{} '.format(char)) for char in chars] + remplace_list
    for tup in remplace_list:
        biblia = biblia.replace(tup[0],tup[1])
    # 7. Replace 2+ white spaces for representation
    biblia = re.sub(' {2,}', ' <white_space> ', biblia)
    # 8. Split into chars
    biblia = biblia.split()
    # 9. Replace non-valid chars
    valid_chars = list(chars) + ['<dieresis>','<tilde>','<white_space>','<end_line>']
    biblia = ['<unknown>' if c not in valid_chars else c for c in biblia]
    return biblia

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

#----------------------------
# PIPELINE
#----------------------------

with open('data/Biblia/procesado_1/biblia_no_encabezados.txt','r', encoding='latin1') as handle:
    biblia_raw = handle.readlines()

biblia = preprocess1(biblia_raw)
# Verify chars
chars = set(biblia)
# Sliding window
biblia_w = list(window(biblia, n=30))

# Run w2v
epochs = int(100)
model = Word2Vec(size=300, window=10, min_count=2, workers=-1)


model = Word2Vec(biblia_w, size=20, window=5, min_count=0, workers=multiprocessing.cpu_count()-1)
print('model generated')

# build the vocabulary
model.build_vocab(biblia_w, update=True)
print('vocabulary created')

# train model
model.train(biblia_w, total_examples=model.corpus_count, epochs=1)
a = model.wv.vectors
model.train(biblia_w, total_examples=model.corpus_count, epochs=1)
b = model.wv.vectors
a == b



model.build_vocab(biblia_w)
chars = list(model.wv.vocab.keys())

# Color_dict
colordict = get_colordict(chars)
tsne = TSNE(n_iter=1000, random_state=0, init='pca', angle=.85)

for i in range(100):
    t0 = time()
    model.train(biblia_w, total_examples=model.corpus_count, epochs=epochs)
    model.save("data/models/char_w2v_{}.model".format(i))
    char_embedding = model.wv.vectors
    np.save('data/embeddings/char_embedding_{}.npy'.format(i), char_embedding)
    print('Saved. Time: %.2f sec' % (time() - t0))

    if i % 5 == 0:
        # Visualise: Takes a while tSNE
        embedding_2d = tsne.fit_transform(char_embedding)
        plt.figure(i)
        plt.scatter(embedding_2d[:,0],[embedding_2d[:,1]], alpha=0.0)
        for c,loc in zip(chars, embedding_2d):
            plt.text(loc[0],loc[1], c, ha="center", va="center", size=9,
                bbox=dict(boxstyle="round",
                        ec=[.3*x for x in colordict[c]],
                        alpha=0.6,
                        fc=colordict[c],
                        )
                )
        plt.tight_layout()
        plt.show()
        # plt.savefig('results/char_embedding_tSNE_{}.pdf'.format(i))
        plt.clf()
    
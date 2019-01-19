import string
from gensim.models import Word2Vec

def tokenise_char(sentences):
    return_ = list()
    for s in sentences:
        for tup in remplace_list:
            s = s.replace(tup[0],tup[1])
        return_.append(s.split())
    return return_

def flat_list(nested_list):
    while any([isinstance(x,list) for x in nested_list]):
        nested_list = [item for sublist in nested_list for item in sublist]
    return nested_list

with open('data/Biblia/procesado_1/biblia_procesada_1.txt','r', encoding='latin1') as handle:
    biblia_raw = handle.readlines()

# Define some vars to replace
biblia_remplaced = [s.split() for s in biblia_raw]
remplace_list = [
('á',' a <tilde> '),
('é',' e <tilde> '),
('í',' i <tilde> '),
('ó',' o <tilde> '),
('ú',' u <tilde> '),
('\n',' <end_line> ')
]

list_chars = [
    'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z',
 'ñ',
 'ü',
]

remplace_list = [(char,' {} '.format(char)) for char in list_chars] + remplace_list
unknown = '<unk>'
pad = '<pad>'

# Replace
biblia_rep = tokenise_char(biblia_raw)

# Verify chars
chars = set(flat_list(biblia_rep))

# Run w2v
epochs = int(1e6)
model = Word2Vec(size=300, window=10, min_count=1, workers=-1)
model.build_vocab(biblia_rep)
%timeit model.train(biblia_rep, total_examples=model.corpus_count, epochs=epochs)

char_embedding = model.wv.vectors
chars = list(model.wv.vocab.keys())

# Visualise
from sklearn.manifold import TSNE
tsne = TSNE(n_iter=10000)
embedding_2d = tsne.fit_transform(char_embedding)

from matplotlib import pyplot as plt
plt.scatter(embedding_2d[:,0], embedding_2d[:,1])
for char,loc in zip(chars, embedding_2d):
    plt.text(loc[0], loc[1], char)
plt.show()
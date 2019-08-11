'''
1. Load vectors from the embedding
2. Reduce the dimensionality using tSNE
3. Represent words in 2D space
'''
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import string
import sys
import os

sys.path.append(os.path.join("..","utils"))
from utils import *

model_name = 'model_size_16_window_5_min_count_0_workers_8_seed_19_iter_5_.w2v'
path_to_model = os.path.join("..","..","data","models","w2v", model_name)

def get_colordict(chars):
    '''
    It returns a dictionary with values character:RGB
    chars is a list of characters
    '''

    modifiers = ['<dieresis>','<tilde>']
    spaces = ['<end_line>','<white_space>']
    punct = list('!,.:;?¿¡')
    tags  = ['<unknown>']
    symbols = list('"#$%&\'()*+-/<=>@[\\]^_`{|}~')
    digits = list(string.digits)
    vowels = list('aeiou')
    consonants = [l for l in set(list(string.ascii_letters.lower())) if l not in vowels]
    consonants.append('ñ')
    others = [c for c in chars if c not in unnest([modifiers, spaces, punct, tags, symbols, digits, vowels, consonants])]

    colordict = dict()
    for k in modifiers:
        colordict[k] =  np.array([255,0,0])/255 # red

    for k in spaces:
        colordict[k] =  np.array([0,255,0])/255 # green

    for k in punct:
        colordict[k] =  np.array([0,0,255])/255 # blue

    for k in tags:
        colordict[k] =  np.array([0,0,128])/255 # dark blue

    for k in symbols:
        colordict[k] =  np.array([200,200,200])/255 # grey

    for k in digits:
        colordict[k] =  np.array([255,0,255])/255 # magenta

    for k in vowels:
        colordict[k] =  np.array([255,128,0])/255 # orange

    for k in consonants:
        colordict[k] =  np.array([0,255,255])/255 # teal

    for k in others:
        colordict[k] =  np.array([100,100,100])/255 # dark grey

    return colordict, others

# Load vectors and words
model = Word2Vec.load(path_to_model)
vectors=np.vstack([model[word] for word in model.wv.vocab.keys()])
words = list(model.wv.vocab)
colordict, others = get_colordict(words)

# Compute tSNE
tsne = TSNE(n_iter=1000000, perplexity=10, learning_rate=1, metric='cosine', init='pca')
embedding_2d = None
embedding_2d = tsne.fit_transform(vectors)

# Plot positions
plt.figure()
plt.scatter(embedding_2d[:,0],[embedding_2d[:,1]], alpha=0.0)

# # Shorten some tags for plotting
# words =  ['<u>' if w == '<unknown>' else w for w in words]
# words =  ['<¨>' if w == '<dieresis>' else w for w in words]
# words =  ['<´>' if w == '<tilde>' else w for w in words]
# words =  ['<\w>' if w == '<white_space>' else w for w in words]
# words =  [r'<\n>' if w == '<end_line>' else w for w in words]

# Plot text
for c,loc in zip(words, embedding_2d):
    
    if c == '<unknown>':
        label = '<u>'
    elif c == '<dieresis>':
        label = '<¨>'
    elif c == '<tilde>':
        label = '<´>'
    elif c == '<white_space>':
        label = r'<\w>'
    elif c == '<end_line>':
        label = r'<\n>'
    else:
        label = c

    plt.text(loc[0], loc[1], label, ha="center", va="center", size=9,
        bbox=dict(boxstyle='Circle',
                alpha=0.5,
                fc=colordict[c],
                ec = 'none'
                )
        )

plt.tight_layout()
plt.axis('off')
plt.show()
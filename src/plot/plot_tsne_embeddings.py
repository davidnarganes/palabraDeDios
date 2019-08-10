'''
1. Load vectors from the embedding
2. Reduce the dimensionality using tSNE
3. Represent words in 2D space
'''
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

path_to_model = 'character_embeddings_guillermo.w2v'

def get_colordict():
    # red_chars = ['<dieresis>', '<tilde>', '<end_line>', '<white_space>']

    modifiers = ['<¨>','<´>']
    spaces = [r'<\n>','<\w>']
    punct = list('!,.:;?¿¡')
    tags  = ['<u>']
    symbols = list('"#$%&\'()*+-/<=>@[\\]^_`{|}~')
    digits = list(string.digits)
    vowels = list('aeiou')
    consonants = [l for l in set(list(string.ascii_letters.lower())) if l not in vowels]
    consonants.append('ñ')

    colordict = dict()
    for k in modifiers:
        colordict[k] =  np.array([255,0,0])/255
    
    for k in spaces:
        colordict[k] =  np.array([0,255,0])/255

    for k in punct:
        colordict[k] =  np.array([0,0,255])/255

    for k in tags:
        colordict[k] =  np.array([0,0,128])/255

    for k in symbols:
        colordict[k] =  np.array([200,200,200])/255

    for k in digits:
        colordict[k] =  np.array([255,0,255])/255

    for k in vowels:
        colordict[k] =  np.array([255,128,0])/255

    for k in consonants:
        colordict[k] =  np.array([0,255,255])/255

    return colordict

model = Word2Vec.load(path_to_model)
vectors=np.vstack([model[word] for word in model.wv.vocab.keys()])
words = list(model.wv.vocab)

colordict = get_colordict()
tsne = TSNE(n_iter=1000000, perplexity=5, learning_rate=1, metric='cosine', init='pca')
embedding_2d = None
embedding_2d = tsne.fit_transform(vectors)
plt.figure()
plt.scatter(embedding_2d[:,0],[embedding_2d[:,1]], alpha=0.0)

words =  ['<u>' if w == '<unknown>' else w for w in words]
words =  ['<¨>' if w == '<dieresis>' else w for w in words]
words =  ['<´>' if w == '<tilde>' else w for w in words]
words =  ['<\w>' if w == '<white_space>' else w for w in words]
words =  [r'<\n>' if w == '<end_line>' else w for w in words]

for c,loc in zip(words, embedding_2d):
    plt.text(loc[0], loc[1], c, ha="center", va="center", size=9,
        bbox=dict(boxstyle='Circle',
                alpha=0.5,
                fc=colordict[c],
                ec = 'none'
                )
        )

plt.tight_layout()
plt.axis('off')
plt.show()
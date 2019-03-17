# base
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from itertools import islice
import multiprocessing
import numpy as np
from random import sample
import string
import time
import re
import os

# gensim
from gensim.models import Word2Vec

# sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

def tokenise_char(s, remplace_list):
    for tup in remplace_list:
        s = s.replace(tup[0],tup[1])
    return s

def mknewdir(dir_):
    if os.path.exists(dir_):
        print('Directory already exists!')
    else:
        os.mkdir(dir_)
        print('Directory created!')

def get_colordict(chars):
    colordict = dict()
    for k in chars:
        if re.compile('\d').match(k):
            colordict[k] = [1.0,.55,.0]
            continue
        elif re.compile('\s').match(k):
            colordict[k] = [.16,.64,.16]
            continue
        elif re.compile('\w').match(k):
            colordict[k] = [.12,.56,1.0]
            continue
        else:
            colordict[k] = [.9,.0,.0]
            continue
    return colordict

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
    t = timer()
    emb_dict = {c:w2v_model.wv[c] for c in w2v_model.wv.vocab.keys()}
    n = multiprocessing.cpu_count()
    with multiprocessing.Pool(n) as p:
        args = zip(window, [emb_dict]*len(window))
        emb = p.starmap(stackemb, args)
        p.close()
    emb = np.asarray(emb, dtype=float)
    print('Time for embedding: %.2f sec' % (timer() - t))
    return emb

def embed_sent(sent, w2v_model):
    emb_dict = {c:w2v_model.wv[c] for c in w2v_model.wv.vocab.keys()}
    emb = [emb_dict[c] for c in sent]
    emb = np.asarray(emb)
    emb = emb.reshape(1,emb.shape[0],emb.shape[1])
    return emb

def preprocess(biblia):
    # 1. Join in one string
    biblia = '\n'.join(biblia)
    # 5. Lowercase
    biblia = biblia.lower()
    # 6. Split chars
    biblia = list(biblia)
    # 6. Replace chars
    remplace_dict = {
        'á':['a','tilde'],
        'é':['e','tilde'],
        'í':['i','tilde'],
        'ó':['o','tilde'],
        'ú':['u','tilde'],
        'ü':['u','tilde'],
        '\n':['end_line'],
        ' ':['white_space'],
        }
    biblia = [b if not b in remplace_dict else remplace_dict[b] for b in biblia]
    # 8. Flat list
    biblia = [item for sublist in biblia for item in sublist]
    # 9. Replace non-valid chars
    valid_chars = string.ascii_lowercase + string.digits + '.,:;?!()-¡¿ñ'
    valid_chars = list(valid_chars)
    valid_chars.extend(['tilde','dieresis','white_space'])
    biblia = ['<unknown>' if c not in valid_chars else c for c in biblia]
    return biblia

def preprocess_all(biblia):
    # 1. Strip text
    biblia = [b.strip() for b in biblia if b.strip()]
    # 1. Join in one string
    biblia = '\n'.join(biblia)
    # 2. Clean tabulation and symbols
    biblia = re.sub('\s+',' ', biblia)
    biblia = re.sub('([?!.])\s([A-Z]\w+|[1-9]+\s)',r'\1\n', biblia)
    biblia = re.sub('\n\s(([A-Z]\w+|[1-9]+\s))',r'\n\1', biblia)
    # 5. Lowercase
    biblia = biblia.lower()
    # 6. Split chars
    biblia = list(biblia)
    # 6. Replace chars
    remplace_dict = {
        'á':['a','tilde'],
        'é':['e','tilde'],
        'í':['i','tilde'],
        'ó':['o','tilde'],
        'ú':['u','tilde'],
        'ü':['u','tilde'],
        '\n':['end_line'],
        ' ':['white_space'],
        }
    biblia = [b if not b in remplace_dict else remplace_dict[b] for b in biblia]
    # 8. Flat list
    biblia = [item for sublist in biblia for item in sublist]
    # 9. Replace non-valid chars
    valid_chars = string.ascii_lowercase + string.digits + '.,:;?!()-¡¿ñ'
    valid_chars = list(valid_chars)
    valid_chars.extend(['tilde','dieresis','white_space'])
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


def plot_embedding(w2v_model, mode='pca'):
    chars = list(w2v_model.wv.vocab)
    colordict = get_colordict(chars)
    char_embedding = np.vstack([w2v_model.wv[w] for w in w2v_model.wv.vocab.keys()])

    if mode =='tsne':
        tsne = TSNE(
            n_iter=100000,
            random_state=0, 
            init='pca',
            angle=.95,
            perplexity=20,
            method='exact',
                )
        embedding_2d = tsne.fit_transform(char_embedding)
    if mode == 'pca':
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(char_embedding)

    plt.figure()
    plt.scatter(embedding_2d[:,0],[embedding_2d[:,1]], alpha=0.0)
    for c,loc in zip(chars, embedding_2d):
        plt.text(loc[0],loc[1], 
            c,
            ha="center",
            va="center",
            size=9,
            bbox=dict(
                boxstyle="round",
                ec=[.3*x for x in colordict[c]],
                alpha=0.6,
                fc=colordict[c]
                )
            )
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/AA_%s_%s_epochs_%s_window_%s_size.pdf' % (
        mode,
        w2v_model.epochs,
        w2v_model.window,
        w2v_model.vector_size
    ))
    plt.close()
    print('Plotted!')  

def generate_versiculo(sentence, w2v_model, model, length=40):
    if type(sentence)!=str:
        raise ValueError("'bait' must be str")
    if len(sentence) != 11:
        raise ValueError("len of sentence must be 11")
    toreturn = sentence
    bait = [sentence]
    chars = list(w2v_model.wv.vocab)

    remplace_dict = {
        '<unknown>':'|',
        'tilde':'`',
        'end_line':'\n',
        'white_space':' ',
        'dieresis':'~'
        }

    while len(toreturn) < length:
        # Deprocess
        bait_pre = preprocess(bait)
        e = embed_sent(bait_pre, w2v_model)
        p = model.predict(e)
        nextchar = np.random.choice(chars, size=1, p = p.ravel())[0]

        # Get nextchar
        # print('Nextchar:', nextchar)
        if len(nextchar) > 1:
            nextchar = remplace_dict[nextchar]
        bait = [bait[0][1:]+nextchar]

        # Update sentence
        toreturn += nextchar
        # print(toreturn)
    return toreturn 

#----------------------------
# PIPELINE
#----------------------------
filename = 'biblia_no_encabezados.txt'
filepath = os.path.join('./data/Biblia/procesado_1', filename)
with open(filepath,'r', encoding='latin1') as handle:
    biblia_raw = handle.readlines()

biblia = preprocess_all(biblia_raw)
# Verify chars
chars = set(biblia)
# Sliding window: len('y dios dijo.....')
biblia_w = list(window(biblia, n=15))

# w2v
seed = 0
w2v_model = Word2Vec(
    size=8,
    window=7,
    min_count=1,
    workers=multiprocessing.cpu_count(),
    seed=seed,
    )

# build the vocabulary
t0 = timer()
w2v_model.build_vocab(biblia_w)
print('Vocabulary generated. Time: %.3f sec' % (timer() - t0))

# train w2v_model
t0 = timer()
w2v_model.train(biblia_w, total_examples=w2v_model.corpus_count, epochs=1000)
print('Model trained. Time: %.3f sec' % (timer() - t0))
modelname = 'data/models/AA_char_w2v_%s_iter_%s_seed_%s_window_%s_size.w2v' % (
    w2v_model.epochs,
    seed,
    w2v_model.window,
    w2v_model.vector_size,
    )
w2v_model.save(modelname)
# w2v_model = Word2Vec.load('data/models/AA_char_w2v_5_iter_0_seed_7_window_12_size.w2v')

# Plot
plot_embedding(w2v_model, mode='tsne')
plot_embedding(w2v_model, mode='pca')

# Embed text
# Sliding window: len('y dios dijo.....')
biblia_w = list(window(biblia, n=12))
embedding = embed_window_parallel(biblia_w, w2v_model)

# Define vars
X = embedding[:,:11,:]
onehot = get_onehot_enc(w2v_model)
y = [c[-1] for c in biblia_w]
y = [onehot[c] for c in y]
y = np.asarray(y).reshape(len(biblia_w),len(chars))

# LSTM
NAME = 'biblia_%s' % (re.sub('[^\w]+','_',time.asctime()))
mknewdir('logs')
tb = TensorBoard(log_dir='logs/%s' % NAME)

rms = RMSprop()
metrics = ['accuracy','categorical_crossentropy']
compile_args = {'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy']}

model = Sequential()
model.add(Bidirectional(LSTM(10, input_shape=X.shape[1:], dropout=.3)))
model.add(Dense(50, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


es = EarlyStopping(monitor='val_acc', patience=5)
args = {'shuffle':True,'callbacks':[es,tb]}
t0 = timer()
for iteration in range(2000):
    print('iter %d' % iteration)
    s = [generate_versiculo('y dios dijo', w2v_model, model) for x in range(3)]
    print('\n%s\n\n' % '\n'.join(s))
    model.fit(X,y, batch_size=10000, epochs = 1, validation_split=.3, shuffle=True, verbose=0, callbacks=[es,tb])
print('Time: %.3f for training' % (timer() - t0))
# Visualise: http://localhost:6006/
modelname = 'data/models/%s.keras' % NAME
model.save(modelname)





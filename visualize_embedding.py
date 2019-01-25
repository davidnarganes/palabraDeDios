import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import string
import gensim

# model_file =  r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\model_sent_len_15_syllables_lower_tildes_punctuation'
model_file =  r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\w2c_model_sent_len_15_syllables_lower_tildes_punctuation'
model = gensim.models.FastText.load(model_file)

vectors=np.vstack([model[word] for word in model.wv.vocab.keys()])

pca = PCA(n_components=2)
bi_comp = pca.fit_transform(vectors)

inds = (0,200)
# create a scatter plot of the projection
plt.scatter(bi_comp[inds[0]:inds[1], 0], bi_comp[inds[0]:inds[1], 1])
words = list(model.wv.vocab)
words = words[inds[0]:inds[1]]
inds_punct = [idx for (idx,w) in enumerate(words) if w in string.punctuation]
plt.scatter(bi_comp[inds_punct,0], bi_comp[inds_punct,1], c='r')


for i, word in enumerate([w for w in words if w in string.punctuation]):
    plt.text(bi_comp[i, 0], bi_comp[i, 1], word)
	# plt.annotate(word, xy=(bi_comp[i, 0], bi_comp[i, 1]))

for i, word in enumerate(words):
	plt.annotate(word, xy=(bi_comp[i, 0], bi_comp[i, 1]))
plt.axis('off')
plt.show()
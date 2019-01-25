import gensim
import multiprocessing

text_file_name = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\sent_len_15_syllables_lower_tildes_punctuation.txt'
# output = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\ft_model_sent_len_15_syllables_lower_tildes_punctuation'
# output = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\w2v_model_sent_len_15_syllables_lower_tildes_punctuation'

# text_file_name = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\sent_len_15_syllables_lower_tildes.txt'
# output = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\w2v_model_sent_len_15_syllables_lower_tildes'
output = r'D:\datos\datos_generacion_texto\vectores_palabras\siglo_oro\ft_model_sent_len_15_syllables_lower_tildes_size_30'

# sentences = gensim.models.word2vec.LineSentence(text_file_name)
with open(text_file_name, 'r',  encoding='utf-8') as data_file:
    sentences = data_file.readlines()

sentences = [x.strip().split() for x in sentences]
# sentences = [gensim.utils.any2unicode(x.strip()).split() for x in sentences]
print('sentences loaded')

model = gensim.models.FastText(sentences, size=30, window=15, min_count=0, workers=multiprocessing.cpu_count()-1)
# model = gensim.models.Word2Vec(sentences, size=20, window=5, min_count=0, workers=multiprocessing.cpu_count()-1)
print('model generated')

# build the vocabulary
model.build_vocab(sentences, update=True)
print('vocabulary created')

# train model
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
print('training finished')
print(model)

model.save(output)

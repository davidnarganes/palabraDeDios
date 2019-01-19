from gensim.models import Word2Vec

with open('data/Biblia/procesado_1/biblia_procesada_1.txt','r', encoding='utf-8') as handle:
    biblia_raw = handle.readlines()

# Define some vars to replace
white_space = '<ws>'
unknown = '<unk>'
pad = '<pad>'

model = Word2Vec(size=300, window=10, min_count=1, workers=-1)
model.build_vocab(char_sent_biblia)
model.train(char_sent_biblia, total_examples=model.corpus_count, epochs=10)
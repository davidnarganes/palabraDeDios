from gensim.models import Word2Vec

white_space = '<\w>'
model = Word2Vec(size=300, window=10, min_count=1, workers=-1)
model.build_vocab(char_sent_biblia)
model.train(char_sent_biblia, total_examples=model.corpus_count, epochs=10)
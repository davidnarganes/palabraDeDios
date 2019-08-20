import sys
import os

sys.path.append(os.path.join("..","utils"))
sys.path.append(os.path.join("..","preprocess"))

import json
from hashlib import sha256
from gensim.models import Word2Vec

from keras.callbacks import LambdaCallback, EarlyStopping

from utils import *
from model_utils import *
from build_models import *
from preprocess import preprocess_func

def onEpochEnd(epoch, _):

    model.reset_states() # clear the hidden states before a new epoch
    saveModelWeights(epoch+1, model, save_model_dir)
    loadModelWeights(epoch+1, prediction_model, save_model_dir)
    for idx in range(2):
        sample_text = sample_biblia[0:window_size-1]
        sample = generateNextElem(sample_text, word2idx, prediction_model)
        print('%s... -> %s' % (idx, sample))

    print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))
  
w2v_file = "model_size_16_window_10_min_count_0_workers_8_seed_19_iter_5_.w2v"
in_filepath = os.path.join("..","..","data","Biblia","AA_preprocesado","biblia_preprocessed.txt")
log_directory = os.path.join("..","..","data","models","log")
w2v_modelpath = os.path.join("..","..","data","models","w2v","model_size_16_window_10_min_count_1_workers_4_seed_123_iter_100_.w2v")
save_model_dir = os.path.join("..","..","data","models","keras")
logpath = r'D:\datos\datos_generacion_texto\Biblia\logs'
save_model_dir = r'D:\datos\datos_generacion_texto\Biblia\char_models'

w2v_model = Word2Vec.load(w2v_modelpath)
embedding_vectors = np.vstack([w2v_model[word] for word in w2v_model.wv.vocab.keys()])
words = list(w2v_model.wv.vocab)
word2idx = {word:idx for idx, word in enumerate(words)}

biblia_preprocessed = read_bible(in_filepath)
biblia_inds = np.array([word2idx[c] for c in biblia_preprocessed])
print('text loaded')

window_size = 100 # number of characters for prediction
overlap = 1 # overlaping between sentences
batch_size = 40 # batch size
num_epochs = 40 # number of epochs
save_freq = 1
hidden_units = [512, 512, 512]
dropout = 0.5

stride = window_size-overlap
chunk_num = int(np.floor((len(biblia_inds)-window_size)/stride+1))
steps_epoch = chunk_num // batch_size

model = multiCuDNNLSTM(embedding_vectors, window_size-1, batch_size, 
                       hidden_units=hidden_units, dropout=dropout)

prediction_model = multiCuDNNLSTM(embedding_vectors, window_size-1, 1, 
                                  hidden_units=hidden_units, dropout=dropout)

config = model.get_config()
sha_code = sha256(str(config).encode()).hexdigest()
save_model_dir = os.path.join(save_model_dir, sha_code)
mknewdir(save_model_dir)

modelpath = os.path.join(save_model_dir, "%s.keras" % sha_code)
config_filepath = modelpath.replace(".keras",".json")
with open(config_filepath, "w") as outfile:
    json.dump(config, outfile, indent=3)

# Calling TensorBard produces an obscure error with the optimizer. Only with this script
callbacks = [LambdaCallback(on_epoch_end=onEpochEnd), \
            EarlyStopping(patience=10, monitor='loss')]

sample_biblia = 'entonces dijo dios hagamos al hombre a nuestra imagen conforme a nuestra semejanza y señoree en los peces del mar en las aves de los cielos en las bestias en toda la tierra y en todo animal que se arrastra sobre la tierra y creó dios al hombre a su imagen a imagen de dios lo creó varón y hembra los creó y los bendijo'
verse_generator = statefulBatchGenerator(biblia_inds, window_size, overlap, batch_size)
model.fit_generator(verse_generator,
                    steps_per_epoch=steps_epoch,
                    epochs=num_epochs,
                    callbacks=callbacks)

# for epoch in range(num_epochs):
#     print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
#     losses, accs = [], []
#     for i, (X, Y) in enumerate(batchGenerator(biblia_inds, window_size, overlap, batch_size)):
#         print(i)
#         loss, acc = model.train_on_batch(X,Y)
#         print(''.join(idx2word(int(idx), word2idx) for idx in X[0]))
#         print('Batch {}: loss = {:.4f}, acc = {:.5f}'.format(i + 1, loss, acc))
        
#         losses.append(loss)
#         accs.append(acc)

    # if (epoch + 1) % save_freq == 0:
    #     saveModelWeights(epoch+1, model, save_model_dir)
    #     loadModelWeights(epoch+1, prediction_model, save_model_dir)
    #     for idx in range(2):
    #         sample_text = sample_biblia[0:window_size-1]
    #         sample = generateNextElem(sample_text, word2idx, prediction_model)
    #         print('%s... -> %s' % (idx, sample))

    #     print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))
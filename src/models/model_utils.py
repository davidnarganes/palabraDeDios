import sys
import os

sys.path.append(os.path.join("..","preprocess"))
sys.path.append(os.path.join("..","utils"))

from utils import *
from model_utils import *

from keras.models import load_model
from preprocess import preprocess_func
from build_models import *

def idx2word(idx, word2idx):
  keys = list(word2idx.keys())
  word = keys[idx]
  if word == '<white_space>':
    word = ' '
  elif word == '<tilde>':
    word = '´'
  return word

def batchGenerator(text_inds, window_size, overlap, batch_size):
    '''
    iterator for batch generation
    '''
    while True:
        stride = window_size-overlap
        chunk_num = int(np.floor((len(text_inds)-window_size)/stride+1))
        batch_num = len(text_inds) // (chunk_num*batch_size)

        row_counter = 0
        for batch_idx in range(batch_num):
            X = np.zeros((batch_size, window_size-1))
            Y = np.zeros((batch_size, 1))

            for row in range(batch_size):
                origin = row_counter*(window_size - overlap)
                X[row, :] = text_inds[origin:origin+window_size-1]
                Y[row, :] = text_inds[origin+window_size-1]
                row_counter += 1

            yield X, Y

def saveModelWeights(epoch, model, save_model_dir):
    mknewdir(save_model_dir)
    model.save_weights(os.path.join(save_model_dir, 'weights.{}.h5'.format(epoch)))

def loadModelWeights(epoch, model, save_model_dir):
    model.load_weights(os.path.join(save_model_dir, 'weights.{}.h5'.format(epoch)))

def samplePrediction(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)

    return np.argmax(probs)

def generateNextElem(text, word2idx, prediction_model, num_generated=200, T=0.5):
    sentence = [word2idx[word] for word in preprocess_func([text])]
    word_idxs = np.array(sentence, ndmin=2)
    for i in range(num_generated):
        prediction = prediction_model.predict(word_idxs)
        idx = samplePrediction(prediction[-1], temperature=T)
        sentence.append(idx)
        word_idxs = np.array(sentence[i+1:], ndmin=2)

    return ''.join(idx2word(idx, word2idx) for idx in sentence)

def onEpochEnd(epoch, _):
    print('\nGenerating text after epoch: %d' % epoch)
        
    saveModelWeights(epoch+1, model, save_model_dir)
    loadModelWeights(epoch+1, prediction_model, save_model_dir)
    for idx in range(2):
        sample_text = sample_biblia[0:window_size-1]
        sample = generateNextElem(sample_text, word2idx, prediction_model)
        print('%s... -> %s' % (idx, sample))

    print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))
  

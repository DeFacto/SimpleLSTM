from __future__ import print_function
import collections
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM, Multiply, Merge
from keras.optimizers import Adam, Adagrad
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling1D
import numpy as np
import argparse
import gensim
import json
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pickle


class Metrics(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
 
 def on_epoch_end(self, epoch, logs={}):
  val_predict = (np.asarray(parallel_model.predict([val_sent_e, val_claim_e]))).round()
  val_targ = val_y
  _val_f1 = f1_score(val_targ, val_predict)
  _val_recall = recall_score(val_targ, val_predict)
  _val_precision = precision_score(val_targ, val_predict)
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  print (' — val_f1: %f — val_precision: %f — val_recall %f' %( _val_f1, _val_precision, _val_recall))
  return
 
metrics = Metrics()



#model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)

model = dict()

with open('../../data/fever/support/label', 'r') as f:
    labels_org = f.read()

with open('../../data/fever/support/sentence', 'r') as f:
    sentences = f.read()

with open('../../data/fever/support/claim', 'r') as f:
    claims = f.read()

from string import punctuation


all_text = ''.join([c for c in sentences if c not in punctuation])
sentences = all_text.split('\n')

all_text = ''.join([c for c in claims if c not in punctuation])
claims = all_text.split('\n')

all_text = ' '.join(claims)
all_text += ' '.join(sentences)
words = all_text.lower().split()

# changing here
words = list(set(words))
vocab_to_int = dict()

for i in range(len(words)):
    vocab_to_int.update({words[i]: i})
# from collections import Counter
# counts = Counter(words)
# vocab = sorted(counts, key=counts.get, reverse=True)
# vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

sent_ints = []
for each in sentences:
    each = each.lower()
    sent_ints.append([vocab_to_int[word] for word in each.split()])

claim_ints = []
for each in claims:
    each = each.lower()
    claim_ints.append([vocab_to_int[word] for word in each.split()])

labels = np.array([1 if l == "Positive" else 0 for l in labels_org.split()])

from collections import Counter

claim_lens = Counter([len(x) for x in claim_ints])
sent_lens = Counter([len(x) for x in sent_ints])
print("Zero-length sentences: {}".format(sent_lens[0]))
print("Maximum sentence length: {}".format(max(sent_lens)))

print("Zero-length claims: {}".format(claim_lens[0]))
print("Maximum claim length: {}".format(max(claim_lens)))

# Filter out that review with 0 length
#claim_ints = [r for r in claim_ints if len(r) > 0]
#sent_ints = [r[0:500] for r in sent_ints if len(r) > 0]

tc = []
ts = []
tl = []

for i in range(len(labels)):
 if len(claim_ints[i])*len(sent_ints[i]) > 0:
  tc.append(claim_ints[i])
  ts.append(sent_ints[i])
  tl.append(labels[i])

claim_ints = np.array(tc)
sent_ints = np.array(ts)
labels = np.array(tl)


from collections import Counter

claim_lens = Counter([len(x) for x in claim_ints])
print("Zero-length claims: {}".format(claim_lens[0]))
print("Maximum claim length: {}".format(max(claim_lens)))

sent_lens = Counter([len(x) for x in sent_ints])
print("Zero-length sents: {}".format(sent_lens[0]))
print("Maximum sent length: {}".format(max(sent_lens)))

mx_sent = max(sent_lens)
mx_claim = max(claim_lens)

claim_seq_len = mx_claim
sent_seq_len = mx_sent
claim_features = np.zeros((len(claim_ints), claim_seq_len), dtype=int)
sent_features = np.zeros((len(sent_ints), sent_seq_len), dtype=int)

for i, row in enumerate(claim_ints):
    claim_features[i, -len(row):] = np.array(row)[:claim_seq_len]

for i, row in enumerate(sent_ints):
    sent_features[i, -len(row):] = np.array(row)[:sent_seq_len]



split_frac = 0.9

split_index = int(split_frac * len(claim_features))

train_claim, val_claim = claim_features[:split_index], claim_features[split_index:]
train_sent, val_sent = sent_features[:split_index], sent_features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 1
split_index = int(split_frac * len(val_claim))

val_claim, test_claim = val_claim[:split_index], val_claim[split_index:]
val_sent, test_sent = val_sent[:split_index], val_sent[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]


n_words = len(vocab_to_int) + 1  # Add 1 for 0 added to vocab

embed_size = 300

w2v_embed = np.ndarray([n_words, embed_size])

for i in range(n_words - 1):
    if words[i] not in model:
        w2v_embed[vocab_to_int[words[i]]] = np.array([0] * embed_size)
    else:
        w2v_embed[vocab_to_int[words[i]]] = model[words[i]]

#with open('fever_word2vec_dic.pkl','wb') as f:
# pickle.dump(w2v_embed,f)

with open('../fever_word2vec_dic.pkl','rb') as f:
 w2v_embed = pickle.load(f)


import random

idx = random.sample(range(len(train_claim)), len(train_claim))

train_claim_s = []
train_sent_s = []
train_y_s = []

for i in idx:
    train_claim_s.append(train_claim[i])
    train_sent_s.append(train_sent[i])
    train_y_s.append(train_y[i])

train_claim = np.array(train_claim_s)
train_sent = np.array(train_sent_s)
train_y = np.array(train_y_s)
test_claim = np.array(test_claim)
test_sent = np.array(test_sent)
test_y = np.array(test_y)


train_claim_e = np.ndarray((len(train_claim), mx_claim, embed_size))
train_sent_e = np.ndarray((len(train_sent), mx_sent, embed_size))

for i in range(len(train_claim)):
    for j in range(mx_claim):
        train_claim_e[i][j][:] = w2v_embed[train_claim[i][j]]

for i in range(len(train_sent)):
    for j in range(mx_sent):
        train_sent_e[i][j][:] = w2v_embed[train_sent[i][j]]

val_claim_e = np.ndarray((len(val_claim), mx_claim, embed_size))
val_sent_e = np.ndarray((len(val_sent), mx_sent, embed_size))

for i in range(len(val_claim)):
    for j in range(mx_claim):
        val_claim_e[i][j][:] = w2v_embed[val_claim[i][j]]

for i in range(len(val_sent)):
    for j in range(mx_sent):
        val_sent_e[i][j][:] = w2v_embed[val_sent[i][j]]



hidden_size = 256
use_dropout = True
vocabulary = n_words

embedding_layer = Embedding(input_dim=vocabulary, output_dim=300)

lstm_out = 150

model1 = Sequential()
#model1.add(embedding_layer)
#model1.add(Embedding(vocabulary, embed_size, input_length=mx_sent))
model1.add(LSTM(lstm_out, return_sequences=False, input_shape=(mx_sent, embed_size)))
#model1.add(LSTM(embed_size, return_sequences=True))
#model1.add(GlobalAveragePooling1D())
#model1.add(TimeDistributed(Dense(1)))
#model1.add(LSTM(embed_size, return_sequences=False))
if use_dropout:
    model1.add(Dropout(0.3))
model1.add(Dense(lstm_out, activation='sigmoid', name='out1'))

model2 = Sequential()
#model2.add(embedding_layer)
#model2.add(Embedding(vocabulary, embed_size, input_length=mx_claim))
#model2.add(LSTM(embed_size, return_sequences=True))
model2.add(LSTM(lstm_out, return_sequences=False, input_shape=(mx_claim, embed_size)))
#model2.add(LSTM(embed_size, return_sequences=True))
#model2.add(LSTM(embed_size, return_sequences = False))
#model2.add(GlobalAveragePooling1D())
#model2.add(TimeDistributed(Dense(1)))  
if use_dropout:
    model2.add(Dropout(0.3))
model2.add(Dense(lstm_out, activation='sigmoid', name='out2'))

model = Sequential()
model.add(Merge([model1, model2], mode='mul'))
#model.add(Dense(600))
#model.add(Dense(600))
#model.add(Dense(300))
#model.add(Dense(100))
#model.add(Merge([model1, model2], mode='cos', dot_axes=1))
model.add(Dense(1, activation = 'sigmoid'))
# model = Multiply()([model1.get_layer('out1').output,model2.get_layer('out2').output])

# model.add(TimeDistributed(Dense(vocabulary)))
# model.add(Activation('softmax'))

#optimizer = Adam()
# model1.compile(loss='mean_squared_error', optimizer='adam')
# parallel_model = multi_gpu_model(model, gpus=2)
parallel_model = model
parallel_model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.001) , metrics=['acc'])
#parallel_model.compile(loss='mean_squared_error', optimizer='adam' )

print(model.summary())
print(model1.summary())
print(model2.summary())
# checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 100
plot_model(parallel_model, to_file='model.png')
parallel_model.fit(x=[train_sent_e, train_claim_e], y=train_y, batch_size=64, epochs=num_epochs,
               validation_data=([val_sent_e,val_claim_e],val_y), callbacks = [metrics])



#print(parallel_model.predict([val_sent_e[:100], val_claim_e[:100]]), val_y[:100])
#print(parallel_model.evaluate([val_sent_e,val_claim_e],val_y))
#parallel_model.save("final_model.hdf5")

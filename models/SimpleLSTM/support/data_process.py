import json
import pickle
import re
import numpy as np
from string import punctuation

with open('../../../data/fever/support/main.json','r') as f:
    total = json.load(f)


def filterfun(word):
    if word in punctuation:
        return False
    else:
        return True


claims = []
sentences = []
for i in range(len(total)):
    claims += re.sub(r'[^\w\s]','',total[i]['claim'].lower()).split()
    claims.append('\n')
    sentences += re.sub(r'[^\w\s]','',total[i]['sentence'].lower()).split()
    sentences.append('\n')

#filtered = filter(filterfun,claims)
#claims = []
#for i in filtered:
#    claims.append(i)

#filtered = filter(filterfun,sentences)
#sentences = []
#for i in filtered:
#    sentences.append(i)

claims = ' '.join(claims)
claims = ''.join(claims).split('\n')
temp = ' '.join(claims).split()
sentences = ' '.join(sentences)
sentences = ''.join(sentences).split('\n')
temp += ' '.join(sentences).split()
words = list(set(temp))
vocab_to_int = dict()


for i in range(len(words)):
    vocab_to_int.update({words[i]: i})

sent_ints = []
claim_ints = []
labels = []
mxclaim = 0
mxsent = 0
for i in range(len(total)):
    s = sentences[i]
    c = claims[i]
    if len(s) > mxsent:
        mxsent = len(s)
    if len(c) > mxclaim:
        mxclaim = len(c)
    sent_ints.append([vocab_to_int[word] for word in s.split()])
    claim_ints.append([vocab_to_int[word] for word in c.split()])
    labels.append(total[i]['lablel'])


claim_features = np.zeros((len(claim_ints), mxclaim), dtype=int)
sent_features = np.zeros((len(sent_ints), mxsent), dtype=int)

for i, row in enumerate(claim_ints):
    claim_features[i, -len(row):] = np.array(row)[:mxclaim]

for i, row in enumerate(sent_ints):
    sent_features[i, -len(row):] = np.array(row)[:mxsent]

with open('../../../data/fever/support/train.pkl','rb') as f:
    train = pickle.load(f)
with open('../../../data/fever/support/val.pkl','rb') as f:
    val = pickle.load(f)

train_claim = []
train_sent = []
val_claim = []
val_sent = []

for i in val['lablel'].keys():
    val_claim.append(claim_features[i])
    val_sent.append(sent_features[i])

for i in train['lablel'].keys():
    train_claim.append(claim_features[i])
    train_sent.append(sent_features[i])


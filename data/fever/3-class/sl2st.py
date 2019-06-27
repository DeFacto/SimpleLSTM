import pandas as pd
import numpy as np

claim = pd.read_table('claim',header=None);
label = pd.read_table('label',header=None);
sentence = pd.read_table('sentence',header=None);

l1 = []
l2 = []
l3 = []
l4 = []
l5 = []

for i in range(len(claim)):
    l1.append(i)
    l2.append(sentence.iloc[i][0])
    l3.append(claim.iloc[i][0])
    l4.append(i)
    if label.iloc[i][0]=='Positive':
        l5.append('agree')
    else:
        l5.append('unrelated')

claim['1'] = l1
claim['2'] = l2
claim['3'] = l3
claim['4'] = l4
claim['5'] = l5

train = claim.iloc[:int(len(claim)*.7)]
test = claim.iloc[int(len(claim)*.7):]

#claim.to_csv('test',header =None,index = False)
body_tr = train[['1','2']]
stance_tr  = train[['3','4','5']]

body_te = test[['1','2']]
stance_te  = test[['3','4','5']]

body_tr.to_csv('body_train.csv',header=None,index=False)
stance_tr.to_csv('stance_train.csv',header=None,index=False)

body_te.to_csv('body_test.csv',header=None,index=False)
stance_te.to_csv('stance_test.csv',header=None,index=False)


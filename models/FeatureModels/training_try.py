import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from feature import Features
import io
import codecs
import pandas
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import spacy
nlp = spacy.load("en")
from nltk.corpus import stopwords
import nltk
def train_tfidf():

	return None

#need to define a proper structure for these functions

def train_baseline():
	with open('iswctotal.json', 'r') as f:  # load the sentences
		data = json.load(f)

	X = []
	for i in range(len(data)):
		X.append(data[i]['features'])
	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]

	#clf = RandomForestClassifier(random_state=0)

	clf = joblib.load('RF.pkl')

	prediction = clf.predict(X)

	for i in range(len(prediction)):
		data[i]['output'] = prediction[i]

	with open('predicted.json', 'w') as f:
		json.dump(data, f)

	return None #not sure about returning


#this function trains using fever
#need to do some cleaning

#return value not clear

'''
def training(config='baseline'):

	reads the dataset (training) file, extract features and train the fact-checking model
	:return: the model's performance


	data = np.array(fever())  # gets features from fever dataset

	X, x = sklearn.model_selection.train_test_split(data, test_size=0.3)  #test/train ratio

	Y = X[:, [X.shape[1] - 1]]
	y = x[:, [x.shape[1] - 1]]
	X = X[:, 0:X.shape[1] - 1]
	x = x[:, 0:x.shape[1] - 1]

	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]

	temp = sklearn.preprocessing.normalize(x[:, [3]], axis=0)
	for i in range(len(temp)):
		x[i][3] = temp[i]

	clf = RandomForestClassifier(random_state=0)
	clf.fit(X, Y)

	joblib.dump(clf, 'RF.pkl') #change name before running



	try:

		if config == 'baseline':
			train_baseline()
		elif config == 'tfidf':
			train_tfidf()
		else:
			# here we keep adding separated functions
			raise Exception('not supported: ' + config)

	except:
		raise
'''
def training_RF(dt,label, config='baseline'):
	'''
#X = sklearn.preprocessing.scale(data)
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=0)

	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]

	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	clf = RandomForestClassifier(n_estimators=25, random_state=40,oob_score=True)
	clf.fit(X_train, np.ravel(y_train))
	predict_labels=clf.predict(X_test)
	print(predict_labels)
	print("RF accuracy: "+ str(clf.score(X_test,np.ravel(y_test))))
	'''
	print(dt[0])
	data=np.array(dt)
	print(data[0])
	X, x , Y, y= sklearn.model_selection.train_test_split(data,label, test_size=0.3)  #test/train ratio

	#Y = X[:, [X.shape[1] - 1]]
	#y = x[:, [x.shape[1] - 1]]
	X = X[:, 0:X.shape[1] ]
	x = x[:, 0:x.shape[1] ]
	temp = sklearn.preprocessing.normalize(X[:, [3]], axis=0)
	for i in range(len(temp)):
		X[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(x[:, [3]], axis=0)
	for i in range(len(temp)):
		x[i][3] = temp[i]
	#print(x)
	#print(X[0])
	clf = RandomForestClassifier(random_state=0,n_estimators=100)
	clf.fit(X, np.ravel(Y))
	print("RF classsifier: ", clf.score(x, np.ravel(y)))
	scores = cross_val_score(clf, X, Y, cv=6)
	print("Cross-validated scores:", scores)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def training_svm2(Data, label):
	Data=np.array(Data)
	label=np.array(label)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3, random_state=0)  #test/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]
	lsvm = LinearSVC()
	lsvm.fit(X_train,np.ravel(y_train))
	score=lsvm.score(X_test,np.ravel(y_test))
	print("SVM accuracy: "+str(score))

def training_svm(Data,label):
	#X=sklearn.preprocessing.normalize(Data, norm='l2')
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3, random_state=0)  #test/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]

	clf = RandomForestClassifier(random_state=0,n_estimators=25)
	clf.fit(X_train, np.ravel(y_train))
	print("RF classsifier: ", clf.score(X_test, np.ravel(y_test)))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	scores = cross_val_score(clf, X_test, y_test, cv=6)
	print("Cross-validated scores:", scores)

	plot_learning_curve(clf, "Random Forest", X_train, y_train, cv=5, n_jobs=-1)

	lsvm = LinearSVC()
	lsvm.fit(X_train,np.ravel(y_train))
	score=lsvm.score(X_test,np.ravel(y_test))
	print("SVM accuracy: "+str(score))

	scores = cross_val_score(lsvm, X_train, y_train, cv=6)
	print("Cross-validated scores:", scores)

	plot_learning_curve(lsvm, "Linear SVM", X_train, y_train, cv=5, n_jobs=-1)
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)

	mlp=MLPClassifier(hidden_layer_sizes=(75,), activation='relu', batch_size='auto', learning_rate_init=0.001, max_iter=2000, shuffle=False, random_state=42,  validation_fraction=0.1, n_iter_no_change=10)
	mlp.fit(X_train, y_train)
	print("MLP accuracy "+str(mlp.score(X_test,np.ravel(y_test))))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	plot_learning_curve(mlp, "Neural Net\n75 Hidden Layers | Activation function-RelU", X_train, y_train, cv=5, n_jobs=-1)
	#parameters = {'kernel':('linear', 'rbf'), 'C':[0.10, 0.1, 10, 100, 1000], 'probability':[True, False], 'coef0':[0.0,0.05,0.1]}
	#svc = svm.SVC(gamma="scale")
	#clf = GridSearchCV(svc, parameters, cv=6, n_jobs=-1)
	#clf.fit(X_train,np.ravel(y_train))
	#print(clf.best_params_)
	#print(clf.best_score_ )
	#print(clf.best_estimator_)

	clf=svm.SVC(C=100, kernel='rbf',probability=True, gamma='scale', coef0=0.0)
	clf.fit(X_train,np.ravel(y_train)).score(X_train, np.ravel(y_train))
	print("SVM_rbf"+str(clf.score(X_test,np.ravel(y_test))))
	op=clf.predict(X_test)
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	for i in range(0,len(op)):
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	plot_learning_curve(clf, "RBF SVM", X_train, y_train, cv=5, n_jobs=-1)
	#plt.show()

def train_MLP(Data,label):
	print(Data[0])
	label=np.array(label)
	Data=np.array(Data)
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Data,label, test_size=0.3,random_state=100)  #tet/train ratio
	#y_train=np.array(y_train)
	#y_test=np.array(y_test)
	#y_train=y_train.reshape(-1,1)
	#y_test=y_test.reshape(-1,1)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]
	mlp=MLPClassifier(hidden_layer_sizes=(100), activation='relu', batch_size='auto', learning_rate_init=0.001, max_iter=30000, validation_fraction=0.1, verbose=True, n_iter_no_change=200)
	mlp.fit(X_train, y_train)
	print("MLP accuracy "+str(mlp.score(X_train,np.ravel(y_train))))
	print("MLP accuracy "+str(mlp.score(X_test,np.ravel(y_test))))
	op=mlp.predict(X_test)
	print(accuracy_score(y_train,mlp.predict(X_train)))
	ct=0
	ct0=0
	count00=0
	count01=0
	count10=0
	count11=0
	acc=0
	for i in range(0,len(op)):
		if op[i]==y_test[i]:
			acc+=1
		if y_test[i]==0:
			ct0+=1
			if(op[i]==0):
				count00+=1
			elif(op[i]==1):
				count01+=1
		elif y_test[i]==1:
			ct+=1
			if(op[i]==1):
				count11+=1
			elif(op[i]==0):
				count10+=1
	print("TN, FN, TP, FP", count00,count01,count11,count10)
	print("neg score",float(count00)/ct0)
	print("pos score",float(count11)/ct)
	print("accuracy",float(acc)/len(op))


def training_RF1(X,y):
	#X=sklearn.preprocessing.normalize(X, norm='l1')
	data=np.array(X)
	y=np.array(y)
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42)
	X_train= X_train[:, 0:X_train.shape[1] ]
	X_test = X_test[:, 0:X_test.shape[1] ]
	temp = sklearn.preprocessing.normalize(X_train[:, [3]], axis=0)
	for i in range(len(temp)):
		X_train[i][3] = temp[i]
	temp = sklearn.preprocessing.normalize(X_test[:, [3]], axis=0)
	for i in range(len(temp)):
		X_test[i][3] = temp[i]
	print(X_test[0])
	print(X_train[0])
	clf = RandomForestClassifier(random_state=40)
	clf.fit(X_train, np.ravel(y_train))
	predict_labels=clf.predict(X_test)
	print("RF accuracy: "+ str(clf.score(X_test,y_test)))

if __name__ == "__main__":
	'''
	train_data=[]
	label=[]
	json_data = open('../data/fever/fever.json')
	data = json.load(json_data)
	f=Features()
	c=0
	for d in data:
		train_data.append(f.extract_features(d['body'], d['spo'][0],d['spo'][1], d['spo'][2]))
		label.append(d['label'])
		c+=1
		if c>10:
			break
	#label=random.choices(population=[0,1], weights=[0.25,0.75],k=len(train_data))
	#print(train_data)
	#print(label)
	exit(0)

	read_file=io.open('omar_feat.txt','r',encoding='utf-8')
	split= [line.strip() for line in read_file]
	l1=[]
	l2=[]
	l3=[]
	l4=[]
	for line in split:
    		l1.append(line.split('\t')[:8])
    		l2.append(line.split('\t')[8:])

	for m in l1:
		ab=[]
		for k in m:
			ab.append(float(k))
		l3.append(ab)
	for i in l2:
		ab=[]
		ab.append(float(i[0]))
		l4.append(ab)
	for i in range(0,10):

		print(l1[i])
		print(l3[i])
		print(l2[i])
		print(l4[i])
	'''
	#df1=pandas.read_csv('omar_feat.txt', sep='\t', lineterminator='\n', header=None, names=['C1','2', '3', '4', '5', '6', '7', '8', '9'])
	#print(df1.sample(5))
	#print(df1.describe().to_csv('desc.csv'))
	#exit(0)
	#y = df1['9'].values.reshape(-1,1)
	#df3=df1.drop(columns=['9'])
	#df=df1.values.reshape(-1,9)
	#df3.to_csv("desc.csv")
	#df=np.array(df1)
	#temp = sklearn.preprocessing.normalize(df[:, [3]], axis=0)
	#for i in range(len(temp)):
	#	df[i][3] = temp[i]
	#print(df[0])
	#df.to_csv("desc1.csv")
	#exit(0)
	train_data=[]
	label=[]
	json_data = open('../data/fever/reject/fever_rej.json')
	data = json.load(json_data)
	f=Features()
	f.word2vecModel()
	c=0
	label1=[]

	for d in data:
		c=0
		label1.append(d['label'])
		try:
			#print(d['sentence'] , d['triples'][0][0],d['triples'][0][1], d['triples'][0][2])
			train_data.append(f.extract_features(d['sentence'], d['triples'][0][0],d['triples'][0][1], d['triples'][0][2]))
			#print((f.extract_features(d['sentence'], d['triples'][0][0],d['triples'][0][1], d['triples'][0][2])))
			'''
			if(c<5):
				wlist={}
				stop_words = set(stopwords.words('english'))
				shsh=d['sentence'].lower()#.replace('.','').replace(',','')
				shsh=' '.join(w for w in shsh.split() if w not in stop_words)
				sent_text = nltk.sent_tokenize(shsh)
				for f1 in sent_text:
					doc=nlp(f1)
					print(doc)
					for word in doc:
						wlist[str(word)]=str(list(word.children))
				print(wlist)
			'''
		except:
			continue
		if d['label']==1:
			label.append(0)
		elif d['label']==0:
			label.append(1)


	print(len(train_data))
	print(len(label))
	print(len(label1))
	#training_RF1(train_data, label)#,y)
	#training_svm2(train_data, label)#,y)
	train_MLP(train_data, label)#,y)
	'''
		if(d['label'])!=2 :
				if(d['spo'][2]!='' and c<5):
					#if(d['label']==0 and c<15000 or d['label']==1 ):
						#train_data.append(f.extract_features(d['body'], d['spo'][0],d['spo'][1], d['spo'][2]))
						label.append(d['label'])
						#print(train_data[c])
						#print(d)
						if(d['label']==1):
							print(d)
							print(train_data[c])
						wlist={}
						stop_words = set(stopwords.words('english'))
						shsh=d['body'].lower()#.replace('.','').replace(',','')

						shsh=' '.join(w for w in shsh.split() if w not in stop_words)
						sent_text = nltk.sent_tokenize(shsh)
						for f1 in sent_text:
							doc=nlp(f1)
							print(doc)
							for word in doc:
								wlist[str(word)]=str(list(word.children))
						print(wlist)
						break
						c+=1


	'''
	exit(0)
	count0=0
	count1=0
	for s in label:
		if(s==0):
			count0+=1
		else:
			count1+=1
	#for i in range(0,len(label)):
	#	if(label[i])==1:
	#		print(train_data[i])
	print(count0)
	print(count1)
	print(data[0])
	print(f.here)
	#training_RF(train_data, label)#,y)
	#training_svm(train_data, label)#,y)
	train_MLP(train_data, label)#,y)

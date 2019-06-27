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
def training_RF(X,y,config='baseline'):
	X=sklearn.preprocessing.normalize(X, norm='l1')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	y_train=np.array(y_train)
	y_test=np.array(y_test)
	y_train=y_train.reshape(-1,1)
	y_test=y_test.reshape(-1,1)
	clf = RandomForestClassifier(random_state=40)
	clf.fit(X_train, np.ravel(y_train))
	predict_labels=clf.predict(X_test)
	print("RF accuracy: "+ str(clf.score(X_test,y_test)))

def training_svm(Data,label):
	X=sklearn.preprocessing.normalize(Data, norm='l1')
	X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.25, random_state=42)
	y_train=np.array(y_train)
	y_test=np.array(y_test)
	y_train=y_train.reshape(-1,1)
	y_test=y_test.reshape(-1,1)
	lsvm = LinearSVC()
	lsvm.fit(X_train,np.ravel(y_train))
	score=lsvm.score(X_test,y_test)
	print("SVM accuracy: "+str(score))
	clf=svm.SVC(C=20, kernel='rbf', random_state=42, gamma='auto')
	clf.fit(X_train,np.ravel(y_train)).score(X_train, np.ravel(y_train))
	print("SVM_rbf"+str(clf.score(X_test,np.ravel(y_test))))

def train_MLP(Data,label):
	X=sklearn.preprocessing.normalize(Data, norm='l1')
	X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.25, random_state=40)
	y_train=np.array(y_train)
	y_test=np.array(y_test)
	y_train=y_train.reshape(-1,1)
	y_test=y_test.reshape(-1,1)
	mlp=MLPClassifier(hidden_layer_sizes=(100,), activation='relu', batch_size='auto', learning_rate_init=0.001, max_iter=200, shuffle=True, random_state=42,  validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
	mlp.fit(X_train, np.ravel(y_train))
	print("MLP accuracy "+str(mlp.score(X_test, np.ravel(y_test))))

if __name__ == "__main__":
	print("here")
	train_data=[]
	label=[]
	json_data = open('../data/fever/fever.json')
	data = json.load(json_data)
	print("here")
	f=Features()
	c=0
	for d in data:
		train_data.append(f.extract_features(d['body'], d['spo'][0],d['spo'][1], d['spo'][2]))
		label.append(d['label'])
	#label=random.choices(population=[0,1], weights=[0.25,0.75],k=len(train_data))
	#print(train_data)
	print("HI")
	training_RF(train_data,label)
	training_svm(train_data,label)
	train_MLP(train_data,label)

import pandas as pd 
import json
from core.defacto.model import DeFactoModelNLP
from feature_extraction.feature_core import featureCore
import pickle
import argparse
import numpy as np

class featureExtractor:

	def __init__(self, task):

		self.task = task
		self.feature_cores = featureCore(self.task)


	def extract_features(self, list_of_defactoNlps, model_name):

		print ("start tf idf")
		list_of_defactoNlps = self.feature_cores.get_tf_idf_score(list_of_defactoNlps)
		print ("start WMD wmd_score")
		list_of_defactoNlps = self.feature_cores.get_wmd_score(list_of_defactoNlps)
		print ("start vector space")
		list_of_defactoNlps = self.feature_cores.get_vector_space_score(list_of_defactoNlps)
		print ("saving into file")
		print ("self.task ", self.task)
		pickle.dump(list_of_defactoNlps, open((self.task+model_name), 'wb'))
    

	def load_datafiles(self, dataset_params):
		
		data = dict()
		for p in dataset_params:
			open_data = open(p['EXP_FOLDER'] + p['DATASET'])
			dataframe = pd.read_json(open_data)
			data[str(p['DATASET'][0:-5])] = dataframe # keys are dataset names w/o extension
		            
		return data


	'''
	Split ratio 0.6, 0.2, 0.2
	'''
	def split_dataset(self, dataset):

		train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])

		return train, validate, test


	def create_DefactoModel(self, data):


		for key, value in data.items():
			
			list_of_defactoNlps = []
			# print ("key ", key)
			train, validate, test = self.split_dataset(data[key])
			# save validation and test dataset 
			pickle.dump(validate, open( ("validate_data_" + self.task + key), 'wb'))
			pickle.dump(test, open(("test_data_" + self.task + key), 'wb'))

			for i in range(len(train)):

				if key == 'fever_sup':
					if len(train["claim"].iloc[i]) > 0 and len(train["sentence"].iloc[i]) > 0:
						list_of_defactoNlps.append(DeFactoModelNLP(claim=train["claim"].iloc[i], label=train["lablel"].iloc[i], sentences=train["sentence"].iloc[i], extract_triple=False))
				else:

					if len(train["claim"].iloc[i]) > 0 and len(train["sentence"].iloc[i]) > 0:
						list_of_defactoNlps.append(DeFactoModelNLP(claim=train["claim"].iloc[i], label=train["label"].iloc[i], sentences=train["sentence"].iloc[i], extract_triple=False))
					

			self.extract_features(list_of_defactoNlps, ("train_data_" + key))
				


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='start training')

	parser.add_argument('task', choices=["classification", "detection"], help="what task should be performed?")

	task = parser.parse_args().task

	if task == 'classification':
		dataset_path = [{'EXP_FOLDER': './data/fever/reject/', 'DATASET': 'fever_rej.json'}, 
						{'EXP_FOLDER': './data/fever/support/', 'DATASET': 'fever_sup.json'}]

	else:
		dataset_path = [{'EXP_FOLDER': './data/fever/3-class/', 'DATASET': 'fever_3.json'}]

	featureExtractor_ = featureExtractor(task)
	data = featureExtractor_.load_datafiles(dataset_path)
	featureExtractor_.create_DefactoModel(data)
import pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support
from core.defacto.model import DeFactoModelNLP

from feature_extraction.feature_extractor import featureExtractor


class tuneModel:


	def __init__(self, task):
		self.featureExtractor = featureExtractor(task)


	def test_on_validation_set(self, validation_set, task):

		list_of_defactoNlps = []
		for i in range(len(validation_set)):

				# if key == 'fever_sup':
				if len(validation_set["claim"].iloc[i]) > 0 and len(validation_set["sentence"].iloc[i]) > 0:
					list_of_defactoNlps.append(DeFactoModelNLP(claim=validation_set["claim"].iloc[i], 
								label=validation_set["label"].iloc[i], sentences=validation_set["sentence"].iloc[i], extract_triple=False))

		self.featureExtractor.extract_features(list_of_defactoNlps, ("save_test_defacto_model_fever3"))


	def compute_score(self, list_of_defactoNlps):

		y_pred = []
		y_true = []

		for model in list_of_defactoNlps:
				
			# print ("predicted label")
			# print (model.method_name['vspace']['Detection']['pred_label'])
			# print (model.method_name)
			y_pred.append((model.method_name['tfidf']['Detection']['pred_label']))
			# print ("model label")
			# print (model.label)
			y_true.append(model.label)

		print (len(y_pred))
		print (len(y_true))
		# for binary, average='binary'
		print (precision_recall_fscore_support(y_true, y_pred, average='weighted')) 


if __name__ == '__main__':

	task = 'detection'
	# validation_set = pickle.load(open("test_data_detectionfever_3", "rb"))
	tunemodel = tuneModel(task)
	# tunemodel.test_on_validation_set(validation_set, task)

	list_of_defactoNlps = pickle.load(open("detectionsave_test_defacto_model_fever3", "rb"))
	print ("computin F1 score")
	tunemodel.compute_score(list_of_defactoNlps)
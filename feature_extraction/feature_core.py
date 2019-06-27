from feature_extraction.tfidf import TFIDF
from feature_extraction.vector_space import VectorSpace
from feature_extraction.wmd import wordMoverDistance

class featureCore:

	def __init__(self, task):

		self.task = task
		self.tfidf = TFIDF()
		self.vs = VectorSpace()
		self.wmd = wordMoverDistance()


	def get_tf_idf_score(self, list_of_defactoNlps):

		# print ("nlp mdoes ", list_of_defactoNlps)
		tf_idf_score = 0
		if self.task == 'classification':
			for model in list_of_defactoNlps:
				# print ("model.claim ", model.claim)
				# print ("model sentence ", model.sentences)
				relevant_sentence, score  = self.tfidf.apply_tf_idf(model.claim, model.sentences)
				#0.2 
				if score >= 0.2: 
					model.method_name["tfidf"] = {"Classification":{"pred_label":1} }
				# 	print ("claim ", model.claim)
				# 	print ("most similar sentence ", relevant_sentence)
				else:
					model.method_name["tfidf"] = {"Classification":{"pred_label":0} }

		else:
			for model in list_of_defactoNlps:
				relevant_sentence, score  = self.tfidf.apply_tf_idf(model.claim, model.sentences)
				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: score > 0.6 --> Support, 0.2 < score < 0.6 --> Refutes, NEI < 0.2
				if score >= 0.2: 
					#detection
					# Supports
					if score > 0.4:
						model.method_name["tfidf"] = {"Detection":{"pred_label":0} }
					# refutes
					elif score <= 0.4: # REFUTES
						model.method_name["tfidf"] = {"Detection":{"pred_label":1} }
				#label as NEI	
				else:
					model.method_name["tfidf"] = {"Detection":{"pred_label":2}}



		return list_of_defactoNlps


	def get_vector_space_score(self, list_of_defactoNlps):


		if self.task == 'classification':
			for model in list_of_defactoNlps:

				relevant_sentence, vector_space_score  = self.vs.apply_vector_space(model.claim, model.sentences)
				if vector_space_score >= 0.2:
					model.method_name["vspace"] = {"Classification":{"pred_label":1} }

				else:
					model.method_name["vspace"] = {"Classification":{"pred_label":0} }					
					
		else:
			for model in list_of_defactoNlps:
				relevant_sentence, vector_space_score  = self.vs.apply_vector_space(model.claim, model.sentences)
				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: vector_space_score > 0.6 --> Support, 0.2 < vector_space_score < 0.6 --> Refutes, NEI < 0.2
				if vector_space_score >= 0.1: 
					#detection
					# Supports
					if vector_space_score > 0.4:
						model.method_name["vspace"] = {"Detection":{"pred_label":0} }
					# refutes
					elif vector_space_score <= 0.4: # REFUTES
						model.method_name["vspace"] = {"Detection":{"pred_label":1} }
				#label as NEI	
				else:
					model.method_name["vspace"] = {"Detection":{"pred_label":2}}

		return list_of_defactoNlps



	def get_wmd_score(self, list_of_defactoNlps):

		# print ("nlp mdoes ", list_of_defactoNlps)
		wmd_score = 0
		if self.task == 'classification':
			for model in list_of_defactoNlps:
				relevant_sentence, wmd_score  = self.wmd.compute_wm_distance(model.claim, model.sentences)
				if wmd_score < 2.0:
				
					model.method_name["wmd"] = {"Classification":{"pred_label":1} }

				else:
					model.method_name["wmd"] = {"Classification":{"pred_label":0} }		
	
		else:
			for model in list_of_defactoNlps:
				relevant_sentence, wmd_score  = self.wmd.compute_wm_distance(model.claim, model.sentences)
				# classification: > 0.2 --> Yes, < 0.2 --> NEI
				# Detection: wmd_score > 0.6 --> Support, 0.2 < wmd_score < 0.6 --> Refutes, NEI < 0.2
				if wmd_score <= 2: 
					#detection
					# Supports
					if wmd_score < 1:
						model.method_name["wmd"] = {"Detection":{"pred_label":0} }
					# refutes
					elif wmd_score >= 1: # REFUTES
						model.method_name["wmd"] = {"Detection":{"pred_label":1} }
				#label as NEI	
				else:
					model.method_name["wmd"] = {"Detection":{"pred_label":2}}				

		return list_of_defactoNlps


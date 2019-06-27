import json
from difflib import SequenceMatcher
from nltk.corpus import brown
import numpy as np
from nltk.corpus import wordnet
import nltk.data
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
#nltk.download('brown')
#import spacy
#from spacy import displacy
from collections import Counter
import en_core_web_sm
import wikipedia
import gensim

'''
def fuzzy_search(search_key, text, threshold):
    lines = text.split("\n")
    out = [] # similar work, line
    for i, line in enumerate(lines):
        words = line.split()
        for word in words:
            similarity = SequenceMatcher(None, word, search_key)
            if similarity.ratio() > threshold:
                out.append([word, i+1])
    return out


def get_wiki_documents(keywords, subject, object, min_similarity = 0.90, max_nr_documents=5, language='en'):
    try:
        documents = []
        wikipedia.set_lang(language)
        for token in keywords:
            # typos
            sim_s = get_jaccard_similarity(token, subject)
            sim_o = get_jaccard_similarity(token, object)
            # semantically similar cases
            doc_s = nlp(subject)
            doc_o = nlp(object)
            doc_token = nlp(token)
            sim_s_token = doc_token.similarity(doc_s)
            sim_o_token = doc_token.similarity(doc_o)

            if ((sim_s or sim_o or sim_s_token or sim_o_token) >= min_similarity):
                wikidata = wikipedia.summary(token, sentences=max_nr_documents)
                documents.extend(wikidata)
        return documents
    except:
        raise
'''
'''
def get_ne_from_text(text):
    try:
        doc = nlp(text)
        return doc.ents
    except:
        raise

def get_cosine_similarity(*strs):
    vectors = [t for t in __get_vectors(*strs)]
    return cosine_similarity(vectors)

def __get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_jaccard_similarity(a, b):

    do perform lemmatization in a and b before calling it!
    :param a:
    :param b:
    :return:

    a = set(a.split())
    b = set(b.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def penn_to_wn(tag):

    convert between a PTB tag to Wordnet tag
    :param tag:
    :return:

    if tag.startswith('N'): return 'n'
    if tag.startswith('V'): return 'v'
    if tag.startswith('J'): return 'a'
    if tag.startswith('R'): return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def symmetric_sentence_similarity(a, b):
    return (__sentence_similarity(a, b) + __sentence_similarity(b, a)) / 2

def __sentence_similarity(sentence1, sentence2):
    s1 = pos_tag(word_tokenize(sentence1))
    s2 = pos_tag(word_tokenize(sentence2))

    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in s1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in s2]

    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score
'''

'''
degree = find_syn('degree')
birth = find_syn('birth')
birth.remove(birth[0])
birth.remove(birth[len(birth)-1])
death = find_syn('death')
'''


'''
def get_feature_pos(data):
    features = []

    for i in range(len(data)):
        try:
            vec = []
            sub = data[i]['subject'].lower()
            obj = data[i]['obj'].lower()
            body = data[i]['body'].lower()
            pred = data[i]['pred'].lower()
            vec.append(get_is_sub(sub,body))
            vec.append(get_is_obj(obj,body))
            vec.append(get_is_pred(pred,body))
            if vec[0] and vec[1]:
                vec.append(get_dis_s_o(sub,obj,body))
            else:
                vec.append(10000)
            vec.append(get_is_sub_rel(sub,body))
            vec.append(get_is_obj_rel(obj,body))

            if pred == 'place of birth' or pred == 'date of birth':
                vec.append(get_is_pred_rel(birth, body))
            elif pred == 'place of death':
                vec.append(get_is_pred_rel(death, body))
            elif pred == 'degree':
                vec.append(get_is_pred_rel(degree, body))
            vec.append(1)
            features.append(vec)
        except:
            continue

    return features


def get_feature_neg(data):
    features = []

    for i in range(len(data)):
        try:
            vec = []
            sub = data[i]['subject'].lower()
            obj = data[i]['obj'].lower()
            body = data[np.random.randint(0,len(data))]['body'].lower()
            pred = data[i]['pred'].lower()
            vec.append(get_is_sub(sub,body))
            vec.append(get_is_obj(obj,body))
            vec.append(get_is_pred(pred,body))
            if vec[0] and vec[1]:
                vec.append(get_dis_s_o(sub,obj,body))
            else:
                vec.append(10000)
            vec.append(get_is_sub_rel(sub,body))
            vec.append(get_is_obj_rel(obj,body))
            if pred == 'place of birth' or pred == 'date of birth':
                vec.append(get_is_pred_rel(birth, body))
            elif pred == 'place of death':
                vec.append(get_is_pred_rel(death, body))
            elif pred == 'degree':
                vec.append(get_is_pred_rel(degree, body))
            vec.append(0)
            features.append(vec)
        except:
            continue

    return features
'''

#eight features now


'''
if __name__ == "__main__":

    #nlp = en_core_web_lg.load()


    s1 = 'diego esteves lives in Porto'
    s2 = 'esteves was born in Rio de Janeiro'
    s3 = 'diego is portuguese'
    s4 = 'piyush is a nice guy'
    print(get_cosine_similarity(s1, s2, s3, s4))
    print(get_jaccard_similarity(s1, s2))
'''


class Features:

    here=0

    def __init__(self):
        '''
        self.body=body[0].lower()
        self.subject=subject[0].lower()
        self.object=subject[0].lower()
        self.predicate=predicate[0].lower()
        '''
    def find_syn(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if len(synonyms)>0:
            return synonyms
        else:
            return word
    def is_p_in_between(self, body, sub, pred, obj):
        si = body.find(sub)
        oi = body.find(obj)
        pi = body.find(pred)
        if (si < pi < oi):
            return 1
        else:
            return 0


    def get_is_sub(self, sub, body):
        if sub in body:
            return 1
        else:
            return 0

    def get_is_obj(self, obj, body):
        if obj in body:
            return 1
        else:
            return 0

    def get_is_pred(self, pred, body):
        if pred in body:
            return 1
            #print(here)
        else:
            return 0

    def get_proof_sentiment(self ):
        return

    def get_dis_s_o(self, sub, obj, body):
        return abs(body.find(sub) - body.find(obj))

    def get_is_sub_rel(self, sub, body):
        sub = sub.split()
        ct=0
        for i in sub:
            if i in body:
                ct+=1
        return (ct * 1.0) / len(sub)

    def get_is_obj_rel(self, obj, body):
        ct = 0
        #if(len(obj)==0):
            #print(obj)
            #print(body)
        obj = obj.split()
        #if(len(obj)==0):
            #print(obj)
            #print(body)
        for i in obj:
            if i in body:
                ct += 1
        return (ct * 1.0) / len(obj)

    def get_is_pred_rel(self, pred, body):
        #print(pred)
        #print(body)
        ct = 0
        for i in pred:
            if i in body:
                ct += 1
        return (ct * 1.0) / len(pred)

    def get_dis_s_o_rel(self, sub, obj, body):
        return

    def word2vecModel(self):
        sentences = brown.sents()
        model = gensim.models.Word2Vec(sentences, min_count=1)
        model.save('brown_model')

    def max_sim(self,body,predicate):

        wordList=[]
        model = gensim.models.Word2Vec.load('brown_model')
        wordList=body.split()
        m=[]
        #model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        for word in wordList:
            try:
                if word in model and predicate in model:
                    #print(word+"\t"+predicate)
                    m.append(model.similarity(word,predicate))
            except:
                m.append(0)
        return(max(m))

    def extract_features(self, doc, subject, predicate, object):
        '''
        print(doc)
        print(subject)
        print(predicate)
        print(object)
        '''
        vec = []
        body=doc.lower()
        subj=subject.lower()
        obj=object.lower()
        pred=predicate.lower()
        #print("body:"+body+"\nSubejct:"+subj+"\nObject:"+obj+"\nPredicate:"+pred)
        vec.append(self.get_is_sub(subj, body))
        vec.append(self.get_is_obj(obj, body))
        vec.append(self.get_is_pred(pred,body))
        if vec[0] and vec[1]:
            self.here+=1
            vec.append(self.get_dis_s_o(subj, obj, body))
            vec.append(self.is_p_in_between(body,subj,pred,obj))
        else:
            vec.append(1000)
            vec.append(0)
        vec.append(self.get_is_sub_rel(subj, body))
        vec.append(self.get_is_obj_rel(obj, body))
        vec.append(self.get_is_pred_rel(self.find_syn(pred), body))
        return vec

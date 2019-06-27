from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def find_similar(tfidf_matrix, document):
    top_n = len(document) #change if need top_n
    index = 0
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(document[index-1], cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def give_sen(claim,document):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform([claim]+document)
    return find_similar(tfidf_matrix,document)


if __name__ == "__main__":
    print(give_sen('diego is a good guy',['diego is a good guy','diego is cool','diego is a guy','diego loves Portugal','diego is a researcher','some unrealted sentence']))
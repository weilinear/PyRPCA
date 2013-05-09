
from time import time
from sklearn.feature_extraction import text
from sklearn import decomposition
from sklearn import datasets
from robustpca import *

n_samples = 5000
n_features = 2000
n_topics = 10
n_top_words = 5

# Load the 20 newsgroups dataset and vectorize it using the most common word
# frequency with TF-IDF weighting (without top 5% stop words)

t0 = time()
print("Loading dataset and extracting TF-IDF features...")
dataset = datasets.fetch_20newsgroups(shuffle=True, random_state=1)

vectorizer = text.CountVectorizer(max_df=0.95, max_features=n_features)
counts = vectorizer.fit_transform(dataset.data[:n_samples])
tfidf = text.TfidfTransformer().fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model on with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
# import pdb; pdb.set_trace()
A, E = augmented_largrange_multiplier(np.array(tfidf.todense().T), lmbda=.1, maxiter=10, inexact=True) # decomposition.NMF(n_components=n_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))
import pdb; pdb.set_trace()
# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(np.abs(E.T)):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]  if topic[i] != 0]))
    print()

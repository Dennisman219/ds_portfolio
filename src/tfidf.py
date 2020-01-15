#!/usr/bin/env python
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
import sys

categories = []
descriptions = []
for line in sys.stdin:
	try:
		line = line.rstrip()
		c,d = line.split(',')
		categories.append(c)
		descriptions.append(d)
	except ValueError as e:
		exit(1)

corpus = descriptions

vocabulary = set()
for d in corpus:
	vocabulary.update(d.split())
	#vocabulary.update(d)
vocabulary = list(vocabulary)

categories_unique = list(set(categories))
#categories_unique = list(categories_unique)
print(categories_unique)


vectorizer = TfidfVectorizer(vocabulary=None)
X = vectorizer.fit_transform(corpus)
#for w, s in vectorizer.vocabulary_.items():
#	sys.stdout.write("{}: {}\n".format(w, s))
#print(len(vectorizer.vocabulary_))

indexes = {}
i = 0
for c in categories:
	if not c in indexes:
		indexes[c] = i
		i += 1
#print(indexes)

tsne = TSNEVisualizer()
tsne.fit(X, categories)
tsne.show()
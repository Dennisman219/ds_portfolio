#!/usr/bin/env python3

import pickle
from yellowbrick.text import TSNEVisualizer

with open('data/agorb.csv', 'rb') as file:
	agora = pickle.load(file)

with open('data/tno/tfidf_vectors_webiq.pkl', 'rb') as file:
	X = pickle.load(file)

with open('data/tno/categorieen.pkl', 'rb') as file:
	c = pickle.load(file) 

tsne = TSNEVisualizer()
tsne.fit(X, c)
tsne.show()
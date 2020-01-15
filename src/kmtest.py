#!/usr/bin/env python
# clustering dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
import pandas as pd

csv = pd.read_csv('data/processed.csv', header=None).dropna()
categories = csv[0] # First column is categories
corpus = csv[1] # Second column is title + description

tfidf = TfidfVectorizer(vocabulary=None)
X = tfidf.fit_transform(corpus)

k = np.unique(categories).shape[0]
print("Clustering into {} clusters".format(k))
km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
km.fit(X)

print("Homogeneity:  {}".format(metrics.homogeneity_score(categories, km.labels_)))
print("Completeness: {}".format(metrics.completeness_score(categories, km.labels_)))
print("V-measure:    {}\n".format(metrics.v_measure_score(categories, km.labels_)))
print("Adjusted Rand-Index:    {}".format(metrics.adjusted_rand_score(categories, km.labels_)))
print("Silhouette Coefficient: {}".format(metrics.silhouette_score(X, km.labels_, sample_size=1000)))
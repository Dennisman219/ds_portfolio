#!/usr/bin/env python
# clustering dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('data/processed.csv', header=None).dropna()
categories = csv[0] # First column is categories
corpus = csv[1] # Second column is title + description

tfidf = TfidfVectorizer(vocabulary=None)
X = tfidf.fit_transform(corpus)
X = X.todense()

k = np.unique(categories).shape[0]
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)

print("Clustering into {} clusters".format(k))
km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
km.fit(X)

print("Homogeneity:  {}".format(metrics.homogeneity_score(categories, km.labels_)))
print("Completeness: {}".format(metrics.completeness_score(categories, km.labels_)))
print("V-measure:    {}\n".format(metrics.v_measure_score(categories, km.labels_)))
print("Adjusted Rand-Index:    {}".format(metrics.adjusted_rand_score(categories, km.labels_)))
print("Silhouette Coefficient: {}".format(metrics.silhouette_score(X, km.labels_, sample_size=1000)))

centers_2d = pca.transform(km.cluster_centers_)
plt.scatter(data2D[:,0], data2D[:,1], c=km.labels_.tolist())
plt.scatter(centers_2d[:,0], centers_2d[:,1], marker='x', s=200, linewidths=3, c='r')
plt.show()
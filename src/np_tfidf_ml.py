#!/usr/bin/env python3
import sys, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score

test_size_f = 0.2

csv = pd.read_csv('data/processed.csv', header=None).dropna()
test_size = int(len(csv) * test_size_f)
train_size = int(len(csv) - test_size)
#print("test: {}, train: {}".format(test_size, train_size))

#csv = csv.dropna()
categories = csv[0]
descriptions = csv[1]
corpus = descriptions
#print(corpus)
#print(categories)

train_d = corpus[:train_size]
train_c = categories[:train_size]
test_d = corpus[train_size:]
test_c = categories[train_size:]
#print("descriptions: test: {}, train: {}".format(len(test_d), len(train_d)))
#print("categories:   test: {}, train: {}".format(len(test_c), len(train_c)))

tfidf = TfidfVectorizer(vocabulary=None)
vectorizer = tfidf.fit(corpus)
#print(vectorizer)

X_train = vectorizer.transform(train_d)
Y_train = train_c
X_test = vectorizer.transform(test_d)
Y_test = test_c
#print(X_test)
#print(X_train)

#tsne = TSNEVisualizer()
#tsne.fit(X_train, categories)
#tsne.show()

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
predictions = sgd.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
recall = recall_score(Y_test, predictions, average='micro')
precision = precision_score(Y_test, predictions, average='micro')
print("SGD accuracy:  {}".format(accuracy))
#print("    recall:    {}".format(recall))
#print("    precision: {}".format(precision))

mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
predictions = mnb.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
recall = recall_score(Y_test, predictions, average='micro')
precision = precision_score(Y_test, predictions, average='micro')
print("MNB accuracy:  {}".format(accuracy))
#print("    recall:    {}".format(recall))
#print("    precision: {}".format(precision))
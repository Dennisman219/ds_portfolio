#!/usr/bin/env python3
"""
		TF*IDF	W2V (1)	W2V (2)
lem		0		1		1
stop	0		1		0
punc	1		1		1
lower	1		1		1
unicode	1		1		1
nummer	1		1		0
n-gram	1		0		0
"""

import pandas as pd, sys, string, math, time, pickle
from polyglot.text import Text, Word
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
#from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#import re

input_file = sys.argv[1]
output_dir = sys.argv[2] # HOEFT GEEN '/' ACHTER MAPNAAM
print('input: \'{}\', output: \'{}\''.format(input_file, output_dir))
predictor_columns = [' Item', ' Item Description'] # Twee kolommen voor predictor
target_column = ' Category'
print('predictor kolommen: \'{}\'; target kolom: \'{}\''.format('\', \''.join(predictor_columns), target_column))

print('{} inladen...'.format(input_file))
csv = pd.read_csv('{}'.format(input_file))

cats = []
predictor = []
for i in range(0, len(csv[predictor_columns[0]])):
	#print("{} {}".format(csv[predictor_columns[0]][i], csv[predictor_columns[1]][i]))
	line = "{} {}".format(csv[predictor_columns[0]][i], csv[predictor_columns[1]][i])
	#line = re.sub(r"\s+", ' ', line)
	line = line.replace(r'\n', ' ')
	line = line.replace(r'\t', ' ')
	line = bytes(line, 'utf-8').decode('utf-8','ignore')
	try:
		text = Text(line)
		#print(i)
		if text.language.code == 'en':
			predictor.append(line)
			cats.append(csv[target_column][i])
	except:
		pass
		#print('weggegooid')

print('Lijst met categorieen opslaan...')
with open('{}/categorieen.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(cats, fin)

target = csv[target_column]
#print(target)

p_bu = predictor

punc_translator = str.maketrans("", "", string.punctuation)
numb_translator = str.maketrans("", "", string.digits)

# TFIDF
print('processen voor tfidf...')
processed_tfidf = []
for l in predictor:
	l = l.translate(punc_translator)
	l = l.lower()
	l = l.encode('ascii', 'ignore').decode("utf-8")
	l = l.translate(numb_translator)
	processed_tfidf.append(l)

print('fit/transform voor tfidf...')
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = tfidf_vectorizer.fit_transform(processed_tfidf)

print('tfidf opslaan...')
with open('{}/tfidf_vectors.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(X, fin)

with open('{}/tfidf_model.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(tfidf_vectorizer, fin)

predictor = p_bu

# PREP GENERIEK 1
print('eerste generieke prep...')
stop_dict = set(map(lambda x: x.lower(), stopwords.words("english")))
processed_prep1 = []
for l in predictor:
	word_tokens = word_tokenize(l)
	lemmatized = [WordNetLemmatizer().lemmatize(w) for w in word_tokens]
	#l = " ".join(lemmatized)
	without_stop = [w for w in lemmatized if not w in stop_dict]
	l = " ".join(without_stop)
	l = l.translate(punc_translator)
	l = l.lower()
	l = l.encode('ascii', 'ignore').decode("utf-8")
	l = l.translate(numb_translator)
	processed_prep1.append(l)

predictor = p_bu

# PREP GENERIEK 2
print('tweede generieke prep...')
processed_prep2 = []
for l in predictor:
	word_tokens = word_tokenize(l)
	lemmatized = [WordNetLemmatizer().lemmatize(w) for w in word_tokens]
	l = " ".join(lemmatized)
	l = l.translate(punc_translator)
	l = l.lower()
	l = l.encode('ascii', 'ignore').decode("utf-8")
	processed_prep2.append(l)

# W2V
# tokenized = [word_tokenize(row) for row in processed_prep1]
# model = Word2Vec(tokenized, size=128, workers=8)
# w2v_vectors_prep1 = []
# for i, row in enumerate(tokenized):
# 	sentence_vectors = [model.wv[word] for word in row if word in model.wv]
# 	# if len(sentence_vectors) == 0:
# 	# 	vectors.append([0] * size)
# 	# else:
# 	# 	sentence_vector = np.average(sentence_vectors, axis=0)
# 	w2v_vectors_prep1.append(sentence_vector)

# FASTTEXT
print('fasttext met eerste prep...')
tokenized = [word_tokenize(row) for row in processed_prep1]
model = FastText(tokenized, size=128, workers=8)
ft_vectors_prep1 = []
for i, row in enumerate(tokenized):
	sentence_vectors = [model.wv[word] for word in row]
	# if len(sentence_vectors) == 0:
	#     vectors.append([0] * size)
	# else:
	#     sentence_vector = np.average(sentence_vectors, axis=0)
	ft_vectors_prep1.append(sentence_vectors)

print('fasttext prep 1 opslaan...')
with open('{}/ft_vectors_prep1.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep1, fin)

with open('{}/ft_model_prep1.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)


print('fasttext met tweede prep...')
tokenized = [word_tokenize(row) for row in processed_prep2]
model = FastText(tokenized, size=128, workers=8)
ft_vectors_prep2 = []
for i, row in enumerate(tokenized):
	sentence_vectors = [model.wv[word] for word in row]
	# if len(sentence_vectors) == 0:
	#     vectors.append([0] * size)
	# else:
	#     sentence_vector = np.average(sentence_vectors, axis=0)
	ft_vectors_prep2.append(sentence_vectors)

print('fasttext prep 2 opslaan...')
with open('{}/ft_vectors_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep2, fin)

with open('{}/ft_model_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)

# DOC2VEC
print('doc2vec met eerste prep...')
documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(processed_prep1)]
model = Doc2Vec(documents, vector_size=128, workers=8)
d2v_vectors_prep1 = [model.docvecs[i] for i, _doc in enumerate(processed_prep1)]

print('doc2vec prep 1 opslaan...')
with open('{}/d2v_vectors_prep1.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep1, fin)

with open('{}/d2v_model_prep1.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)


print('doc2vec met tweede prep...')
documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(processed_prep2)]
model = Doc2Vec(documents, vector_size=128, workers=8)
d2v_vectors_prep2 = [model.docvecs[i] for i, _doc in enumerate(processed_prep2)]

print('doc2vec prep 2 opslaan...')
with open('{}/d2v_vectors_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep2, fin)

with open('{}/d2v_model_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)
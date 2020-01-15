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

def get_interpol_cat(c):
	#print(c)
	return interpol_map['Interpol'].values[interpol_map.index[interpol_map['Agora'] == c].tolist()[0]]

input_file = sys.argv[1]
output_dir = sys.argv[2] # HOEFT GEEN '/' ACHTER MAPNAAM
print('input: \'{}\', output: \'{}\''.format(input_file, output_dir))
predictor_columns = ['title', 'content'] # Twee kolommen voor predictor
target_column = 'category'
print('predictor kolommen: \'{}\'; target kolom: \'{}\''.format('\', \''.join(predictor_columns), target_column))

interpol_map = pd.read_csv('data/interpol_map.csv')

print('{} inladen...'.format(input_file))
csv = pd.read_csv('{}'.format(input_file))
csv = csv[csv[target_column] != 'Other']

agora_csv = pd.read_csv('data/agorb.csv')
agora_csv = agora_csv[agora_csv[' Category'] != 'Other']
agora_csv.columns = ["Vendor", target_column, predictor_columns[0], predictor_columns[1], "Price" , "Origin", "Destination", "Rating", "Remarks", "Dummy"]

num_rows_webiq = len(csv)
print("lengte webiq csv: {}".format(num_rows_webiq))
print("lengte agora csv: {}".format(len(agora_csv)))

#print("columns renamed")
csv = pd.concat([csv, agora_csv], sort=False)

csv[target_column] = csv[target_column].apply(lambda x: get_interpol_cat(x))

items = csv[predictor_columns[0]].values
item_descriptions = csv[predictor_columns[1]].values
categories = csv[target_column].values
cats = []
predictor = []
for i in range(0, len(items)):
	#print("{} {}".format(items[i], item_descriptions[i]))
	line = "{} {}".format(items[i], item_descriptions[i])
	#line = re.sub(r"\s+", ' ', line)
	line = line.replace(r'\n', ' ')
	line = line.replace(r'\t', ' ')
	line = bytes(line, 'utf-8').decode('utf-8','ignore')
	try:
		text = Text(line)
		#print(i)
		if text.language.code == 'en':
			predictor.append(line)
			cats.append(categories[i])
	except:
		pass
		#print('weggegooid')

# for i in range(0, len(agora_csv[' Item'])):
# 	line = "{} {}".format(agora_csv[' Item'][i], agora_csv[' Item Description'][i])
# 	line = line.replace(r'\n', ' ')
# 	line = line.replace(r'\t', ' ')
# 	predictor.append(line)

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

processed_tfidf_webiq = processed_tfidf[:num_rows_webiq]
processed_tfidf_agora =	processed_tfidf[num_rows_webiq:]

print('fit/transform voor tfidf...')
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
#X = tfidf_vectorizer.fit_transform(processed_tfidf)

vectorizer = tfidf_vectorizer.fit(processed_tfidf)
X_webiq = vectorizer.transform(processed_tfidf_webiq)
X_agora = vectorizer.transform(processed_tfidf_agora)

print('tfidf opslaan...')
with open('{}/tfidf_vectors_webiq.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(X_webiq, fin)

with open('{}/tfidf_vectors_agora.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(X_agora, fin)

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

processed_prep1_webiq = processed_prep1[:num_rows_webiq]
processed_prep1_agora =	processed_prep1[num_rows_webiq:]

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

processed_prep2_webiq = processed_prep2[:num_rows_webiq]
processed_prep2_agora =	processed_prep2[num_rows_webiq:]

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

ft_vectors_prep1_webiq = ft_vectors_prep1[:num_rows_webiq]
ft_vectors_prep1_agora = ft_vectors_prep1[num_rows_webiq:]

print('fasttext prep 1 opslaan...')
with open('{}/ft_vectors_prep1_webiq.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep1_webiq, fin)

with open('{}/ft_vectors_prep1_agora.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep1_agora, fin)

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

ft_vectors_prep2_webiq = ft_vectors_prep2[:num_rows_webiq]
ft_vectors_prep2_agora = ft_vectors_prep2[num_rows_webiq:]

print('fasttext prep 2 opslaan...')
with open('{}/ft_vectors_prep2_webiq.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep2_webiq, fin)

with open('{}/ft_vectors_prep2_agora.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(ft_vectors_prep2_agora, fin)

with open('{}/ft_model_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)

# DOC2VEC
print('doc2vec met eerste prep...')
documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(processed_prep1)]
model = Doc2Vec(documents, vector_size=128, workers=8)
d2v_vectors_prep1 = [model.docvecs[i] for i, _doc in enumerate(processed_prep1)]

d2v_vectors_prep1_webiq = d2v_vectors_prep1[:num_rows_webiq]
d2v_vectors_prep1_agora = d2v_vectors_prep1[num_rows_webiq:]

print('doc2vec prep 1 opslaan...')
with open('{}/d2v_vectors_prep1_webiq.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep1_webiq, fin)

with open('{}/d2v_vectors_prep1_agora.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep1_agora, fin)

with open('{}/d2v_model_prep1.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)


print('doc2vec met tweede prep...')
documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(processed_prep2)]
model = Doc2Vec(documents, vector_size=128, workers=8)
d2v_vectors_prep2 = [model.docvecs[i] for i, _doc in enumerate(processed_prep2)]

d2v_vectors_prep2_webiq = d2v_vectors_prep2[:num_rows_webiq]
d2v_vectors_prep2_agora = d2v_vectors_prep2[num_rows_webiq:]

print('doc2vec prep 2 opslaan...')
with open('{}/d2v_vectors_prep2_webiq.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep2_webiq, fin)

with open('{}/d2v_vectors_prep2_agora.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(d2v_vectors_prep2_agora, fin)

with open('{}/d2v_model_prep2.pkl'.format(output_dir), 'wb+') as fin:
	pickle.dump(model, fin)
# Portfolio Dennis van Gilst KB-74
## Datacamp
### Completion
I have completed 10/11 courses (91%).

[datacamp]: datacamp.png
![datacamp]

## Reflection
Please see reflection.md for my three starr reflections.

## Research Project
### Task definition
TNO is a partner in the TITANIUM (Tools for the Investigation of Transactions in Underground Markets) project set up by the European Union. International law enforcement in intrerested in gaining insights into the transaction on the dark web. TNO has created the assignment to create a classifier which can take text-based content sourced from the dark web and classify this text into a category. This provides information about what topics are being discussed. Interpol has provided a list of categories to use for classification.

We have formulated the following research question:
```
How can a pipeline be created that classifies dark web text based content to a predetermined topics list?
```
We have formulated the follow subquestions:
```
- How are the darkweb forums / markets structured?
- What labels of the dataset provided by TNO are relevant for the research?
- What strategies can be used to preprocess the available data?
- What feature extraction methods are available for text classification?
- What machine learning algorithms can be used for natural language processing?
- What preprocessing method works best on the dark web text content?
- What feature extraction methods give the best vectorization of the text content?
- What machine learning algorithms give the best result?
- What combination of processing, feature extraction and machine learning algorithms gives the best pipeline?
- How can unsupervised machine learning be applied for the validation of the classifier?
```

### Evaluation 
As mentioned in the paper, the two dataset we used are structured quite differently. The Agora data features many rows but little text per row and the Web-IQ data features fewer rows but more text per row. We achieved the highest scores when training and testing with the Agora set.\
Training on the Agora data and testing on the Web-IQ data results in low scores. I suspect we would be able to achieve higher scores if we could combine both dataset together in order to train a model on mixed dataset features many records with varying length per record.

### Conlusion
The research found that the best pipeline for classifying text-based content consisted of a preprocessing component that minimally alters the text. The vectorization component used tf-idf and applies n-grams to create shingles of two words to create feature vectors of the used corpus. Finally, a Linear SVC model was trained that can be used to classify new data.

### Planning
At the beginning of the project a general planning was created to which we can refer to keep track of our general progress and things to do. In this planning we tried to follow a itterative process, i.e. we can return to tasks after 'completing' it earlier.

[general_planning]: general_planning.png
![general_planning]

Later, when more detailed goals and to-do points were defined we created a trello board in which we created epics and tasks which were picked up and completed in sprints of two weeks. Tasks were assigned to a person either at creation or when someone decided to pick up a open task.

[trello_board]: trello_board.png
![trello_board]

## Predictive Analysis
### Selecting a model
In the beginning of the project we wanted to see if unsupervised machine learning was promising enough to continue using. I created the script kmtest.py which uses tf-idf and a k-means clustering algorithm to cluster the vectors together. This model was selected for a couple of reasons:
- The model is described as a simple model which might be smart given my limited understanding at the time of creation.
- The model is unsupervised, which is what we wanted to try out.
- The model clusters the vectors which aligns best with our classifiction problem.

### Configuring a model
The model is configured with the following lines:
```python
k = np.unique(categories).shape[0]
km = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=1)
```
The model is configured to create as many clusters as there are categories. When using full hierarchical categories k = 104, when using primary categories k = 14.\
The model is configured to use 'k-means++' initialization. This is described as a 'smart' initialization technique and is beneficial to speed.\
A max_iter of 1000 was chosen over the default of 300 to ensure plenty of iterations to form clusters.\
An n_init of 1 was chosen over the default of 10 in order to force the model to try one random starting distribution of clusters to save time.

### Training a model
I created the script np_tfidf_ml.py in which I train two models on tf-idf vectors.\
This is an example of training a SGDClassifier model from the sklearn library, which uses stochastic gradient descent (SGD) to train a support vector machine (SVM).
```python
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
predictions = sgd.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
recall = recall_score(Y_test, predictions, average='micro')
precision = precision_score(Y_test, predictions, average='micro')
print("SGD accuracy:  {}".format(accuracy))
#print("    recall:    {}".format(recall))
#print("    precision: {}".format(precision))
```

### Evaluating a model
The script uses cross-validation the validate its models.
```python
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
```

The script also trains a Multinominal Naive Bayes (MNB) model.
```python
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
predictions = mnb.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
recall = recall_score(Y_test, predictions, average='micro')
precision = precision_score(Y_test, predictions, average='micro')
print("MNB accuracy:  {}".format(accuracy))
#print("    recall:    {}".format(recall))
#print("    precision: {}".format(precision))
```
Recall and precision we excluded for both models because they were exactly the same as accuracy, which I presume to be an error (see below).
```bash
SGD accuracy:  0.9372258152051015
#   recall:    0.9372258152051015
#   precision: 0.9372258152051015
MNB accuracy:  0.9119442811270408
#   recall:    0.9119442811270408
#   precision: 0.9119442811270408
```
Both models achieve a high score. The SVM/SGD model scores highest. Both these models were trained and testen of the full Agora data (110595 rows). When using balanced data with categories capped at max 500 occurences (resulting in 6433 rows) the following scores are achieved:
```bash
SGD accuracy:  0.7161741835147745
MNB accuracy:  0.40902021772939345
```
Both models perform quite poorly, but the MNB score the lowest by far. It seems this model does not perform well when using litte data.

### Visualising output of model


## Domain Knowledge
### Introduction of subject field
We are building a classifier which is trained classifies text from listings from marketplaces on the dark web.\
The dark web is a part of the deep web.\ The deep web is a part of the internet which is not indexable by search engines like Google or crawlers. The dark web adds another layer to this by applying encryption and using peer-to-peer networking. This results in user being anonymous and untrackable. The user of the dark web use it to sell illegal items. International law inforcement is interested in gaining insights into transactions made via the dark web. Our classifier can take text-based content from listing of items being sold and predict to which category the item belongs.

### Literature research
I have looked into many different Data Science related topics for this minor. I spent the most effort into gaining knowledge about topics related to our project, such as:
- Processing text-based data in ways beneficial to machine learning
- Vectorizing text-based data to allow for training of a model.
- What models are available for classification
- Validating the performance of a model

### Terminology
I created the following list of definitions of terms used most often in our project:

Pipeline: A standard set of steps required to take to perform successfull machine learning.

Preprocessing: Altering raw data to benefit certain machine learning techniques.\
Lemmatization: The process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.\
Stemming: The process of reducing inflected (or sometimes derived) words to their word stem.\
N-grams: A contiguous sequence of n items from a given sample of words

Vectorization: Using a technique to transform text-based content into numbers or vectors usable by machine learning algorithms\
Word embeddings: The collective name for a set of language modeling and feature learning techniques in NLP.\
tf-idf: term frequency, inverse document frequency. A Vectorization technique that is intended to reflect how important a word is to a document in a collection or corpus. The weight of a word is calculated by multiplying the frequency of a term in a document by the inverse frequency of the term in other documents. A vector consist the weights of each word in a document (one vector per document).\
word2vec: A group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words.\
doc2vec: An unsupervised algorithm to generate vectors for sentences/paragraphs/documents.\
fasttext: A library for learning of word embeddings and text classification created by Facebook's AI Research lab.

Machine learning: The scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead.\
Training: Using historical data to allow a model to generalize the relations in this data.\
Classification: The problem of identifying to which of a set of categories a new observation belongs.\
Regression: Estimating the relationships between a dependent variable (often called the 'target variable') and one or more independent variables (often called 'predictors', 'covariates', or 'features').\
Hyper parameter: A parameter whose value is set before the learning process begins. By contrast, the values of other parameters are derived via training.\
Support Vector Machine: Supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.\
K-means clustering: A method of vector quantization which aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.

Performance: A way of showing numerically how well a machine learning model is able to correctly perform regression or classification.\
Validation: Using labeled data the calculate the performance of a model. Not applicable to unsupervised models, as they do not use labeled data.\
Cross validation: Using data to validate a model that differs from the data that was used during training.\
True positive: A datapoint which a model correctly labeled as of a certain class.\
True negative: A datapoint which a model correctly labeled as not of a certain class.\
False positive: A datapoint which a model incorrectly labeled as of a certain class.\
False negative: A datapoint which a model incorrectly labeled as not of a certain class.\
Accuracy: A score of the performance of a model between zero and one. Calculated by dividing the number of correct predictions by the number of total predictions.\
Precision: A score of the performance of a model between zero and one. Calculated by dividing the number of true positive predictions by the number total positive predictions.\
Recall: A score of the performance of a model between zero and one. Calculated by dividing the number of true positive predictions by sum of the true positive predictions and false negative predictions.\
F1 score: A score of the performance of a model between zero and one. Calculated by dividing two by the sum of the inverse recall score and the  inverse prediction score.\
Confusion matrix: A table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

Codebook: A document which provides insights into a used dataset by describing its origins, structure, data types and used components.\
Deep web: Part of the World Wide Web whose contents are not indexed by standard web search-engines.\
Dark web: Onderdeel van het deep web maar niet rechtstreeks toegankelijk. Er is speciale software noodzakelijk die de gebruiker anonimiteit moet verschaffen zoals Tor\
NLP: Natural language processing. An interdisciplinary field concerned with the statistical or rule-based modeling of natural language from a computational perspective, as well as the study of appropriate computational approaches to linguistic questions.
Agora: A marketplace on the dark web which no longer exists. We use the term to refer to the dataset containing data from this marketplace.

## Data Preprocessing
### Data exploration
A TSNE/tf-idf projection of the full dataset shows that the dataset is dominated by listings about Drugs.

[tsne-volledig]: tsne-tfidf-volledig.png
![tsne-volledig] TSNE plots were created with tfidf.py

A Biased dataset can result in poor performance of classification. We experimented with balancing the dataset and comparing the performance.

### Data cleansing
The Agora datafile contained several syntax errors with resulted in faulty parsing of the file. The lines containing errors were removed or fixed and a new file without errors called 'agorb.csv' has been created.

### Data preparation
I created the program prep.py to process the data. This program can undertake several operations to prepare the data for vectorization and machine learning.

The listing title and description is combined to create a single predictor.
```python
self.data = self.agora[' Item'] + " " + self.agora[' Item Description']
```

The program can save full hierarchical categories or only primary categories.
```python
if self.primcat:
    c = self.agora[' Category'][t].lower().split('/')[0]
else:
	c = self.agora[' Category'][t].lower()
```

The program removes numbers and punctuation.
```python
row = row.translate(str.maketrans('', '', '{}1234567890'.format(string.punctuation))
```

The program removes stop words.
```python
if w in self.stop_words:
    del words[i]
```

The program can optionally apply stemming or lemmatization.
```python
if self.lemmatize:
    lemmed = self.lemmer.lemmatize(w)
	words[i] = lemmed
	w = lemmed
	del lemmed
if self.stem:
	stemmed = self.stemmer.stem(w)
	words[i] = stemmed
	w = stemmed
	del stemmed
```

The program can balance the data by counting how often a category has occured. The program skips a listing if it has reached the maximum amount of occurences.
```python
if self.catocur_max > 0:
	try:
		if catocur[c] >= self.catocur_max:
			t += 1
			continue
		else:
			catocur[c] += 1
	except KeyError:
		catocur[c] = 1
```

In the script tno_prepvec_agora.py I decided to remove listing from the 'Other' category as these were useless to training.
```python
csv = csv[csv[target_column] != 'Other']
```

### Data explanation
I created [this](https://docs.google.com/spreadsheets/d/1gN1_2B79F_eKS_axDTHM5dBmgCnCSXjKqo3MmujwT8Y/edit?usp=sharing) codebook to provide insights into the Agora data.

### Data Visualisation
A TSNE/tf-idf plot of a balanced dataset shows better clusters. This plot supports our decision that this dataset is usable for training a classification model and that balancing the data should enhance performance. Some listings should still be difficult to classify, as the plot shows since many listings are grouped and mixed together in the center.

[tsne-balanced]: tsne-tfidf-balance500.png
![tsne-balanced]

## Communication
### Presentations
I have given more than two presentations. I do not remember which ones exactly. I do remember the following ones:
- Technical presentation about NLP ([slides](https://docs.google.com/presentation/d/1rSYB52LkQphUzZceA-eIvbsHqV2y7bYZiyG5Nq0XZLM/edit?usp=sharing))
- Preperation of final presentation (10th of januari) ([slides](https://drive.google.com/file/d/181d7qLdFrMkpcqN7AoUJOiJLNOWKkO65/view?usp=sharing))
- 

### Writing paper
I contributed mostly to the following paragraphs of the paper:
- Abstract
- Introduction
- 2.4 Pipeline
- 2.6 Preprocessing
- 2.7 Vectorization
- 2.11 Experiment Setup
- 3.1 Agora pipeline comparison
- 3.2 Web-IQ pipeling comparison
- 4 Conclusion
- 5 Discussion

Apart from these chapters I have reviewed the entire paper multiple times and left comments where I felt changes or a discussion was necessary.
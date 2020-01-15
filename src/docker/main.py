#!/usr/bin/env python3
from flask import Flask, render_template,request,url_for, jsonify

from sklearn.svm import *
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import random 
import time
import numpy as np
import pickle

app = Flask(__name__)

def load_model(fn):
	with open(fn, "rb") as f:
		m = pickle.load(f)
	return m

model = load_model("models/svc-model-webiq.pkl")
tfidf = load_model("models/tfidf_model.pkl")

@app.route('/predict_class', methods=['POST'])
def rank():
	global model
	if not request.json or not 'posts_list' in request.json:
		abort(400)

	posts = request.json['posts_list']
	posts = posts.split('\n')

	print(posts)

	v = tfidf.transform(posts)
	pred = model.predict(v).tolist()

	response = {
		"input" : posts,
		"output" : pred
	}

	return jsonify(response), 200

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5050)

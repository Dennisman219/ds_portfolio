#!/usr/bin/env python3

import pickle

with open('data/tno/categorieen.pkl', 'rb') as file:
	var = pickle.load(file)

print(var)
print(type(var))
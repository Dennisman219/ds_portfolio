#!/usr/bin/env python
class prepper:
	def __init__(self, **kwargs):
		import pandas as pd, sys, string, math, time

		self.catocur_max = catocur_max
		self.primcat = primcat
		self.lemmatize = lemmatize
		self.stem = stem
		self.nrows = nrows

		if lemmatize:
			from nltk.stem import WordNetLemmatizer
			self.lemmer = WordNetLemmatizer() 
		if stem:
			from nltk.stem.snowball import SnowballStemmer
			self.stemmer = SnowballStemmer("english")

		if srows is not None:
			self.agora = pd.read_csv('data/agorb.csv', nrows=self.nrows, skiprows=range(1, srows), error_bad_lines=False, warn_bad_lines=False)
		else:
			self.agora = pd.read_csv('data/agorb.csv', nrows=self.nrows, error_bad_lines=False, warn_bad_lines=False)

		self.stop_words = set(open('data/stop_words.txt', 'r').read().split('\n'))

		#self.agora = self.agora[self.agora.groupby(' Category')[' Category'].transform(len) > 1]
		#self.agora = self.agora.dropna
		self.data = self.agora[' Item'] + " " + self.agora[' Item Description']

	def get_categories(self):
		cats = []
		for c in self.agora[' Category']:
			if self.primcat:
				cats.append(c.lower().split('/')[0])
			else:
				cats.append(c.lower())
		return cats

	def process(self):
		processed = []
		#categories = []
		catocur = {}
		t = 0
		s = time.time()
		try:
			for row in self.data:
				if self.primcat:
					c = self.agora[' Category'][t].lower().split('/')[0]
				else:
					c = self.agora[' Category'][t].lower()

				if self.catocur_max > 0: #TODO catocur min?
					try:
						if catocur[c] >= self.catocur_max:
							t += 1
							continue
						else:
							catocur[c] += 1
					except KeyError:
						catocur[c] = 1

				row = str(row).lower()
				row = row.translate(str.maketrans('', '', '{}1234567890'.format(string.punctuation))) # Remove punctuation and numbers
				words = row.split(' ')
				words = list(filter(None, words)) # Remove empty strings

				#TODO ngrams

				i = 0
				for w in words:
					if self.lemmatize:
						lemmed = self.lemmer.lemmatize(w) # Lemmatize word
						words[i] = lemmed
						w = lemmed
						del lemmed
					if self.stem:
						stemmed = self.stemmer.stem(w) # Stem word
						words[i] = stemmed
						w = stemmed
						del stemmed

					if w in self.stop_words:
						# Remove occurance of word
						del words[i]

					i += 1
				del i

				d = ' '.join(words)
				processed.append('{},{}'.format(c,d))
				del d

				t += 1

		except KeyboardInterrupt:
			pass

		e = time.time()
		return processed

	def __str__(self):
		return "prepper(catocur_max={}, primcat={}, lemmatize={}, stem={})".format(self.catocur_max, self.primcat, self.lemmatize, self.stem)

if __name__ == "__main__":
	import sys, time, string
	from pathlib import Path

	nrows = None
	srows = None
	catocur_max = 0
	#debug = False
	lemmatize = False
	stem = False
	force_process = False
	primcat = False
	no_pipe = False
	oi = 2
	for opt in sys.argv[1:]:
		if opt == '--nrows':
			nrows = int(sys.argv[oi])
		elif opt == '--srows':
			srows = int(sys.argv[oi])
		elif opt == '--balance':
			catocur_max = int(sys.argv[oi])
		#elif opt == '-d':
		#	debug = True
		elif opt == '--refresh':
			force_process = True
		elif opt == '--primary':
			primcat = True
		elif opt == '--lemmatize':
			lemmatize = True
		elif opt == '--stem':
			stem = True
		elif opt == '--nopipe':
			no_pipe = True
			
		oi += 1
	del oi

	"""
	save = False
	processed_file = Path('data/processed.csv')
	if processed_file.is_file():
		source_file = Path('data/agorb.csv')
		if processed_file.stat().st_ctime < source_file.stat().st_mtime:
			save = True
	else:
		save = True
	"""

	do_process = ( not Path('data/processed.csv').is_file() or force_process ) or ( Path('data/processed.csv').stat().st_ctime < Path('data/agorb.csv').stat().st_mtime )
	#sys.stdout.write('pfile: {}\nsfile: {}\ndo_process: {}\n'.format(Path('data/processed.csv').stat().st_ctime, Path('data/agorb.csv').stat().st_mtime, do_process))

	#TODO preprocessing parameters in output file
	if do_process:
		p = prepper(nrows=nrows, srows=srows, catocur_max=catocur_max, lemmatize=lemmatize, stem=stem, primcat=primcat)
		#print(p)
		processed = p.process()
		
		import numpy as np
		train = np.array(processed)

		if Path('data/processed.csv').is_file():
				Path('data/processed.csv').unlink()
		np.savetxt('data/processed.csv', train, delimiter=',', fmt='%s')

		if not no_pipe:
			for l in processed:
				sys.stdout.write("{}\n".format(l))
				sys.stdout.flush()
		else:
			sys.stdout.write('Total lines processed: {}\n'.format(len(processed)))
	else:
		import csv
		with open('data/processed.csv', 'rt') as csvfile:
			reader = csv.reader(csvfile)
			if not no_pipe:
				for row in reader:
					sys.stdout.write('{},{}\n'.format(row[0], row[1]))
					sys.stdout.flush()
			else:
				sys.stdout.write('Data already available in \'data/processed.csv\' ({} lines)\n'.format(len(csvfile.readlines())))


	#print(len(processed))
	#print(len(p.get_categories()))

"""
	if debug:
		sys.stdout.write('reduction:\t{} -> {} ({}%)\n'.format(b, a, math.ceil((b-a)/b*100)))
		#sys.stdout.write('performance:\t{} rows in {}s ({}rows/s)\n'.format(t, round(e-s, 2), round(t/(e-s), 2)))
	else:
		sys.stdout.write('{},{}\n'.format(c, repr(' '.join(words).rstrip('\n'))))
		sys.stdout.flush()

#agora = agora[agora.groupby(' Category')[' Category'].transform(len) > 1]
#stop_words = open('data/stop_words.txt', 'r').read().splitlines()

#data = []
#for i in range(agora.shape[0]):
#	data.append('{} {}'.format(agora[' Item'][i], agora[' Item Description'][i]))
"""
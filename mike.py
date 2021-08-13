import nltk
from nltk.stem.lancaster import LancasterStemmer as LS
import numpy 
import tensorflow
import json
import random 
import pickle
stemmer = LS()

#nltk.download('punkt')

with open ('mind.json') as mind:
	data = json.load(mind)


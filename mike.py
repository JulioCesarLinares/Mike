import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
import numpy as np
import tensorflow
import json
import tflearn
import random
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

recortador = LancasterStemmer()

with open ('mind.json') as mente:
	palabras_conocidas = json.load(mente)

palabras = []
grupos = []
tokens = []
copia_tokens = []
copia_grupos = []
caract_ignorados = ['?','¿','!','¡']

for grupo in palabras_conocidas["datos"]:
	for posibilidad in grupo["posibilidades"]:
		tokens = nltk.word_tokenize(posibilidad)  #Divide las palabras
		palabras.extend(tokens)
		copia_tokens.append(tokens)
		copia_grupos.append(grupo["tipo"])
	if grupo["tipo"] not in grupos:
		grupos.append(grupo["tipo"])

palabras = [recortador.stem(word.lower()) for word in palabras if word not in
		caract_ignorados]
palabras = sorted(palabras)
grupos = sorted(grupos)

#print("Este es copia_tokens: {}\n".format(copia_tokens))
#print("Este es copia_grupos: {}\n".format(copia_grupos))
#print("Este es tokens: {}\n".format(tokens))
#print("Este es palabras: {}\n".format(palabras))

entrenamiento = []
salida = []
salida_vacia = [0 for a in range(len(grupos))]

for n, tok in enumerate(copia_tokens):
	matrix = []
	tokens =  [recortador.stem(w.lower()) for w in tok]
	for w in palabras:
		if w in tokens:
			matrix.append(1)
		else:
			matrix.append(0)
	filaSalida = salida_vacia[:]
	filaSalida[grupos.index(copia_grupos[n])] = 1
	entrenamiento.append(matrix)
	salida.append(filaSalida)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

entrenamiento = np.array(entrenamiento)
salida = np.array(salida)

ops.reset_default_graph()

modelo = tflearn.input_data(shape = [None,len(entrenamiento[0])])
modelo = tflearn.fully_connected(modelo,10)
modelo = tflearn.fully_connected(modelo,10)
modelo = tflearn.fully_connected(modelo,len(salida[0]),activation='softmax')
modelo = tflearn.regression(modelo) #,optimizer='sdg',loss='categorical_crossentropy')
modelo = tflearn.DNN(modelo)

#model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

his = modelo.fit(entrenamiento,salida, n_epoch = 1000, batch_size = 10, show_metric=False)
modelo.save("modelo.tflearn")

def run():
	while True:
		entrada = input("Habla tú: ")
		matrix = [0 for ceros in range(len(palabras))]
		ProcessingEntrada = nltk.word_tokenize(entrada)
		ProcessingEntrada = [recortador.stem(wo.lower()) for wo in ProcessingEntrada]
		for pala in ProcessingEntrada:
			for i,pal in enumerate(palabras):
				if pala == pal:
					matrix[i]=1
		resultados = modelo.predict([np.array(matrix)])
		IndicesResultados = np.argmax(resultados)
		grupo = grupos[IndicesResultados]
		for group in palabras_conocidas["datos"]:
			if group["tipo"] == grupo:
				respuesta = group["respuestas"]
		print(">: ",random.choice(respuesta))
run()
print("Hola, me llamo Mike, como te llamas? ",'\n')


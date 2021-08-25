import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
import numpy as np
import tensorflow
import json
import tflearn
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

recortador = LancasterStemmer()

with open ('mind.json') as mente:
	palabras_conocidas = json.load(mente)

palabras = []
grupos = []
tokens = []
copia_palabras = []
copia_grupos = []
caract_ignorados = ['?','¿','!','¡']

for grupo in palabras_conocidas["datos"]:
	for posibilidad in grupo["posibilidades"]:
		tokens = nltk.word_tokenize(posibilidad) 
		palabras.extend(tokens)
		copia_palabras.append(tokens)
		copia_grupos.append(grupo["tipo"])
	if grupo["tipo"] not in grupos:
		grupos.append(grupo["tipo"])

palabras = [recortador.stem(word.lower()) for word in palabras if word not in
		caract_ignorados]
palabras = sorted(list(set(palabras)))
grupos = sorted(grupos)

print("Este es copia_palabras: {}\n".format(copia_palabras))
print("Este es copia_grupos: {}\n".format(copia_grupos))
print("Este es tokens: {}\n".format(tokens))
print("Este es palabras: {}\n".format(palabras))
print("Este es grupos: {}\n".format(grupos))

entrada = []
salida = []
salida_vacia = [0 for a in range(len(grupos))]

for n, pal in enumerate(copia_palabras):
	matrix = []
	tokens =  [recortador.stem(w.lower()) for w in pal]
	print("Este es tokens", tokens)
	for palab in palabras:
		if palab in tokens:
			matrix.append(1)
		else:
			matrix.append(0)
	filaSalida = salida_vacia[:]
	filaSalida[grupos.index(copia_grupos[n])] = 1
	entrada.append(matrix)
	salida.append(filaSalida)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
print("este es entre:",entrada)
entrada = np.array(entrada) #Crear la matriz
salida = np.array(salida)
print("entre",len(entrada[0]))
print("salida",salida)

ops.reset_default_graph() #reiniciar red neuronal, dejarla en limpio

red = tflearn.input_data(shape = [None,len(entrada[0])]) #forma de datos de entrada 
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]),activation='softmax') #Forma de datos de salida, limitado por el numero de grupos, función de activación
red = tflearn.regression(red) #Nos permite saber valores de las probabilidades 
modelo = tflearn.DNN(red) 

modelo.fit(entrada,salida, n_epoch = 1000, batch_size = 70, show_metric=True)
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
		if IndicesResultados < 0.1:
			print(">: No te entendí")
		else:
			for group in palabras_conocidas["datos"]:
				if group["tipo"] == grupo:
					respuesta = group["respuestas"]
			print(">: ",random.choice(respuesta))
run()


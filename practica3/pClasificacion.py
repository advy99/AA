# -*- coding: utf-8 -*-
"""
Práctica 3 - Problema de clasificación.
Nombre Estudiante: Antonio David Villegas
"""


import numpy as np
import matplotlib.pyplot as plt

# pandas para leer el csv
import pandas as pd

# los modelos a usar
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

# para hacer validación cruzada
from sklearn.model_selection import cross_val_score

# metricas
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

np.random.seed(1)

def leer_datos(fichero):

	datos = pd.read_csv(fichero, header=None)

	valores = datos.values

	# los datos son todas las filas y todas las columnas menos la última
	x = valores[:, :-1]

	# la última columna son las etiquetas
	y = valores[:, -1]

	return x, y


def evaluar(clasificador, x, y, x_test, y_test, cv):
	# ajustamos el modelo
	print("Ajustando el modelo")
	clasificador.fit(x,y)
	# aplicamos validación cruzada
	print("Aplicando 10-k-folds cross-validation")
	resultado_regresion_lineal = cross_val_score(clasificador, x, y, scoring='neg_mean_squared_error', cv=10)

	# predecimos training acorde al modelo
	print("Prediciendo las etiquetas dentro de training")
	y_predecida = clasificador.predict(x).round()
	# miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
	print("Obteniendo E_in a partir de la predicción")
	aciertos = accuracy_score(y, y_predecida)
	print("Porcentaje de aciertos usando en training", aciertos)
	print("E_in: ", 1-aciertos)


	print("Prediciendo las etiquetas en test")
	# predecimos test acorde al modelo
	y_predecida_test = clasificador.predict(x_test).round()
	# miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
	print("Obteniendo E_test a partir de la predicción")
	aciertos = accuracy_score(y_test, y_predecida_test)
	print("Porcentaje de aciertos en test: ", aciertos)
	print("E_test: ", 1-aciertos)


	print("\n\n")
	input("\n--- Pulsar tecla para continuar ---\n")
	print("\n\n")




datos_tra = "datos/optdigits.tra"

print("Leyendo datos de", datos_tra )
x, y = leer_datos(datos_tra)
print("Leidos ", y.size, " datos con sus respectivas etiquetas")

print("Cada dato del conjunto de datos tiene ", x[0].size, " variables.")

input("\n--- Pulsar tecla para continuar ---\n")

# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
unique, contador = np.unique(y, return_counts=True)

# print(contador)
# print(np.arange(10))


# mostramos el número de elementos de cada clase
plt.title("Recuento de los valores del conjunto de training")
plt.ylabel("Número de elementos de cada clase")
plt.xlabel("Posibles clases (0, ..., 9)")
plt.xticks(np.arange(10))
plt.bar(np.arange(10), contador, width=0.4)

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")



# lectura de test
datos_test = "datos/optdigits.tes"

print("Leyendo datos de", datos_test )
x_test, y_test = leer_datos(datos_test)
print("Leidos ", y_test.size, " datos con sus respectivas etiquetas")

print("Cada dato del conjunto de datos tiene ", x_test[0].size, " variables.")
input("\n--- Pulsar tecla para continuar ---\n")

# https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
unique, contador = np.unique(y_test, return_counts=True)

# print(contador)
# print(np.arange(10))


# mostramos el número de elementos de cada clase
plt.title("Recuento de los valores del conjunto de training")
plt.ylabel("Número de elementos de cada clase")
plt.xlabel("Posibles clases (0, ..., 9)")
plt.xticks(np.arange(10))
plt.bar(np.arange(10), contador, width=0.4)

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")


ridgeC = RidgeClassifier()
print("Evaluando Ridge Classifier:")
evaluar(ridgeC, x, y, x_test, y_test, 10)


SGD = SGDClassifier()
print("Evaluando SGD Classifier:")
evaluar(SGD, x, y, x_test, y_test, 10)

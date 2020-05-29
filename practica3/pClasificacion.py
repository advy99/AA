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

# para contar tiempos
import time

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
	inicio = time.time();
	print("\tAjustando el modelo")
	clasificador.fit(x,y)
	# aplicamos validación cruzada una vez tenemos el modelo ajustado
	print("\tAplicando", cv, "-k-folds cross-validation")
	resultado_cross_val = cross_val_score(clasificador, x, y, scoring='accuracy', cv=cv)
	fin = time.time()
	print("\tTiempo (en segundos) necesario para ajustar el modelo y aplicar cross-validation: ", fin-inicio)
	print("\n")

	# predecimos training acorde al modelo
	print("\tPrediciendo las etiquetas dentro de training")
	y_predecida = clasificador.predict(x).round()
	# miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
	print("\tObteniendo E_in a partir de la predicción")
	aciertos = accuracy_score(y, y_predecida)
	print("\tPorcentaje de aciertos usando en training", aciertos)
	print("\tE_in: ", 1-aciertos)


	print("\tPrediciendo las etiquetas en test")
	# predecimos test acorde al modelo
	y_predecida_test = clasificador.predict(x_test).round()
	# miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
	print("\tObteniendo E_test a partir de la predicción")
	aciertos = accuracy_score(y_test, y_predecida_test)
	print("\tPorcentaje de aciertos en test: ", aciertos)
	print("\tE_test: ", 1-aciertos)

	print("\tEvaluación de aciertos usando cross-validation: ", resultado_cross_val.mean())


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





# evaluaciones de los modelos
# la primera linea declaramos el objeto de Scikit-Learn para ajustar los parametros
# y hecho esto llamamos a la función evaluar con el modelo, los datos y
# el número de k-folds para cross-validation


ridgeCsinReg = RidgeClassifier(alpha=0)
print("Evaluando Ridge Classifier sin regularización:")
evaluar(ridgeCsinReg, x, y, x_test, y_test, 10)


ridgeCReg = RidgeClassifier(alpha=1)
print("Evaluando Ridge Classifier con regularización 1 (por defecto):")
evaluar(ridgeCReg, x, y, x_test, y_test, 10)

ridgeCReg01 = RidgeClassifier(alpha=0.1)
print("Evaluando Ridge Classifier con regularización 0.1:")
evaluar(ridgeCReg01, x, y, x_test, y_test, 10)

ridgeCReg2 = RidgeClassifier(alpha=2)
print("Evaluando Ridge Classifier con regularización 2:")
evaluar(ridgeCReg2, x, y, x_test, y_test, 10)




# logiticReg = LogisticRegression(penalty='l2', solver='newton-cg', C=1)
# print("Evaluando Regresión Logística:")
# evaluar(logiticReg, x, y, x_test, y_test, 10)
#
# logiticReg0 = LogisticRegression(penalty='l2', solver='newton-cg', C=1)
# print("Evaluando Regresión Logística sin regularización:")
# evaluar(logiticReg0, x, y, x_test, y_test, 10)
#


SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante 0.01:")
evaluar(SGD, x, y, x_test, y_test, 10)


SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.0001:")
evaluar(SGD, x, y, x_test, y_test, 10)


SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.01:")
evaluar(SGD, x, y, x_test, y_test, 10)

SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.1:")
evaluar(SGD, x, y, x_test, y_test, 10)

SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 1:")
evaluar(SGD, x, y, x_test, y_test, 10)



SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante 0.01 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, 10)


SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.0001 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, 10)


SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.01 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, 10)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, 10)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, 10)

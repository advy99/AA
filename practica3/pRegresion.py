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
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

# para hacer validación cruzada
from sklearn.model_selection import cross_val_score

# metricas
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# para contar tiempos
import time

# para dividir train y test
from sklearn.model_selection import train_test_split

np.random.seed(1)



def preprocesar(x, porcentaje_perdidos_aceptado):

	a_eliminar = []
	for i in range(0, x[0].size):
		contador = 0
		valor_medio = 0
		for val in x[:, i]:
			# https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
			if np.isnan(val):
				contador += 1
			else:
				valor_medio += val

		valor_medio = valor_medio/x[:, 0].size


		if contador != 0:
			print(contador, " datos tienen el atributo ", i, " perdido " )

		if contador > (x[:, 0].size * porcentaje_perdidos_aceptado):
			print("El atributo ", i, " tiene más de un ", porcentaje_perdidos_aceptado*100, " por ciento de datos perdidos, luego lo eliminamos")
			a_eliminar.append(i)
		elif contador != 0:
			print("El atributo ", i, " no esta perdido en más de un ", porcentaje_perdidos_aceptado*100, " por ciento. Los valores perdidos se sustituirán por la suma del valor medio.")
			for j in range(0, x[:, i].size):
				if np.isnan(x[:, i][j]):
					x[:, i][j] = valor_medio


	x = np.delete(x, a_eliminar, 1)

	return x


def leer_datos(fichero, proporcion_test):

	# leemos los datos, na_values para que los interprete como Not A Number (NaN)
	datos = pd.read_csv(fichero, header=None, na_values='?')

	valores = datos.values

	# los datos son todas las filas y todas las columnas menos la última
	# y las 5 primeras, al no servir para la predicción
	x = valores[:, 5:-1]

	# la última columna son las etiquetas
	y = valores[:, -1]

	x = preprocesar(x.copy(), 0.2)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=proporcion_test)

	return x_train, x_test, y_train, y_test


def evaluar(clasificador, x, y, x_test, y_test, cv):
	# ajustamos el modelo
	inicio = time.time();
	print("\tAjustando el modelo")
	clasificador.fit(x,y)
	# aplicamos validación cruzada una vez tenemos el modelo ajustado
	print("\tAplicando", cv, "-k-folds cross-validation")
	# usamos de metrica la tasa de aciertos (accuracy)
	resultado_cross_val = cross_val_score(clasificador, x, y, scoring='neg_mean_squared_error', cv=cv)
	fin = time.time()
	print("\tTiempo (en segundos) necesario para ajustar el modelo y aplicar cross-validation: ", fin-inicio)
	print("\n")

	print("\tEvaluación media de aciertos usando cross-validation: ", resultado_cross_val.mean())
	print("\tE_in usando cross-validation: ", 1-resultado_cross_val.mean())



	print("\tPrediciendo las etiquetas en test")
	# predecimos test acorde al modelo
	y_predecida_test = clasificador.predict(x_test).round()
	# miramos la tasa de aciertos, es decir, cuantos ha clasificado bien
	print("\tObteniendo E_test a partir de la predicción")
	aciertos = mean_absolute_error(y_test, y_predecida_test)
	print("\tPorcentaje de aciertos en test: ", aciertos)
	print("\tE_test: ", 1-aciertos)


	print("\n\n")
	input("\n--- Pulsar tecla para continuar ---\n")
	print("\n\n")

	# devolvemos E_test
	return y_predecida_test




datos = "datos/communities.data"

print("Leyendo datos de", datos )
x, x_test, y, y_test = leer_datos(datos, 0.2)
print("Leidos ", y.size, " datos con sus respectivas etiquetas")

print("Cada dato del conjunto de datos tiene ", x[0].size, " variables.")

input("\n--- Pulsar tecla para continuar ---\n")

# print(contador)
# print(np.arange(10))

#
# # mostramos el número de elementos de cada clase
# plt.title("Recuento de los valores del conjunto de training")
# plt.ylabel("Número de elementos de cada clase")
# plt.xlabel("Posibles clases (0, ..., 9)")
# plt.xticks(np.arange(10))
# plt.scatter(np.arange(10), contador, width=0.4)
#
# plt.show()
# input("\n--- Pulsar tecla para continuar ---\n")
#
#
#
#
# # https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
# unique, contador = np.unique(y_test, return_counts=True)
#
# # print(contador)
# # print(np.arange(10))
#
#
# # mostramos el número de elementos de cada clase
# plt.title("Recuento de los valores del conjunto de training")
# plt.ylabel("Número de elementos de cada clase")
# plt.xlabel("Posibles clases (0, ..., 9)")
# plt.xticks(np.arange(10))
# plt.bar(np.arange(10), contador, width=0.4)
#
# plt.show()
# input("\n--- Pulsar tecla para continuar ---\n")





# evaluaciones de los modelos
# la primera linea declaramos el objeto de Scikit-Learn para ajustar los parametros
# y hecho esto llamamos a la función evaluar con el modelo, los datos y
# el número de k-folds para cross-validation

kfolds = 10


ridgeCsinReg = Ridge(alpha=0)
print("Evaluando Ridge Regression con factor de regularización 0 (sin regularización):")
evaluar(ridgeCsinReg, x, y, x_test, y_test, kfolds)

ridgeCReg01 = Ridge(alpha=0.1)
print("Evaluando Ridge Regression con factor de regularización 0.1:")
evaluar(ridgeCReg01, x, y, x_test, y_test, kfolds)

ridgeCReg = Ridge(alpha=1)
print("Evaluando Ridge Regression con factor de regularización 1 (por defecto):")
y_pred_test = evaluar(ridgeCReg, x, y, x_test, y_test, kfolds)


ridgeCReg2 = Ridge(alpha=2)
print("Evaluando Ridge Regression con factor de regularización 2:")
evaluar(ridgeCReg2, x, y, x_test, y_test, kfolds)




logiticReg = LogisticRegression(penalty='l2', solver='newton-cg', C=0.001, multi_class='multinomial')
print("Evaluando Regresión Logística con peso de regularización 0.001:")
evaluar(logiticReg, x, y, x_test, y_test, kfolds)

logiticReg0 = LogisticRegression(penalty='l2', solver='newton-cg', C=1, multi_class='multinomial')
print("Evaluando Regresión Logística con peso de regularización 1:")
evaluar(logiticReg0, x, y, x_test, y_test, kfolds)

logiticReg0 = LogisticRegression(penalty='l2', solver='newton-cg', C=2, multi_class='multinomial')
print("Evaluando Regresión Logística con peso de regularización 2:")
evaluar(logiticReg0, x, y, x_test, y_test, kfolds)



SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.0001 y funcion de perdida square loss:")
evaluar(SGD, x, y, x_test, y_test, kfolds)


SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.01 y funcion de perdida square loss:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.1 y funcion de perdida square loss:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='squared_loss', penalty='l2', alpha=1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 1 y funcion de perdida square loss:")
evaluar(SGD, x, y, x_test, y_test, kfolds)



SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante 0.01 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

# no podemos evaluar usando un esquema constante con alpha = 0


SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.0001 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante y factor de regularización 0.0001 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)




SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.01 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante y factor de regularización 0.01 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)




SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 0.1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.1, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante y factor de regularización 0.1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)




SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=1, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje optima (variable) y factor de regularización 1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=1, learning_rate='constant', eta0=0.01, max_iter=5000)
print("Evaluando SGD Classifier con tasa de aprendizaje constante y factor de regularización 1 y función de perdida hinge:")
evaluar(SGD, x, y, x_test, y_test, kfolds)




print("Matriz de confusión con y_test y los valores predecidos: ")
print(confusion_matrix(y_test, y_pred_test))

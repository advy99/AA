# -*- coding: utf-8 -*-
"""
Práctica 3 - Problema de clasificación.
Nombre Estudiante: Antonio David Villegas
"""


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

np.random.seed(1)

def leer_datos(fichero):

	datos = pd.read_csv(fichero, header=None)

	valores = datos.values

	# los datos son todas las filas y todas las columnas menos la última
	x = valores[:, :-1]

	# la última columna son las etiquetas
	y = valores[:, -1]

	return x, y







print("Leyendo datos de la carpeta datos/")
x, y = leer_datos("datos/optdigits.tra")


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

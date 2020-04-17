# -*- coding: utf-8 -*-
"""
PRACTICA 2 - EJ 2
Nombre Estudiante: Antonio David Villegas Yeguas
"""
import numpy as np
import matplotlib.pyplot as plt
from sympy import Add
from sympy.solvers import solve
from sympy import Symbol


# Fijamos la semilla
#np.random.seed()

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON


# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)


def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
	w = np.copy(vini)
	mejora = True
	iteraciones = 0

	while mejora and iteraciones < max_iter:
		mejora = False

		for i in range(0, len(datos)):
			valor = signo(w.T.dot(datos[i]))

			if valor != label[i]:
				w = w + label[i] * datos[i]
				mejora = True

		iteraciones += 1

	return w, iteraciones

#CODIGO DEL ESTUDIANTE

intervalo_trabajo = [-50, 50]

puntos_2d = simula_unif(100, 2, intervalo_trabajo)

a, b = simula_recta(intervalo_trabajo)

etiquetas = []
posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

for punto in puntos_2d:
	etiquetas.append(f(punto[0], punto[1], a, b))

for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)

	plt.scatter(puntos_2d[indice, 0], puntos_2d[indice, 1], c=colores[etiqueta], label="{}".format(etiqueta))



# le metemos unos al principio para que concuerde
puntos_2d = np.c_[np.ones((puntos_2d.shape[0], 1), dtype=np.float64), puntos_2d]

w_0 = np.zeros(3)

# no queremos tener limite de iteraciones, las ponemos a infinito
w, iteraciones = ajusta_PLA(puntos_2d, etiquetas, np.Inf, w_0)

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta con la que etiquetamos')

# obtenemos 0 = w_0 + w_1 * x1 + w_2 * x_2

# x_2 = (-w_0 - w_1 * x_1 )/ w_2


plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con PLA')


plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {}, etiquetados segun una recta".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

print("W obtenida: " + str(w) + "\n Iteraciones: " + str(iteraciones))

input("\n--- Pulsar tecla para continuar ---\n")





# Random initializations
iterations = []
for i in range(0,10):
	w_0 = simula_unif(3, 1, [0, 1]).reshape(1, -1)[0]
	w, iteraciones = ajusta_PLA(puntos_2d, etiquetas, np.Inf, w_0)
	iterations.append(iteraciones)
    #CODIGO DEL ESTUDIANTE

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


"""
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL(?):
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")



# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM

#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
"""

# -*- coding: utf-8 -*-
"""
PRACTICA 2 - EJ 3 - BONUS
Nombre Estudiante: Antonio David Villegas Yeguas
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


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



# Funcion para calcular el error
def Err(x,y,w):
	## Función dada en las diapositivas:
	#
	# E_in(w) = 1/N * SUM(w^T * x_n - y_n)^2
	#
	# basicamente, hacemos w * x - y al cuadrado, y le hacemos la media

	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html
	# cambiamos la forma de y, ya que como vemos en las diapositivas, y tiene una columna
	# y muchas filas, pero aqui tenemos una fila y muchas columnas
	err = np.square(x.dot(w) - y.reshape(-1,1))

	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
	err.mean()

	return err

def dErr(x, y, w):
	#derivada como hemos visto en teoria
	# la parte interna sigue igual
	derivada = x.dot(w) - y.reshape(-1,1)

	# el cuadrado pasa a ser un *2, y restamos una al exponente
	derivada = 2 * np.mean(x * derivada, axis=0)

	derivada = derivada.reshape(-1, 1)

	return derivada

# Gradiente Descendente Estocastico
def sgd(x, y, tasa_aprendizaje, tam_batch, maxIteraciones = 4000):
    #
	# diapositiva 17, jussto antes del metodo de newton

	# el tamaño de w será dependiendo del numero de elementos de x
	w = np.zeros((x.shape[1], 1), np.float64)

	iterations = 0

	nueva_epoca = False
	indice_actual = 0

	indices_minibatch = np.random.choice(x.shape[0], x.shape[0], replace=False)

	# en este caso solo tenemos de condicion las iteraciones
	while iterations < maxIteraciones:

		iterations = iterations + 1

		if nueva_epoca:
			nueva_epoca = False
			indices_minibatch = np.random.choice(x.shape[0], x.shape[0], replace=False)

		if indice_actual+tam_batch < x.shape[0]:
			minibatch = indices_minibatch[indice_actual:indice_actual+tam_batch]
			indice_actual += tam_batch
		else:
			minibatch = indices_minibatch[indice_actual:x.shape[0]]
			indice_actual = 0
			nueva_epoca = True


		w = w - tasa_aprendizaje * dErr(x[minibatch], y[minibatch], w)
		iterations += 1



	return w, iterations


# Pseudoinversa
def pseudoinverse(matriz_x, vector_y):
    #
	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
	x_traspuesta = matriz_x.transpose()

	# cambiamos de forma Y, no sabemos cuantas filas tendrá, pero tendrá una única columna
	y_traspuesto = vector_y.reshape(-1, 1)

	# multiplicamos x por su traspuesta https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
	x_pseudoinversa = x_traspuesta.dot(matriz_x)

	# calculamos la inversa: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
	x_pseudoinversa = np.linalg.inv(x_pseudoinversa)

	x_pseudoinversa = x_pseudoinversa.dot(x_traspuesta)

	# finalmente calculamos w con la inversa e y
	w = x_pseudoinversa.dot(y_traspuesto)

	return w


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

input("\n--- Pulsar tecla para continuar ---\n")


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

tasa_aprendizaje = 0.01

intervalo_trabajo = [0, 1]
w, iteraciones = sgd(x, y, tasa_aprendizaje, 32, 10000)

fig, ax = plt.subplots()

ax.plot()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con sgd')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7, 0])
plt.legend()
plt.show()

print("Error obtenido usando sgd dentro de la muestra (Ein): ", Err(x, y, w).mean())
print("Error obtenido usando sgd para los datos de test (Etest): ", Err(x_test, y_test, w).mean())


input("\n--- Pulsar tecla para continuar ---\n")


intervalo_trabajo = [0, 1]
w = pseudoinverse(x, y)

fig, ax = plt.subplots()

ax.plot()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con pseudoinverse')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7, 0])
plt.legend()
plt.show()

print("Error obtenido usando pseudoinverse dentro de la muestra (Ein): ", Err(x, y, w).mean())
print("Error obtenido usando pseudoinverse para los datos de test (Etest): ", Err(x_test, y_test, w).mean())


input("\n--- Pulsar tecla para continuar ---\n")

def signo(x):
	if x >= 0:
		return 1
	return -1


def num_errores_puntos(x, y, w):

	errores = 0
	for i in range(len(x)):
		if signo(w.T.dot(x[i])) != y[i]:
			errores += 1

	errores = errores/len(x)

	return errores

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))





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
				w = w + label[i] * datos[i].reshape(-1, 1)
				mejora = True

		iteraciones += 1

	return w, iteraciones


#POCKET ALGORITHM
def pocket(x, y, iteraciones, w_ini):

	# en el ej 2 nos daba mejores resultados si el comienzo lo haciamos aleatorio entre [0, 1]
	#w_mejor = simula_unif(3, 1, [0, 1]).reshape(1, -1)[0]
	w_mejor = w_ini.copy()

	ein_w_mejor = num_errores_puntos(x, y, w_mejor)
	w = w_mejor.copy()

	it = 0

	while it < iteraciones:
		w, basura = ajusta_PLA(x, y, 1, w.copy())
		ein_w = num_errores_puntos(x, y, w)
		if ein_w < ein_w_mejor:
			w_mejor = w.copy()
			ein_w_mejor = ein_w

		it += 1

	return w_mejor

#CODIGO DEL ESTUDIANTE
#w, iteraciones = sgd(x, y, tasa_aprendizaje, 32, 10000)

print("Esto va a tardar un poco, python hace lo que puede (o eso dice, porque no me creo que este lenguaje este sobre C)")
w = pocket(x, y, 10000, w.copy())

fig, ax = plt.subplots()

ax.plot()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con pocket')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7, 0])
plt.legend()
plt.show()

print("Error obtenido usando pocket dentro de la muestra (Ein): ", num_errores_puntos(x, y, w))

input("\n--- Pulsar tecla para continuar ---\n")


fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con pocket')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.xlim(intervalo_trabajo)
plt.ylim([-7, 0])
plt.legend()
plt.show()


print("Error obtenido usando pocket para los datos de test (Etest): ", num_errores_puntos(x_test, y_test, w))


input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE

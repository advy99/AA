#  Antonio David Villegas Yeguas
#


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
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
	err = np.square(x.dot(w) - y.reshape(-1,1))

	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
	err.mean()

	return err

def dErr(x, y, w):
	# la parte interna sigue igual
	derivada = x.dot(w) - y.reshape(-1,1)

	# el cuadrado pasa a ser un *2, y restamos una al exponente
	derivada = 2 * np.mean(x * derivada)

	return derivada

# Gradiente Descendente Estocastico
def sgd(x, y, tasa_aprendizaje, tam_batch, maxIteraciones):
    #
	# diapositiva 17, jussto antes del metodo de newton

	w = np.zeros((3, 1), np.float64)

	iterations = 0

	while iterations < maxIteraciones:

		iterations = iterations + 1

		# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
		# sacamos tantos indices aleatorios como tam_batch (tamaño del minibatch) queramos
		# lo sacamos a partir de la forma de x, es decir, x tendrá N filas y 1 columna
		# pues escogemos de entre las N filas tam_batch
		indices_minibatch = np.random.choice(x.shape[0], tam_batch, replace=False)

		# aplicamos la funcion dada en la diapositiva 17 del tema 1
		w = w - tasa_aprendizaje * dErr(x[indices_minibatch], y[indices_minibatch], w)

	return w, iterations

# Pseudoinversa
def pseudoinverse(matriz_x, vector_y):
    #

	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
	x_traspuesta = matriz_x.transpose()

	# cambiamos de forma Y, no sabemos cuantas filas tendrá, pero tendrá una única columna
	y_traspuesto = y.reshape(-1, 1)

	# multiplicamos x por su traspuesta https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
	x_pseudoinversa = x_traspuesta.dot(matriz_x)

	# calculamos la inversa: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
	x_pseudoinversa = np.linalg.inv(x_pseudoinversa)

	x_pseudoinversa = x_pseudoinversa.dot(x_traspuesta)

	# finalmente calculamos w con la inversa e y
	w = x_pseudoinversa.dot(y_traspuesto)

	return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w, iteraciones = sgd(x, y, 0.1, 64, 100)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

w = pseudoinverse(x, y)
print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")


print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...

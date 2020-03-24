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
	derivada = 2 * np.mean(x * derivada, axis=0)

	derivada = derivada.reshape(-1, 1)

	return derivada

# Gradiente Descendente Estocastico
def sgd(x, y, tasa_aprendizaje, tam_batch, maxIteraciones = 1000):
    #
	# diapositiva 17, jussto antes del metodo de newton

	w = np.zeros((x.shape[1], 1), np.float64)

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
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


eta = 0.01
tam_batch = 64

w, iteraciones = sgd(x, y, eta, tam_batch)
print ('Bondad del resultado para grad. descendente estocastico con tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones:\n'.format(eta, tam_batch, iteraciones))
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

num5 = 1
num1 = -1

etiquetas = (num1, num5)
colores = {num1: 'blue', num5: 'red'}
valores = {num1: 1, num5: 5}

plt.clf()

for etiqueta in etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(y == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(x[indice, 1], x[indice,2], c=colores[etiqueta], label='{}'.format(valores[etiqueta]))

# ecuacion y = w0 + w1 * x1 + w2 * x2, queremos averiguar x2
# pintamos dos puntos, x1 = 0 y x1 = 1, siempre suponemos y = 0
# en el caso de x1 = 0, tenemos 0 = w0 + 0 * w2 * x2
# luego x2 = -w0/w2
x2_para_x1_0 = -w[0]/w[2]

# en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
# luego x2 = (-w0 - w1) /w2
x2_para_x1_1 = (-w[0] - w[1])/w[2]

plt.plot([0, 1], [x2_para_x1_0, x2_para_x1_1], 'k-', label='Modelo de regresión obtenido')

plt.title('Bondad del resultado para grad. descendente estocastico\ncon tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones:\n'.format(eta, tam_batch, iteraciones))
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")






#Seguir haciendo el ejercicio...

w = pseudoinverse(x, y)
print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")


plt.clf()

for etiqueta in etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(y == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(x[indice, 1], x[indice,2], c=colores[etiqueta], label='{}'.format(valores[etiqueta]))

# ecuacion y = w0 + w1 * x1 + w2 * x2, queremos averiguar x2
# pintamos dos puntos, x1 = 0 y x1 = 1, siempre suponemos y = 0
# en el caso de x1 = 0, tenemos 0 = w0 + 0 * w2 * x2
# luego x2 = -w0/w2
x2_para_x1_0 = -w[0]/w[2]

# en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
# luego x2 = (-w0 - w1) /w2
x2_para_x1_1 = (-w[0] - w[1])/w[2]

plt.plot([0, 1], [x2_para_x1_0, x2_para_x1_1], 'k-', label='Modelo de regresión obtenido')

plt.title('Bondad del resultado usando la pseudoinversa\n')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")






"""

PUNTO 2


"""


print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...


# Apartado a)

# usamos la funcion dada para obtener la muestra
muestra_entrenamiento = simula_unif(1000, 2, 1)

plt.clf()
plt.scatter(muestra_entrenamiento[:, 0], muestra_entrenamiento[:, 1], c='b')
plt.title('Muestra de entrenamiento generada por una distribución uniforme')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")


# apartado b)

# funcion F dada
def F(x1, x2):
	valor = np.square(x1 - 0.2) + np.square(x2) - 0.6

	return np.sign(valor)

# funcion para aplicar el ruido
# Parametro 1: array de etiquetas donde aplicar el ruido: suponemos que son 1/-1
# parametro 2: porcentaje de etiquetas al que aplicar el ruido
def ruido(etiquetas, porcentaje):

	n_etiquetas = etiquetas.copy()

	num_etiquetas = len(n_etiquetas)

	num_a_aplicar = num_etiquetas * porcentaje
	num_a_aplicar = int(round(num_a_aplicar))

	indices = np.random.choice(range(num_etiquetas), num_a_aplicar, replace=False)

	print('Numero de etiquetas: ', num_etiquetas )
	print('Numero al que le tenemos que aplicar ruido: ', num_a_aplicar)
	print('Tamaño de los indices a los que les vamos a aplicar ruido: ', len(indices))

	input("\n--- Pulsar tecla para continuar ---\n")


	for i in indices:
		n_etiquetas[i] = -n_etiquetas[i]

	return n_etiquetas

etiquetas = F(muestra_entrenamiento[:, 0], muestra_entrenamiento[:, 1])


posibles_etiquetas = (1, -1)
colores = {1: 'blue', -1: 'red'}

plt.clf()

for etiqueta in posibles_etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(etiquetas == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(muestra_entrenamiento[indice, 0], muestra_entrenamiento[indice,1], c=colores[etiqueta],  label='{}'.format( etiqueta ))

plt.title('Muestra de entrenamiento generada por una distribución uniforme antes de aplicar ruido')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


etiquetas = ruido(etiquetas, 0.1)



plt.clf()

for etiqueta in posibles_etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(etiquetas == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(muestra_entrenamiento[indice, 0], muestra_entrenamiento[indice,1], c=colores[etiqueta],  label='{}'.format( etiqueta ))

plt.title('Muestra de entrenamiento generada por una distribución uniforme tras aplicar ruido')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")



# apartado d


unos = np.ones((muestra_entrenamiento.shape[0], 1), dtype=np.float64)
print(unos[: 10])

print(muestra_entrenamiento[: 10])

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
caracteristicas = np.c_[unos, muestra_entrenamiento]


print(caracteristicas[: 10])


#https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html

unos = np.ones((1000, 1), dtype=np.float64)

eta = 0.01
w, iteraciones = sgd(caracteristicas, etiquetas, eta, 64)



plt.clf()



# ecuacion y = w0 + w1 * x1 + w2 * x2, queremos averiguar x2
# pintamos dos puntos, x1 = 0 y x1 = 1, siempre suponemos y = 0
# en el caso de x1 = 0, tenemos 0 = w0 + 0 * w2 * x2
# luego x2 = -w0/w2
x2_para_x1_0 = -w[0]/w[2]

# en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
# luego x2 = (-w0 - w1) /w2
x2_para_x1_1 = (-w[0] - w[1])/w[2]

plt.plot([0, 1], [x2_para_x1_0, x2_para_x1_1], 'k-', label='Modelo de regresión obtenido')


for etiqueta in posibles_etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(etiquetas == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(muestra_entrenamiento[indice, 0], muestra_entrenamiento[indice,1], c=colores[etiqueta],  label='{}'.format( etiqueta ))

plt.title('Muestra de entrenamiento generada por una distribución uniforme tras aplicar ruido')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')
plt.ylim(bottom = -1.1, top = 1.1)
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")











## para mas adelante

#caracteristicas = np.c_[unos, muestra_entrenamiento, muestra_entrenamiento[:, 0]*muestra_entrenamiento[:, 1],  np.square(muestra_entrenamiento[:, 0]),  np.square(muestra_entrenamiento[:, 1])]

# -*- coding: utf-8 -*-
"""
TRABAJO 2.
Nombre Estudiante: Antonio David Villegas
"""



import numpy as np
import matplotlib.pyplot as plt
from sympy import Add
from sympy.solvers import solve
from sympy import Symbol, var


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
def sgd(x, y, tasa_aprendizaje, tam_batch, maxIteraciones = 2000):
    #
	# diapositiva 17, jussto antes del metodo de newton

	# el tamaño de w será dependiendo del numero de elementos de x
	w = np.zeros((x.shape[1], 1), np.float64)

	iterations = 0

	# en este caso solo tenemos de condicion las iteraciones
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

# ejecutamos y valoramos los errores
w, iteraciones = sgd(x, y, eta, tam_batch)
print ('Bondad del resultado para grad. descendente estocastico con tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones:\n'.format(eta, tam_batch, iteraciones))
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

print ('Bondad media del resultado para grad. descendente estocastico con tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones:\n'.format(eta, tam_batch, iteraciones))
print ("Ein: ", Err(x,y,w).mean())
print ("Eout: ", Err(x_test, y_test, w).mean())

input("\n--- Pulsar tecla para continuar ---\n")

num5 = 1
num1 = -1

etiquetas = (num1, num5)
colores = {num1: 'b', num5: 'r'}
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

plt.title('\nBondad del resultado para grad. descendente estocastico\ncon tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones:\n'.format(eta, tam_batch, iteraciones))
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

plt.clf()

for etiqueta in etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(y_test == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(x_test[indice, 1], x_test[indice,2], c=colores[etiqueta], label='{}'.format(valores[etiqueta]))

# ecuacion y = w0 + w1 * x1 + w2 * x2, queremos averiguar x2
# pintamos dos puntos, x1 = 0 y x1 = 1, siempre suponemos y = 0
# en el caso de x1 = 0, tenemos 0 = w0 + 0 * w2 * x2
# luego x2 = -w0/w2
x2_para_x1_0 = -w[0]/w[2]

# en el caso de x1 = 1, tenemos 0 = w0 + w1 * w2 * x2
# luego x2 = (-w0 - w1) /w2
x2_para_x1_1 = (-w[0] - w[1])/w[2]
plt.plot([0, 1], [x2_para_x1_0, x2_para_x1_1], 'k-', label='Modelo de regresión obtenido')

plt.title('\nBondad del resultado para grad. descendente estocastico\ncon tasa de aprendizaje {}, tamaño de batch de {} y {} iteraciones\n fuera de la muestra:\n'.format(eta, tam_batch, iteraciones))
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

print ('Bondad media del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w).mean())
print ("Eout: ", Err(x_test, y_test, w).mean())

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


plt.clf()

for etiqueta in etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(y_test == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(x_test[indice, 1], x_test[indice,2], c=colores[etiqueta], label='{}'.format(valores[etiqueta]))

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



# apartado c
print('Comprobación de que tenemos las caracteristicas correctas: ')

unos = np.ones((muestra_entrenamiento.shape[0], 1), dtype=np.float64)
print(unos[: 10])

print(muestra_entrenamiento[: 10])

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
# unimos las caracteristicas que vamos a usar en una única variable
caracteristicas = np.c_[unos, muestra_entrenamiento]


print(caracteristicas[: 10])

input("\n--- Pulsar tecla para continuar ---\n")


#https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html

eta = 0.01
w, iteraciones = sgd(caracteristicas, etiquetas, eta, 64)


print ('Bondad del resultado para SGD:\n')
print ("Ein: ", Err(caracteristicas,etiquetas,w))
print ("Ein medio: ", Err(caracteristicas,etiquetas,w).mean())
input("\n--- Pulsar tecla para continuar ---\n")



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




# apartado d)

total = 0
Ein_ite = []
Eout_ite = []

print('Comenzando 1000 iteraciones, esto va a tardar un poco, en fin, python ...')

# aplicamos 1000 veces lo explicado en todos los pasos
while total < 1000:
	if (total == 500):
		print('Llevamos la mitad, ¡tu puedes python!')
	total = total + 1
	# generamos la muestra
	m = simula_unif(1000, 2, 1)
	# le aplicamos la función y el ruido
	etiq = F(m[:, 0], m[:, 1])
	etiq = ruido(etiq, 0.1)
	unos = np.ones((m.shape[0], 1), dtype=np.float64)
	# creamos las caracteristicas
	c = np.c_[unos, m]
	eta = 0.01
	# obtenemos el modelo a partir de la muestra de entrenamiento y lo evaluamos
	w, iteraciones = sgd(c, etiq, eta, 64, 1500)
	Ein_ite.append(Err(c, etiq, w))

	# generamos otra muestra de test y la evaluamos
	m_out = simula_unif(1000, 2, 1)
	etiq_out = F(m_out[:, 0], m_out[:, 1])
	etiq_out = ruido(etiq_out, 0.1)
	c_out = np.c_[unos, m_out]
	Eout_ite.append(Err(c_out, etiq_out, w))


Ein_ite = np.array(Ein_ite, dtype=np.float64)
Eout_ite = np.array(Eout_ite, dtype=np.float64)

Ein_medio = Ein_ite.mean()
Eout_medio = Eout_ite.mean()


# apartado e) -> en el PDF

print('Error medio dentro de la muestra: ', Ein_medio)
print('Error medio fuera de la muestra: ', Eout_medio)


input("\n--- Pulsar tecla para continuar ---\n")

















## Experimento con distintas caracteristicas


muestra_entrenamiento = simula_unif(1000, 2, 1)

plt.clf()
plt.scatter(muestra_entrenamiento[:, 0], muestra_entrenamiento[:, 1], c='b')
plt.title('Muestra de entrenamiento generada por una distribución uniforme\n para la versión con más caracterésticas')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")


# apartado b)


etiquetas = F(muestra_entrenamiento[:, 0], muestra_entrenamiento[:, 1])


posibles_etiquetas = (1, -1)
colores = {1: 'blue', -1: 'red'}

plt.clf()

for etiqueta in posibles_etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(etiquetas == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(muestra_entrenamiento[indice, 0], muestra_entrenamiento[indice,1], c=colores[etiqueta],  label='{}'.format( etiqueta ))

plt.title('Muestra de entrenamiento generada por una distribución uniforme antes de aplicar ruido para la versión con más caracterésticas')
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

plt.title('Muestra de entrenamiento generada por una distribución uniforme tras aplicar ruido para la versión con más caracterésticas')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")



# apartado c
print('Comprobación de que tenemos las caracteristicas correctas: ')

unos = np.ones((muestra_entrenamiento.shape[0], 1), dtype=np.float64)
print(unos[: 10])

print(muestra_entrenamiento[: 10])

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.c_.html
# unimos las caracteristicas que vamos a usar en una única variable
caracteristicas = np.c_[unos, muestra_entrenamiento, muestra_entrenamiento[:, 0]*muestra_entrenamiento[:, 1],  np.square(muestra_entrenamiento[:, 0]),  np.square(muestra_entrenamiento[:, 1])]


print(caracteristicas[: 10])

input("\n--- Pulsar tecla para continuar ---\n")


eta = 0.01
w, iteraciones = sgd(caracteristicas, etiquetas, eta, 64)


print ('Bondad del resultado para SGD:\n')
print ("Ein: ", Err(caracteristicas,etiquetas,w))
print ("Ein medio: ", Err(caracteristicas,etiquetas,w).mean())
input("\n--- Pulsar tecla para continuar ---\n")


plt.clf()


print('Esto tarda un poco, que estamos calculando bastantes valores ...')

valores_entre_cero_uno = []
# tendremos la parte superior y la inferior, ya que la función tiene dos valores en el mismo punto
valores_a_dibujar_sup = []
valores_a_dibujar_inf = []

# y = w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1*x2 + w[4]*x1^2 + w[5]*x2^2
# desde -1 hasta 1, el rango en el que generamos los puntos
i = -1

z = Symbol('x')

while i <= 1:
	# resolvemos la ecuacion para cada punto de i, suponiendo y = 0
	valores_funcion = solve(w[0] + w[1]*i + w[2]*z + w[3]*i*z + w[4]*i*i + w[5]*z*z, z)

	# si los valores obtenidos no son imaginarios los añadimos a los que hay que dibujar
	# (Sympy representa los imaginarios como instancia de Add)
	if not isinstance(valores_funcion[0][0], Add):
		valores_entre_cero_uno.append(i)
		valores_a_dibujar_inf.append(valores_funcion[0])
	if not isinstance(valores_funcion[1][0], Add):
		valores_a_dibujar_sup.append(valores_funcion[1])
	i = i + 1/100

valores_a_dibujar_sup = np.array(valores_a_dibujar_sup)
valores_a_dibujar_inf = np.array(valores_a_dibujar_inf)

# dibujamos ambas lineas, la superior y la inferior
plt.plot(valores_entre_cero_uno, valores_a_dibujar_sup, "-k", label='Modelo obtenido')
plt.plot(valores_entre_cero_uno, valores_a_dibujar_inf, "-k")

# dibujamos los puntos
for etiqueta in posibles_etiquetas:
	# en Y buscamos los puntos que coinciden con la etiqueta
	indice = np.where(etiquetas == etiqueta)
	# los dibujamos como scatterplot con su respectivo color
	plt.scatter(muestra_entrenamiento[indice, 0], muestra_entrenamiento[indice,1], c=colores[etiqueta],  label='{}'.format( etiqueta ))

plt.title('Muestra de entrenamiento generada por una distribución uniforme\n tras aplicar ruido para la versión con más caracterésticas')
plt.xlabel('Valor x_1')
plt.ylabel('Valor x_2')
#plt.ylim(bottom = -1.1, top = 1.1)
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")




# apartado d)

total = 0
Ein_ite = []
Eout_ite = []

print('Comenzando 1000 iteraciones, esto va a tardar un poco, en fin, python ...')

# aplicamos 1000 veces lo explicado en todos los pasos
while total < 1000:
	if (total == 500):
		print('Llevamos la mitad, ¡tu puedes python!')
	total = total + 1
	# generamos la muestra
	m = simula_unif(1000, 2, 1)
	# aplicamos la funcion y el ruido
	etiq = F(m[:, 0], m[:, 1])
	etiq = ruido(etiq, 0.1)
	unos = np.ones((m.shape[0], 1), dtype=np.float64)
	# calculamos las caracteristicas
	c = np.c_[unos, m, m[:, 0]*m[:, 1],  np.square(m[:, 0]),  np.square(m[:, 1])]
	eta = 0.01
	# obtenemos y evaluamos el modelo a partir de la muestra de entrenamiento
	w, iteraciones = sgd(c, etiq, eta, 64, 1500)
	Ein_ite.append(Err(c, etiq, w))

	# obtenemos y evaluamos la muestra de test con el modelo obtenido
	m_out = simula_unif(1000, 2, 1)
	etiq_out = F(m_out[:, 0], m_out[:, 1])
	etiq_out = ruido(etiq_out, 0.1)
	c_out = np.c_[unos, m, m[:, 0]*m[:, 1],  np.square(m[:, 0]),  np.square(m[:, 1])]
	Eout_ite.append(Err(c_out, etiq_out, w))


Ein_ite = np.array(Ein_ite, dtype=np.float64)
Eout_ite = np.array(Eout_ite, dtype=np.float64)

Ein_medio = Ein_ite.mean()
Eout_medio = Eout_ite.mean()


# apartado e) -> en el PDF

print('Error medio dentro de la muestra: ', Ein_medio)
print('Error medio fuera de la muestra: ', Eout_medio)


input("\n--- Pulsar tecla para continuar ---\n")

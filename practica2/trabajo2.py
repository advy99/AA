# -*- coding: utf-8 -*-
"""
PRACTICA 2 - EJ 2
Nombre Estudiante: Antonio David Villegas Yeguas
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(0)

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

plt.clf()

indices_positivos = np.where(np.array(etiquetas) == 1)
indices_positivos = indices_positivos[0]

indices_negativos = np.where(np.array(etiquetas) == -1)
indices_negativos = indices_negativos[0]


# aplicamos el ruido a los positivos
num_a_aplicar = len(indices_positivos) * 0.1
num_a_aplicar = int(round(num_a_aplicar))
indices = np.random.choice(indices_positivos, num_a_aplicar, replace=False)

for i in indices:
	etiquetas[i] = -etiquetas[i]

# aplicamos el ruido a los negativos
num_a_aplicar = len(indices_negativos) * 0.1
num_a_aplicar = int(round(num_a_aplicar))
indices = np.random.choice(indices_negativos, num_a_aplicar, replace=False)

for i in indices:
	etiquetas[i] = -etiquetas[i]


w_0 = np.zeros(3)

# tenemos ruido, si no poneos límite cicla infinito
w, iteraciones = ajusta_PLA(puntos_2d, etiquetas, 10000, w_0)

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta con la que etiquetamos')

# obtenemos 0 = w_0 + w_1 * x1 + w_2 * x_2

# x_2 = (-w_0 - w_1 * x_1 )/ w_2


plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con PLA')


for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)
	# ahora no es 0 y 1, si no 1 y 2, porque le he metido un primer 1 para usar w en el perceptron
	plt.scatter(puntos_2d[indice, 1], puntos_2d[indice, 2], c=colores[etiqueta], label="{}".format(etiqueta))



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
	w, iteraciones = ajusta_PLA(puntos_2d, etiquetas, 10000, w_0)
	iterations.append(iteraciones)
    #CODIGO DEL ESTUDIANTE

print('Valor medio de iteraciones necesario para converger (no converge, es el máximo de iteraciones): {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def error_erm(x, y, w):

	num_elementos = x.shape[1]

	error = np.float64(0.0)

	for i in range(num_elementos):
		error += np.log(1 + np.e**(-y[i]*w.T.dot(x[i])))

	error = error/num_elementos

	return error

def grad(x, y, w):

	return -(y * x)/(1 + np.exp(y * w.T.dot(x)))



def sgdRL(x, y, tasa_aprendizaje, error_permitido = 0.01):
    #
	# diapositiva 17, jussto antes del metodo de newton

	# el tamaño de w será dependiendo del numero de elementos de x
	w = np.zeros((x.shape[1],), np.float64)
	w_ant = w.copy()

	acabar = False

	epocas = 0


	# en este caso solo tenemos de condicion las iteraciones
	while not acabar:

		# aplicamos una permutacion aleatoria a los indices del minibatch
		indices_minibatch = np.random.choice(x.shape[0], x.shape[0], replace=False)

		# tamaños de minibatch = 1
		for i in indices_minibatch:
			w = w - tasa_aprendizaje * grad(x[i], y[i], w)

		epocas += 1
		dist = np.linalg.norm(w_ant - w)

		if dist < error_permitido:
			acabar = True

		w_ant = w.copy()


	return w, epocas


#CODIGO DEL ESTUDIANTE

intervalo_trabajo = [0, 2]

x = simula_unif(100, 2, intervalo_trabajo)


# escogemos dos puntos y hacemos que la recta pase por estos
puntos_recta = np.random.choice(x.shape[0], 2, replace=False)

# ecuacio nde una recta: m = y_2 - y_1 / x_2 - x_1

a = (x[puntos_recta[1]][1] - x[puntos_recta[0]][1]) / (x[puntos_recta[1]][0] - x[puntos_recta[0]][0])

# y = ax + b

b = x[puntos_recta[0]][1] - (a*x[puntos_recta[0]][0])

etiquetas = []

posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

for punto in x:
	etiquetas.append(f(punto[0], punto[1], a, b))

for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)

	plt.scatter(x[indice, 0], x[indice, 1], c=colores[etiqueta], label="{}".format(etiqueta))



# y = a*x + b

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta obtenida aleatoriamente')


plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {}, etiquetados segun una recta".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

etiquetas = np.array(etiquetas)
x = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]


w, epocas = sgdRL(x, etiquetas, 0.01, 0.01)


posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con sgdRL')


plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta obtenida aleatoriamente')


for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)
	# ahora no es 0 y 1, si no 1 y 2, porque le he metido un primer 1 para usar w en el perceptron
	plt.scatter(x[indice, 1], x[indice, 2], c=colores[etiqueta], label="{}".format(etiqueta))



plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {}, etiquetados segun una recta".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()



print("W obtenida: " + str(w) + "\n Épocas: " + str(iteraciones))
print("Error obtenido dentro de la muestra (Ein): " + str(error_erm(x, etiquetas, w)))


input("\n--- Pulsar tecla para continuar ---\n")



# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE
x_test = simula_unif(1000, 2, intervalo_trabajo)

etiquetas = []

posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

for punto in x_test:
	etiquetas.append(f(punto[0], punto[1], a, b))


etiquetas = np.array(etiquetas)
x_test = np.c_[np.ones((x_test.shape[0], 1), dtype=np.float64), x_test]


posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

plt.plot(intervalo_trabajo, [ (-w[0]-w[1]*intervalo_trabajo[0])/w[2], (-w[0]-w[1]*intervalo_trabajo[1])/w[2]], 'y-', label='Recta obtenida con sgdRL')


plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta obtenida aleatoriamente')


for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)
	# ahora no es 0 y 1, si no 1 y 2, porque le he metido un primer 1 para usar w en el perceptron
	plt.scatter(x_test[indice, 1], x_test[indice, 2], c=colores[etiqueta], label="{}".format(etiqueta))



plt.title("Nube de 1000 puntos bidimensionales en el intervalo {} {}, etiquetados segun una recta".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


print("Error obtenido fuera de la muestra (Eout): " + str(error_erm(x_test, etiquetas, w)))


input("\n--- Pulsar tecla para continuar ---\n")

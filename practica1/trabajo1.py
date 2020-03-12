# -*- coding: utf-8 -*-
"""
TRABAJO 1.
Nombre Estudiante: Antonio David Villegas
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# Función dada en el ejercicio 1.
def E(u,v):
	return np.float64( (u * np.e**(v) - 2 * v * np.e**(-u))**2 ) # function

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return np.float64( 2 * (u*np.e**(v) - 2 * v * np.e**(-u) ) * (np.e**(v) + 2 * v * np.e**(-u) ) )#Derivada parcial de E con respecto a u

#Derivada parcial de E con respecto a v
def dEv(u,v):
    return np.float64( 2 * (u*np.e**(v) - 2 * v * np.e**(-u) ) * (u * np.e**(v) - 2 * np.e**(-u) ) )#Derivada parcial de E con respecto a v

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])



# Función dada en el ejercicio 1.3.
def F(x,y):
	return np.float64( (x - 2)**2 + 2 * (y + 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)  ) # function

#Derivada parcial de F con respecto a x
def dFx(x, y):
    return np.float64( 2*x - 4 + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) )#Derivada parcial de E con respecto a u

#Derivada parcial de F con respecto a y
def dFy(x, y):
    return np.float64( 4*y + 8 + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) )#Derivada parcial de E con respecto a v

#Gradiente de E
def gradF(x, y):
    return np.array([dFx(x, y), dFy(x, y)])


def gradient_descent(funcion, gradFuncion, w_0, tasa_aprendizaje, maxIter, maxError):
    #
    # gradiente descendente
    #
	iterations = 0
	error = funcion(w_0[0], w_0[1])

	w_j = w_0

	# Condición de parada, llegamos al límite de iteraciones
	# o el error es menor que el error máximo permitido
	while iterations < maxIter and error > maxError:

		w_j = w_j - tasa_aprendizaje * gradFuncion(w_j[0], w_j[1])
		error = funcion(w_j[0], w_j[1])

		iterations = iterations + 1


	w = w_j

	return w, iterations


eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(E, gradE, initial_point, eta, maxIter, error2get)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor de la función en dichas coordenadas', E(w[0], w[1]))


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")

plt.show()








"""
Apartado 1.3: cambiamos la función

"""





eta = 0.01
maxIter = 50
error2get = -np.Infinity
initial_point = np.array([1.0,-1.0])
w, it = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor de la función en dichas coordenadas', F(w[0], w[1]))


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')

input("\n--- Pulsar tecla para continuar ---\n")

plt.show()






"""
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
    return

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...
"""

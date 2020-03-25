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

	valores = np.array(funcion(w_j[0], w_j[1]))

	# Condición de parada, llegamos al límite de iteraciones
	# o el error es menor que el error máximo permitido
	while iterations < maxIter and error > maxError:

		w_j = w_j - tasa_aprendizaje * gradFuncion(w_j[0], w_j[1])
		error = funcion(w_j[0], w_j[1])
		valores = np.append(valores, error)
		print('Valor de la función tras ', iterations, ' iteraciones: ', error )
		iterations = iterations + 1


	w = w_j

	return w, iterations, valores


eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it, valores_descenso = gradient_descent(E, gradE, initial_point, eta, maxIter, error2get)


print ('Tasa de aprendizaje: ', eta)
print ('Error mínimo permitido: ', error2get)
print ('Numero de iteraciones maximas: ', maxIter)
print ('Numero de iteraciones: ', it)
print ('Coordenadas iniciales: (', initial_point[0], ', ', initial_point[1],')')
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Valor de la función en dichas coordenadas', E(w[0], w[1]))

input("\n--- Pulsar tecla para continuar ---\n")


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


plt.show()

input("\n--- Pulsar tecla para continuar ---\n")






"""
Apartado 1.3: cambiamos la función

"""



def mostrar_funcion_F(eta, maxIter, error2get, initial_point):
	w, it, valores_descenso = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)


	print ('Tasa de aprendizaje: ', eta)
	print ('Error mínimo permitido: ', error2get)
	print ('Numero de iteraciones maximas: ', maxIter)
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas iniciales: (', initial_point[0], ', ', initial_point[1],')')
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
	ax.set(title='Ejercicio 1.3. Descenso de gradiente sobre F(x,y)\n con punto inicial {}, {}, tasa de aprendizaje {} y un maximo de iteraciones de {}'.format(initial_point[0], initial_point[1], eta, maxIter))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('F(x,y)')


	plt.show()

	input("\n--- Pulsar tecla para continuar ---\n")


	plt.clf()

	plt.title('Ejercicio 1.3. Descenso de gradiente sobre F(x,y)\n con punto inicial {}, {}, tasa de aprendizaje {} y un maximo de iteraciones de {}'.format(initial_point[0], initial_point[1], eta, maxIter))
	plt.xlabel('Iteraciones')
	plt.ylabel('Valor de F(x,y)')
	plt.scatter(range(valores_descenso.size), valores_descenso[:])
	plt.plot(range(valores_descenso.size), valores_descenso[:])

	plt.show()


	input("\n--- Pulsar tecla para continuar ---\n")

	return w, it, valores_descenso





eta = 0.01
maxIter = 50

# en este caso no tenemos el error como minimo, luego lo establecemos a
# -infinito para que no influya en la ejecución, solo tenga en cuenta
# las iteraciones
error2get = -np.Infinity
initial_point = np.array([1.0,-1.0])

diferencias_valor = []

## llamamos a la función que nos ejecutará y mostrará las gráficas de F
w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor.append(valor_descenso)



"""
 con tasa de aprendizaje a 0.1 en lugar de 0.01
"""


eta = 0.1
maxIter = 50

# en este caso no tenemos el error como minimo, luego lo establecemos a
# -infinito para que no influya en la ejecución, solo tenga en cuenta
# las iteraciones
error2get = -np.Infinity
initial_point = np.array([1.0,-1.0])
#w, it = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)

w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor.append(valor_descenso)

## mostramos en la misma grafica las diferencias
plt.clf()

diferencias_valor = np.array(diferencias_valor)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')

plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Tasa aprendizaje = 0.1')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con una tasa de aprendizaje de 0.1 y de 0.01 con punto inicial {}, {}'.format(initial_point[0], initial_point[1]))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")






"""
 con los distintos puntos dados
"""


eta = 0.01
maxIter = 50

# en este caso no tenemos el error como minimo, luego lo establecemos a
# -infinito para que no influya en la ejecución, solo tenga en cuenta
# las iteraciones
error2get = -np.Infinity
initial_point = np.array([2.1,-2.1])



w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor = []
diferencias_valor.append(valor_descenso)

# con tasa de aprendizaje 0.1
eta = 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor.append(valor_descenso)

## mostramos en la misma grafica las diferencias
plt.clf()

diferencias_valor = np.array(diferencias_valor)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')

plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Tasa aprendizaje = 0.1')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con una tasa de aprendizaje de 0.1 y de 0.01 con punto inicial {}, {}'.format(initial_point[0], initial_point[1]))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")








eta = 0.01
maxIter = 50

# en este caso no tenemos el error como minimo, luego lo establecemos a
# -infinito para que no influya en la ejecución, solo tenga en cuenta
# las iteraciones
error2get = -np.Infinity
initial_point = np.array([3.0,-3.0])

w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor = []
diferencias_valor.append(valor_descenso)

# con tasa de aprendizaje 0.1
eta = 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor.append(valor_descenso)

## mostramos en la misma grafica las diferencias
plt.clf()

diferencias_valor = np.array(diferencias_valor)


plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')

plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Tasa aprendizaje = 0.1')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con una tasa de aprendizaje de 0.1 y de 0.01 con punto inicial {}, {}'.format(initial_point[0], initial_point[1]))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")







eta = 0.01
maxIter = 50

# en este caso no tenemos el error como minimo, luego lo establecemos a
# -infinito para que no influya en la ejecución, solo tenga en cuenta
# las iteraciones
error2get = -np.Infinity
initial_point = np.array([1.5,1.5])

w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor = []
diferencias_valor.append(valor_descenso)


eta = 0.1

# con tasa de aprendizaje 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F(eta, maxIter, error2get, initial_point)

diferencias_valor.append(valor_descenso)

## mostramos en la misma grafica las diferencias
plt.clf()

diferencias_valor = np.array(diferencias_valor)


plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')

plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Tasa aprendizaje = 0.1')

plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con una tasa de aprendizaje de 0.1 y de 0.01 con punto inicial {}, {}'.format(initial_point[0], initial_point[1]))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")





"""

Tabla con resultados -> en el PDF de documentación

Ejercicio 1.4 -> en el PDF de documentación

"""

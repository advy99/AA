# -*- coding: utf-8 -*-
"""
TRABAJO 3 (BONUS).
Nombre Estudiante: Antonio David Villegas
"""


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO Bonus\n')
print('Ejercicio 1\n')



# Función dada en el ejercicio 1.3.
def F(x,y):
	return np.float64( (x - 2)**2 + 2 * (y + 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)  ) # function

#Derivada parcial de F con respecto a x
def dFx(x, y):
    return np.float64( 2*x - 4 + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) )#Derivada parcial de E con respecto a u

#Derivada parcial de F con respecto a y
def dFy(x, y):
    return np.float64( 4*y + 8 + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) )#Derivada parcial de E con respecto a v

# segunda derivada parcial con respecto a x
def ddFx(x, y):
	return np.float64(-8*np.square(np.pi) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y) + 2 )

# segunda derivada parcial con respecto a y
def ddFy(x, y):
	return np.float64(-8*np.square(np.pi) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y) + 4 )

# segunda derivada respecto a x e y (son equivalentes a la de Y y x)
def ddF(x, y):
	return np.float64(8*np.square(np.pi) * np.cos(2*x*np.pi)*np.cos(2*y*np.pi))

# matriz hessiana
def matriz_hessiana(x, y):
	#https://en.wikipedia.org/wiki/Hessian_matrix
	return np.array([[ddFx(x,y), ddF(x,y)],[ddF(x,y), ddFy(x,y)]])

#Gradiente de E
def gradF(x, y):
    return np.array([dFx(x, y), dFy(x, y)])


def metodo_newton(funcion, gradFuncion, matriz_hessiana, w_0, eta = 1, maxIter = 50):
    #
    # metodo de newton -> diapositiva 18 de teoria
    #
	iterations = 0
	error = funcion(w_0[0], w_0[1])

	w_j = w_0

	valores = np.array(funcion(w_j[0], w_j[1]))

	# Condición de parada, llegamos al límite de iteraciones
	while iterations < maxIter:
		# seguimos la formula dada en teoria
		hessiana_invertida = -np.linalg.inv(matriz_hessiana(w_j[0], w_j[1]))

		# si la hessiana es positiva_definida, apunta a un minimo, por lo que nos acercamos a ese mínimo
		positiva_definida = w_j + eta * hessiana_invertida.dot(gradFuncion(w_j[0], w_j[1]).reshape(-1, 1)).reshape(-1,)

		# si es negativa_definida, apunta a un máximo, por lo que nos alejamos de el
		negativa_definida = w_j - eta * hessiana_invertida.dot(gradFuncion(w_j[0], w_j[1]).reshape(-1, 1)).reshape(-1,)

		# nos quedamos con la suposición que nos minimice
		if funcion(positiva_definida[0], positiva_definida[1]) > funcion(negativa_definida[0], negativa_definida[1]):
			w_j = negativa_definida
		else:
			w_j = positiva_definida

		error = funcion(w_j[0], w_j[1])
		valores = np.append(valores, error)
		print('Valor de la función tras ', iterations, ' iteraciones: ', error )
		iterations = iterations + 1

	w = w_j

	return w, iterations, valores




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

		# aplicamos la función dada en teoria
		w_j = w_j - tasa_aprendizaje * gradFuncion(w_j[0], w_j[1])
		error = funcion(w_j[0], w_j[1])
		# guardamos los valores, los usaremos más adelante
		valores = np.append(valores, error)
		print('Valor de la función tras ', iterations, ' iteraciones: ', error )
		iterations = iterations + 1


	w = w_j

	return w, iterations, valores


def mostrar_funcion_F_newton(maxIter, initial_point, eta):
	w, it, valores_descenso = metodo_newton(F, gradF, matriz_hessiana, initial_point, eta)


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
	ax.set(title='Ejercicio 1.3. Descenso de gradiente sobre F(x,y)\n con punto inicial {}, {}, tasa de aprendizaje de {} y un maximo de iteraciones de {}'.format(initial_point[0], initial_point[1], eta, maxIter))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('F(x,y)')

	plt.show()
	input("\n--- Pulsar tecla para continuar ---\n")


	plt.clf()

	plt.title('Ejercicio 1.3. Metodo de Newton sobre F(x,y)\n con punto inicial {}, {}, tasa de aprendizaje {} y un maximo de iteraciones de {}'.format(initial_point[0], initial_point[1], eta, maxIter))
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
w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)

diferencias_valor.append(valor_descenso)

input("\n--- Pulsar tecla para continuar ---\n")



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

w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)

diferencias_valor.append(valor_descenso)


w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, 1)

diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")


diferencias_valor_grad = []

w, iteraciones, valor_descenso = gradient_descent(F, gradF, initial_point, 0.01, maxIter, -np.Infinity)
diferencias_valor_grad.append(valor_descenso)

plt.clf()

diferencias_valor = np.array(diferencias_valor)
diferencias_valor_grad = np.array(diferencias_valor_grad)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')
plt.scatter(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y')
plt.scatter(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k')


plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Newton Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Newton Tasa aprendizaje = 0.1')
plt.plot(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y', label='Newton Tasa aprendizaje = 1')
plt.plot(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k', label='Grad. des Tasa aprendizaje = 0.01')


plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con tasa de apr. de 0.1, 0.01 y 1 del método de Newton vs Grad. descendente con w_0 {}, {}'.format(initial_point[0], initial_point[1]))
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
diferencias_valor = []


w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)
diferencias_valor.append(valor_descenso)

input("\n--- Pulsar tecla para continuar ---\n")

eta = 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)
diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")


w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, 1)

diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")


diferencias_valor_grad = []

w, iteraciones, valor_descenso = gradient_descent(F, gradF, initial_point, 0.01, maxIter, -np.Infinity)
diferencias_valor_grad.append(valor_descenso)

plt.clf()

diferencias_valor = np.array(diferencias_valor)
diferencias_valor_grad = np.array(diferencias_valor_grad)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')
plt.scatter(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y')
plt.scatter(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k')


plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Newton Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Newton Tasa aprendizaje = 0.1')
plt.plot(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y', label='Newton Tasa aprendizaje = 1')
plt.plot(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k', label='Grad. des Tasa aprendizaje = 0.01')


plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con tasa de apr. de 0.1, 0.01 y 1 del método de Newton vs Grad. descendente con w_0 {}, {}'.format(initial_point[0], initial_point[1]))
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
diferencias_valor = []


w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)
diferencias_valor.append(valor_descenso)

input("\n--- Pulsar tecla para continuar ---\n")
eta = 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)
diferencias_valor.append(valor_descenso)

input("\n--- Pulsar tecla para continuar ---\n")




w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, 1)

diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")

diferencias_valor_grad = []

w, iteraciones, valor_descenso = gradient_descent(F, gradF, initial_point, 0.01, maxIter, -np.Infinity)
diferencias_valor_grad.append(valor_descenso)

plt.clf()

diferencias_valor = np.array(diferencias_valor)
diferencias_valor_grad = np.array(diferencias_valor_grad)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')
plt.scatter(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y')
plt.scatter(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k')


plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Newton Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Newton Tasa aprendizaje = 0.1')
plt.plot(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y', label='Newton Tasa aprendizaje = 1')
plt.plot(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k', label='Grad. des Tasa aprendizaje = 0.01')


plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con tasa de apr. de 0.1, 0.01 y 1 del método de Newton vs Grad. descendente con w_0 {}, {}'.format(initial_point[0], initial_point[1]))
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
diferencias_valor = []
diferencias_valor_grad = []


w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)
diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")
eta = 0.1
w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, eta)

diferencias_valor.append(valor_descenso)

input("\n--- Pulsar tecla para continuar ---\n")



w, iteraciones, valor_descenso = mostrar_funcion_F_newton(maxIter, initial_point, 1)

diferencias_valor.append(valor_descenso)


input("\n--- Pulsar tecla para continuar ---\n")

w, iteraciones, valor_descenso = gradient_descent(F, gradF, initial_point, 0.01, maxIter, -np.Infinity)
diferencias_valor_grad.append(valor_descenso)

plt.clf()

diferencias_valor = np.array(diferencias_valor)
diferencias_valor_grad = np.array(diferencias_valor_grad)

plt.scatter(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b')
plt.scatter(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r')
plt.scatter(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y')
plt.scatter(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k')


plt.plot(range(diferencias_valor[0].size), diferencias_valor[0, :], c='b', label='Newton Tasa aprendizaje = 0.01')
plt.plot(range(diferencias_valor[1].size), diferencias_valor[1, :], c='r', label='Newton Tasa aprendizaje = 0.1')
plt.plot(range(diferencias_valor[2].size), diferencias_valor[2, :], c='y', label='Newton Tasa aprendizaje = 1')
plt.plot(range(diferencias_valor_grad[0].size), diferencias_valor_grad[0, :], c='k', label='Grad. des Tasa aprendizaje = 0.01')


plt.xlabel('Iteraciones')
plt.ylabel('Valor de F(x,y)')
plt.title('Comparación con tasa de apr. de 0.1, 0.01 y 1 del método de Newton vs Grad. descendente con w_0 {}, {}'.format(initial_point[0], initial_point[1]))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")




"""

Tabla con resultados -> en el PDF de documentación

Ejercicio 1.4 -> en el PDF de documentación

"""

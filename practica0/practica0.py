# -*- coding: utf-8 -*-

# importamos numpy entero
import numpy as np

# importamos matplotlib.pylot para poder hacer gráficas
import matplotlib.pyplot as plt

# importamos los datasets de sklearn
from sklearn import datasets



"""
Funciones que usaremos en la parte 1 y 2

"""

def pintar_iris(caracteristicas, colores, etiquetas, titulo, etiquetaX, etiquetaY, posLeyenda):

	for i in range(0, len(caracteristicas)):
		plt.scatter(caracteristicas[i][:, 0], caracteristicas[i][:, 1],  c=colores[i], label=etiquetas[i])


	# Ponemos las etiquetas del eje X, Y y el título
	plt.xlabel(etiquetaX)
	plt.ylabel(etiquetaY)
	plt.title(titulo)

	# Añadimos la legenda, que deenderá del parametro 'label' al dibujar, también le decimos
	# que esté arriba a la izquierda
	plt.legend(loc=posLeyenda)

	# Activamos la rejilla, para ver mejor los valores
	plt.grid(True, which='both')





"""
Parte 1:

Referencia: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
(Documentación oficial de scikit-learn)

"""
iris = datasets.load_iris()

# almacenamos todas las caracteristicas de iris
caracteristicas = iris.data[:]

# almacenamos la clase
clase = iris.target

# nos quedamos con las dos ultimas caracteristicas (dos últimas columnas)
print("Caracteristicas (primera fila): " + str(caracteristicas[0]) )
caracteristicas = caracteristicas[:, -2:]
print("Caracteristicas (primera fila) tras quedarnos con las dos últimas: "  + str(caracteristicas[0]) )

"""
 Dibujamos en un Scatter Plot:
 Eje X: De todas las caracteristicas, el primer valor con el que nos quedamos
 Eje Y: De todas las caracteristicas, el segundo valor con el que nos quedamos

Tenemos que recordar que nos quedamos con los dos últimas caracteristicas,
así que en realidad estamos mostrando el penultimo y el último valor de Iris.

Tambien establecemos el color a la clase

"""

"""
 Dividimos las caracteristicas en tres partes, según su clase.

 Esto lo conseguimos con el metodo split, pasando una lista como argumento,
 si le pasamos la lista [a, b] con el vector v nos hará la división

 v[0:a]
 v[a:b]
 v[b:]

 De esta forma, si pasamos como argumento los indices de la clase 1 y 2, tendrémos
 tres partes, cada una con el tipo de clase.

"""

caracteristicas = np.split(caracteristicas, [list(clase).index(1), list(clase).index(2)])

# Tenemos tres colores, rojo, verde y azul, para tres etiquetas, flor de clase 0, 1 y 2
colores = ["r", "g", "b"]
etiquetas = ["Clase de flor 0", "Clase de flor 1", "Clase de flor 2"]

titulo = 'Parte 1 (recordar que en realidad nos quedamos solo con los dos últimos valores)'
etiqueta_x = 'X: Penultimo valor de los datos de iris'
etiqueta_y = 'Y: Último valor de los datos de iris'

# Dibujamos los tres tipos de clases (anteriormente divididos con split)
pintar_iris(caracteristicas, colores, etiquetas, titulo, etiqueta_x, etiqueta_y, "upper left")

# Mostramos el gráfico
plt.show()

# limpiamos la figura
plt.clf()


"""
Parte 2:

"""

training = []
test = []


# Para todas las clases (clase 0, 1 y 2)
for i in range(0, len(caracteristicas)):

	datos = []

	# para cada clase, el 80 % de esa clase escogemos un numero aleatorio
	# por ejemplo, si tenemos 10 valores, escogemos 8 aleatorios
	num_training = int(len(caracteristicas[i])*0.8 )

	for j in range(0, num_training ):
		# escogemos un indice al azar de entre el numero de elementos que nos quedan
		indice = np.random.randint(low=0, high=len(caracteristicas[i]))

		# añadimos ese elemento a datos
		datos.append(caracteristicas[i][indice] )

		# y lo eliminamos del conjunto inicial
		caracteristicas[i] = np.delete(caracteristicas[i], obj=indice, axis=0)

	# el 80 % escogido al azar es training
	training.append(datos)

	# el resto que nos queda en caracteristicas será nuestro test
	test.append(caracteristicas[i])


"""
 Como nota, añadir que hacemos el 80% de cada clase, por lo que seguimos
 teniendo el 80 % del total, pero asegurando que se mantiene la proporción
 de los elementos de cada clase
"""

training = np.array(training)

test = np.array(test)

"""
Dibujamos la division de training
"""

colores = ["r", "g", "b"]
etiquetas = ["(Training) Clase de flor 0", "(Training) Clase de flor 1", "(Training) Clase de flor 2"]

titulo = 'Parte 2: Training'
etiqueta_x = 'X: Penultimo valor de los datos de training'
etiqueta_y = 'Y: Último valor de los datos de training'

# Dibujamos los tres tipos de clases de training
pintar_iris(training, colores, etiquetas, titulo, etiqueta_x, etiqueta_y, "upper left")

# Mostramos el gráfico
plt.show()

# Limpiamos lo que teniamos
plt.clf()


"""
Dibujamos la division de test

"""

# Tenemos tres colores, rojo, verde y azul, para tres etiquetas, flor de clase 0, 1 y 2
colores = ["r", "g", "b"]
etiquetas = ["(Test) Clase de flor 0", "(Test) Clase de flor 1", "(Test) Clase de flor 2"]

titulo = 'Parte 2: Test'
etiqueta_x = 'X: Penultimo valor de los datos de training'
etiqueta_y = 'Y: Último valor de los datos de training'

# Dibujamos los tres tipos de clases de training
pintar_iris(test, colores, etiquetas, titulo, etiqueta_x, etiqueta_y, "upper left")

plt.show()

# limpiamos lo que teniamos
plt.clf()



"""
Parte 3:

"""


valores_2pi = []

i = 0

while i < 2*np.pi:
	valores_2pi.append(i)
	i = i + 2*np.pi/100

sin_valores = []
cos_valores = []

sin_cos_valores = []

for i in valores_2pi:
	sin = np.sin(i)
	cos = np.cos(i)
	sin_valores.append(sin)
	cos_valores.append(cos)
	sin_cos_valores.append(sin + cos)



plt.plot(valores_2pi[:], sin_valores[:] , "--k", label="sin(x)")
plt.plot(valores_2pi[:], cos_valores[:], "--b", label="cos(x)")
plt.plot(valores_2pi[:], sin_cos_valores[:], "--r", label="sin(x) + cos(x)")


plt.xlabel("Eje X: Desde 0 hasta 2*pi")
plt.ylabel("Eje Y: Valor de X en la función")
plt.title("Parte 3: Sin(x), cos(x) y sin(x) + cos(x)")

# Añadimos la legenda, que deenderá del parametro 'label' al dibujar, también le decimos
# que esté arriba a la izquierda
plt.legend(loc="upper center")

# Activamos la rejilla, para ver mejor los valores
plt.grid(True, which='both')

plt.show()

# limpiamos la figura
plt.clf()

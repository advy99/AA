# -*- coding: utf-8 -*-
"""
PRACTICA 2 - EJ 1
Nombre Estudiante: Antonio David Villegas Yeguas
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

# obtenemos de forma uniforme 50 puntos bidimensionales en [-50, 50]
puntos_simula_unif = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE

# los dibujamos todos
plt.scatter(puntos_simula_unif[:, 0], puntos_simula_unif[:, 1], c="b")
plt.title("Nube de 50 puntos usando simula_unif con dimensión 2 en el rango [-50, 50]")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


plt.clf()

# obtenemos con una distribución normal 50 puntos bidimensionales en [5, 7]
puntos_simula_gaus = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE
plt.scatter(puntos_simula_gaus[:, 0], puntos_simula_gaus[:, 1], c="r")
plt.title("Nube de 50 puntos usando simula_gaus con dimensión 2 y sigma [5, 7]")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

plt.clf()


# dibujamos ambos conjuntos de puntos a la vez en la misma gráfica
plt.scatter(puntos_simula_unif[:, 0], puntos_simula_unif[:, 1], c="b", label="Puntos simula_unif")
plt.scatter(puntos_simula_gaus[:, 0], puntos_simula_gaus[:, 1], c="r", label="Puntos simula_gaus")
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")


plt.title("Nube de 100 puntos, 50 usando simula_unif con dimensión 2 y rango [-50, 50] \ny 50 usando simula_gaus con dimensión 2 y sigma [5, 7]")
plt.legend()

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")





###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1



def f(x, y, a, b):
	return signo(y - a*x - b)


def num_errores_puntos_f(x, y, a, b):

	errores = 0
	for i in range(len(x)):
		if signo(f(x[i][0], x[i][1], a, b)) != y[i]:
			errores += 1

	errores = errores/len(x)

	return errores


# obtenemos los 50 puntos y la recta para etiquetarlos
intervalo_trabajo = [-50, 50]

puntos_2d = simula_unif(100, 2, intervalo_trabajo)

a, b = simula_recta(intervalo_trabajo)

etiquetas = []
posibles_etiquetas = (1, -1)
colores = {1: 'b', -1: 'r'}

# etiquetamos cada punto a corde con la función de la recta
for punto in puntos_2d:
	etiquetas.append(f(punto[0], punto[1], a, b))

# dibujamos los puntos sin etiquetar
plt.scatter(puntos_2d[:, 0], puntos_2d[:, 1], c="b")
plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {}".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")


plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# dibujamos los puntos etiquetados, cada uno de un color
for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)

	plt.scatter(puntos_2d[indice, 0], puntos_2d[indice, 1], c=colores[etiqueta], label="{}".format(etiqueta))



# y = a*x + b
# dibujamos la recta de etiquetado, usando los valores en x_1 de intervalo_trabajo para calcular x_2
plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta obtenida aleatoriamente')


plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {}, etiquetados segun una recta".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


print("Porcentaje de errores en el etiquetado: ", num_errores_puntos_f(puntos_2d, etiquetas, a, b))


input("\n--- Pulsar tecla para continuar ---\n")



# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

# no podemos usar el for, tenemos que sacarlos a la vez
# si hacemos primero unos, al meter el ruido en el positivo, pasan a ser negativo,
# y se les podría llegar a meter ruido otra vez al hacer el ruido del negativo
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

# volvemos a dibujar ahora los conjuntos con ruido

for etiqueta in posibles_etiquetas:
	indice = np.where(np.array(etiquetas) == etiqueta)

	plt.scatter(puntos_2d[indice, 0], puntos_2d[indice, 1], c=colores[etiqueta], label="{}".format(etiqueta))



# y = a*x + b

plt.plot(intervalo_trabajo, [a*intervalo_trabajo[0] + b, a*intervalo_trabajo[1] + b], 'k-', label='Recta obtenida aleatoriamente')

plt.xlabel("Valor x de los puntos obtenidos")
plt.ylabel("Valor y de los puntos obtenidos")

plt.title("Nube de 100 puntos bidimensionales en el intervalo {} {},\n etiquetados segun una recta con ruido (10% de los positivos y 10% de los negativos con ruido)".format(intervalo_trabajo[0], intervalo_trabajo[1]))
plt.legend()
plt.xlim(intervalo_trabajo)
plt.ylim(intervalo_trabajo)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Porcentaje de errores en el etiquetado: ", num_errores_puntos_f(puntos_2d, etiquetas, a, b))

input("\n--- Pulsar tecla para continuar ---\n")







###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

# funciones dadas, usan solo un parametro para que funcione la función dada para dibujar
def f1(x):
	return np.float64((x[:, 0]-10)**2 + (x[:, 1] - 20)**2 - 400)

def f2(x):
	return np.float64( 0.5*(x[:, 0]+10)**2 + (x[:, 1]-20)**2 - 400 )

def f3(x):
	return np.float64( 0.5*(x[:, 0]-10)**2 - (x[:, 1]+20)**2 - 400 )

def f4(x):
	return np.float64( x[:, 1] - 20*x[:, 0]**2 - 5*x[:, 0] + 3 )



def num_errores_puntos_f_n(fun, x, y):

	errores = 0
	val = fun(x)
	for i in range(len(x)):
		if signo(val[i]) != y[i]:
			errores += 1

	errores = errores/len(x)

	return errores

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01

    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
                cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')

    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


#CODIGO DEL ESTUDIANTE

# llamamos a la funcion con el conjunto de puntos del apartado 2b, pero con las distintas funciones de frontera
plot_datos_cuad(puntos_2d, etiquetas, f1, "Funcion f1")

print("Porcentaje de errores en el etiquetado usando f1: ", num_errores_puntos_f_n(f1, puntos_2d, etiquetas))
input("\n--- Pulsar tecla para continuar ---\n")


plot_datos_cuad(puntos_2d, etiquetas, f2, "Funcion f2")

print("Porcentaje de errores en el etiquetado usando f2: ", num_errores_puntos_f_n(f2, puntos_2d, etiquetas))
input("\n--- Pulsar tecla para continuar ---\n")


plot_datos_cuad(puntos_2d, etiquetas, f3, "Funcion f3")

print("Porcentaje de errores en el etiquetado usando f3: ", num_errores_puntos_f_n(f3, puntos_2d, etiquetas))
input("\n--- Pulsar tecla para continuar ---\n")



plot_datos_cuad(puntos_2d, etiquetas, f4, "Funcion f4")

print("Porcentaje de errores en el etiquetado usando f4: ", num_errores_puntos_f_n(f4, puntos_2d, etiquetas))
input("\n--- Pulsar tecla para continuar ---\n")


input("\n--- Pulsar tecla para continuar ---\n")

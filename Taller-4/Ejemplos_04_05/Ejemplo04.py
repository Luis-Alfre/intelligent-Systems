# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import pandas
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#url = "iris.txt"
# Se crea una lista de nombres para las etiquetas del frame
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# Se crea un data frame
dataset = pandas.read_csv(url, names=names)

# Se muestra la dimensión del la tabla 
print(dataset.shape)

# Se muestran los primeros n datos
n = 10
print(dataset.head(n))

# Se muestran las estadísticas de los datos contenidos
print(dataset.describe())

# Se muestra la cantidad de datos agrupados por el atributo class
print(dataset.groupby('class').mean())

# Diagrama de cajas
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
# Histagrama de datos
dataset.hist()
plt.show()

# Se crea una ventana para desplegar la figura
fig = plt.figure('scatter')
# Se genera un manejador para el axes principal
ax = fig.add_subplot(111)
# Se pone titulo al axes
ax.set_title('Setosa vs Versicolor con caracteristicas sepal-length/sepal-width') 


# Se seleccionan los datos cuya clase es setosa
setosa = dataset.loc[dataset['class'] == 'Iris-setosa']

# de la seleccion anterior se toman las dos primeras caracteristicas
slSetosa = setosa.loc[:,'sepal-length']
swSetosa = setosa.loc[:,'sepal-width']
# Se crea el diagrama de puntos con los datos anteriores
ax.scatter(slSetosa,swSetosa, c='r', label = 'Setosa')

# Se seleccionan los datos cuya clase es versicolor
versicolor = dataset.loc[dataset['class'] == 'Iris-versicolor']
# de la seleccion anterior se toman las dos primeras caracteristicas
slVersicolor = versicolor.loc[:,'sepal-length']
swVersicolor = versicolor.loc[:,'sepal-width']

# Se crea el diagrama de puntos con los datos anteriores
ax.scatter(slVersicolor,swVersicolor, c='b', label = 'versicolor')

# Se agregan nombres a los ejes
ax.set_xlabel('sepal-length') 
ax.set_ylabel('sepal-width') 
# Se busca le mejor ubicación para la leyenda.
ax.legend(loc='best')
# Despliegue de la gráfica
plt.show()



# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import numpy as np
from statistics import mode


fileName = 'iris_data.txt'

rawData = open(fileName)

data = np.loadtxt(rawData, delimiter=",")
print(data.shape)
print(data)

# Entradas
# Datos: tabla con las caracteristicas de cada item
# clase: vector que contiene las etiquetas de clase
# x: dato a clasificar
# k: cardinal de vecinos a tener en cuenta

# Salida:
# clase[indice]: El valor de la clase a la que pertenece el valor x.
def kNeighbor(indices, clase):
    clases = []
    for i in range(np.size(indices)):
        clases.append(clase[indices[i]])
    return mode(clases)


def knn(datos, clase, x, k):
    [f,c] = datos.shape
    distancia = np.zeros(f)
    for i in range(f):
        distancia[i] = np.linalg.norm(datos[i,:]-x)
    indices = np.argsort(distancia)[:k]
    result = kNeighbor(indices, clase)
    # print ('////',indices)
    # print('---',clase[indice])
    return result


datos = data[:,0:4]
clase = data[:,4]


# x = data[103,:4].reshape(1,4)

x = np.array([9,3,6,3]).reshape(1,4)

r = knn(datos,clase,x,8)
print(r)
##  1. Complementar el algoritmo anterior para que pueda aplicarse para k>1.


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
train = np.concatenate((data[:40,:], data[50:90,:],data[100:140,:]), axis=0)
test = np.concatenate((data[40:50,:], data[90:100,:],data[140:150,:]), axis=0)



res = np.zeros(30)

k=8
datos = train[:,:4]
clase = train[:,4]
for i in range(30):
    x = test[i,:4]
    print(x)
    res[i]= knn(datos,clase,x,k)

claseTest = test[:,4]
error = (np.sum((res!=claseTest).astype(int))/30)*100
print(error)
##  1. Complementar el algoritmo anterior para que pueda aplicarse para k>1.


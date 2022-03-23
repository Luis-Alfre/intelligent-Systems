# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import numpy as np
import matplotlib.pyplot as plt

fileName = 'iris_data.txt'

rawData = open(fileName)

data = np.loadtxt(rawData, delimiter=",")

print(data.shape)

print(data)

# visualizamos las dos primeras componentes:
plt.figure(1)
plt.suptitle('Datos de Iris')
plt.scatter(data[:,0],data[:,1])


# visualizamos cada clase de un color diferente
plt.figure(2)
plt.suptitle('Iris separadas por clases')
plt.scatter(data[:50,0],data[:50,3],color ='red', label = 'Setosa')
plt.scatter(data[51:100,0],data[51:100,3],color ='blue', label = 'Virginica')
plt.scatter(data[101:150,0],data[101:150,3],color ='green', label = 'Versicolor')
plt.legend()

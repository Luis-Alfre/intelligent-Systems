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
from sklearn.cluster import KMeans

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
plt.scatter(data[:50,1],data[:50,2],color ='red', label = 'Setosa')
plt.scatter(data[50:100,1],data[50:100,2],color ='blue', label = 'Virginica')
plt.scatter(data[100:150,1],data[100:150,2],color ='green', label = 'Versicolor')
plt.legend()

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data[:,:4])
label = kmeans.labels_

c1 = data[label[:]==0]
c2 = data[label[:]==1]
c3 = data[label[:]==2]

plt.figure(3)
plt.suptitle('Iris separadas por clases')
plt.scatter(c1[:,1],c1[:,2],color ='red', label = 'Clase 1')
plt.scatter(c2[:,1],c2[:,2],color ='blue', label = 'Clase 2')
plt.scatter(c3[:,1],c3[:,2],color ='green', label = 'Clase 3')
plt.legend()



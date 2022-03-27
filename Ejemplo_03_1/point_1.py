# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import numpy as np

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

def knn(datos, clase, x, k):
    [f,c] = datos.shape
    distancia = np.zeros(f)
    for i in range(f):
        distancia[i] = np.linalg.norm(datos[i,:]-x)
    indice = int(distancia.argmin())
    #print (distancia)
    #print(indice)
    return clase[indice]

train = np.concatenate((data[:20,:], data[60:70,:],data[110:120,:]), axis=0)

testSub1 = np.concatenate((data[:10,:], data[50:60,:],data[100:110,:]), axis=0)
testSub2 = np.concatenate((data[10:20,:], data[60:70,:],data[110:120,:]), axis=0)
testSub3 = np.concatenate((data[20:30,:], data[70:80,:],data[120:130,:]), axis=0)
testSub4 = np.concatenate((data[30:40,:], data[80:90,:],data[130:140,:]), axis=0)
testSub5 = np.concatenate((data[40:50,:], data[90:100,:],data[140:150,:]), axis=0)

res1 = np.zeros(30)
res2 = np.zeros(30)
res3 = np.zeros(30)
res4 = np.zeros(30)
res5 = np.zeros(30)

k=1
datos = train[:,:4]
clase = train[:,4]

for i in range(30):
    x = testSub1[i,:4]
    res1[i]= knn(datos,clase,x,k)

claseTest = testSub1[:,4]
error = (np.sum((res1!=claseTest).astype(int))/75)*100
print(error)
##  1. Complementar el algoritmo anterior para que pueda aplicarse para k>1.


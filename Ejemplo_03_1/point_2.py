# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

from re import T
import numpy as np
from statistics import mode


fileName = 'iris_data.txt'

rawData = open(fileName)

data = np.loadtxt(rawData, delimiter=",")
#print(data.shape)
#print(data)

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


# testSub1 = np.concatenate((data[:10,:], data[50:60,:],data[100:110,:]), axis=0)
# testSub2 = np.concatenate((data[10:20,:], data[60:70,:],data[110:120,:]), axis=0)
# testSub3 = np.concatenate((data[20:30,:], data[70:80,:],data[120:130,:]), axis=0)
# testSub4 = np.concatenate((data[30:40,:], data[80:90,:],data[130:140,:]), axis=0)
# testSub5 = np.concatenate((data[40:50,:], data[90:100,:],data[140:150,:]), axis=0)

# res1 = np.zeros(30)
# res2 = np.zeros(30)
# res3 = np.zeros(30)
# res4 = np.zeros(30)
# res5 = np.zeros(30)

k = 5

t1 = 0
t2 = 50
t3 = 100

print('k = ',k,'\n')
sumError = 0 
for it in range(5):

    res = np.zeros(30)
    testSub1 = np.concatenate((data[t1:(t1+10),:], data[t2:(t2+10),:],data[t3:(t3+10),:]), axis=0)
    train = np.concatenate((np.delete(data[:50,:], slice(t1,t1+10),0), np.delete(data[50:100,:], slice(t2-50,t2+10-50),0), np.delete(data[100:150,:], slice(t3-100,t3+10-100),0)), axis=0)

    datos = train[:,:4]
    clase = train[:,4]
    for i in range(30):
        x = testSub1[i,:4]
        res[i]= knn(datos,clase,x,k)

    t1+=10
    t2+=10
    t3+=10
    claseTest = testSub1[:,4]
    print('Subconjunto # ', it+1)
    print('Clases reales', claseTest)
    print('Clases Obtenidas', res)
    print('# Desaciertos ', np.sum((res!=claseTest).astype(int)))
    error = (np.sum((res!=claseTest).astype(int))/30)*100
    sumError+=error
    print('Porcentaje error ',error,'% \n')

print('Error promedio', sumError/5)

# claseTest = testSub1[:,4]
# error = (np.sum((res1!=claseTest).astype(int))/75)*100
# print(error)
##  1. Complementar el algoritmo anterior para que pueda aplicarse para k>1.


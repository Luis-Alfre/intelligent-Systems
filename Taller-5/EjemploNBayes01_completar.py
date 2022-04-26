# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import numpy as np
import statistics as st
import math


# Escribir el código que implemente el clasificador naive Bayes de distribución gausiana
# para datos continuos

# Parámetroe de entrada de la fución:
# datos: un array bidmensional (matriz de f x c) con las caracteristicas de los ejemplos disponibles
# clases: un array unidimensional ( vector tamaño f), con las clases a las que pertenece cada ejemplo
# X: un array unidimensional (vector de tamaño c), correpondiente al ejemplo que queremos clasificar


def standardDeviationAndMedia(cat, datos, clases,f,c):
    standarDev = np.zeros((len(cat),c))
    arrayMedia = np.zeros((len(cat),c))
    index = 0
    for i in cat:
        itermClases = np.where(clases == i)
        for data in range(c):
            h = datos[itermClases[0],data:data+1]
            h = h.flatten().tolist()
            standarDev[index][data]=np.std(h)
            arrayMedia[index][data]=np.average(h)
            h = []
        index+=1
        itermClases = []
    return(standarDev, arrayMedia)

def naiveBayesCal(X, standar,media,cat):
    rtas = []
    pi = np.pi
    for x in range(len(X)):
        exp = np.exp(-(np.power((X[x]-media[x]),2)/(2*np.power(standar[0],2))))
        rta = (1/(np.sqrt(2*pi*standar[x])))*(exp)
        rtas.append(rta)
    rtas.append(1/len(cat))
    return np.prod(rtas)
    


def naiveBayesGausiano(datos,clases,X):
    [f,c] = datos.shape
    # se crea un vector con las etiquetas de clases
    cat = np.unique(clases)
    claseX = 0
    [standar,media]=standardDeviationAndMedia(cat,datos,clases,f,c)
    rta = []
    for c in range(len(cat)):
        rta.append(naiveBayesCal(X, standar[c],media[c],cat))
    rta = np.argsort(rta)
    claseX = cat[rta[len(rta)-1]]
    return claseX



# Probar el clasificador para la base de datos de iris

# Probar el clasificador para la base de datos de vino
    
####################### Programa Principal ########################
    

def validation(data):
    train = np.concatenate((data[:25,:], data[50:75,:],data[100:125,:]), axis=0)
    test = np.concatenate((data[25:50,:], data[75:100,:],data[125:150,:]), axis=0)

    res = np.zeros(75)

    datos = train[:,:4]
    clase = train[:,4]
    for i in range(75):
        x = test[i,:4]
        res[i]= naiveBayesGausiano(datos,clase,x)

    claseTest = test[:,4]
    # print('Clases reales', claseTest)
    # print('Clases Obtenidas', res)
    print('# Desaciertos ', np.sum((res!=claseTest).astype(int)))
    error = (np.sum((res!=claseTest).astype(int))/75)*100
    print('Porcentaje error ',error,'%')



def validationCross(data):
    t1 = 0
    t2 = 50
    t3 = 100

    sumError = 0 
    for it in range(5):

        res = np.zeros(30)
        train = np.concatenate((data[t1:(t1+10),:], data[t2:(t2+10),:],data[t3:(t3+10),:]), axis=0)
        test = np.concatenate((np.delete(data[:50,:], slice(t1,t1+10),0), np.delete(data[50:100,:], slice(t2-50,t2+10-50),0), np.delete(data[100:150,:], slice(t3-100,t3+10-100),0)), axis=0)

        datos = test[:,:4]
        clase = test[:,4]
        for i in range(30):
            x = train[i,:4]
            res[i]= naiveBayesGausiano(datos,clase,x)

        t1+=10
        t2+=10
        t3+=10
        claseTest = train[:,4]
        print('Subconjunto # ', it+1)
        # print('Clases reales', claseTest)
        # print('Clases Obtenidas', res)
        print('# Desaciertos ', np.sum((res!=claseTest).astype(int)))
        error = (np.sum((res!=claseTest).astype(int))/30)*100
        sumError+=error
        print('Porcentaje error ',error,'% \n')

    print('Error promedio', sumError/5)




filename = 'iris_data.txt'
raw_data = open(filename)
data = np.loadtxt(raw_data, delimiter=",")

datos = data[:,:4]
clases = data[:,4]
X = np.array([5.8,3.1,4.9,1.5])

prediccion = naiveBayesGausiano(datos,clases,X)
print("La clase de X es ",prediccion, "\n ")

print("*** Validación *** \n")
validation(data)

print("\n *** Validación cuzada con 5 Subconjutos *** \n")

validationCross(data)

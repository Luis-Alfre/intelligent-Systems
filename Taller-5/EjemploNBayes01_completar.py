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



# Escribir el código que implemente el clasificador naive Bayes de distribución gausiana
# para datos continuos

# Parámetroe de entrada de la fución:
# datos: un array bidmensional (matriz de f x c) con las caracteristicas de los ejemplos disponibles
# clases: un array unidimensional ( vector tamaño f), con las clases a las que pertenece cada ejemplo
# X: un array unidimensional (vector de tamaño c), correpondiente al ejemplo que queremos clasificar

def naiveBayesGausiano(datos,clases,X):
    [f,c] = datos.shape
    # se crea un vector con las etiquetas de clases
    cat = np.unique(clases)
    claseX = 0
    
    # continuar...
    
    return claseX



# Probar el clasificador para la base de datos de iris

# Probar el clasificador para la base de datos de vino
    
####################### Programa Principal ########################
    
filename = 'iris_data.txt'
raw_data = open(filename)
data = np.loadtxt(raw_data, delimiter=",")
print(data.shape)
print(data[:5,:])

datos = data[:,:4]
clases = data[:,4]
X = np.array([5.8,3.1,4.9,1.5])

prediccion = naiveBayesGausiano(datos,clases,X)
print(" La clase de X es ",prediccion)




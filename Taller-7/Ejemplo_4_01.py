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

######## Escribir la funcion hipótesis #####

# Entradas:
# th0, th1, parametros de la funcion
# x: variable a evaluar

# Salidas: s: resultado de h(th0,th1,x)

def hipotesis(th0,th1,x):
    
    s = th0 + th1*x
    # calculo de la función
    # s =  expresion
    
    return s

####### Escribir la funcion de coste ###########
    
# Entradas:
# th0, th1, variables de la funcion de coste
# x,y: entrada y salida de los datos a aproximar

# Salida
# c: valor del coste de la aproximación para los valores th0, th1 de entrada
    
def costeJ(th0,th1,x,y):
    m = len(x) 
    error = 0.0
    for i in range(m):
        error += ((hipotesis(th0,th1,x[i])-y[i])**2)    
    return error/(2*m)

######### Escribir la función que impleenta el descenso por el gradiente
    
def descensoPorGradiente(th0,th1,x,y, alpha, iterm):
    m = len(x)
    arrayCoste = []
    for j in range(iterm):
        dervA = 0
        dervB = 0
        for i in  range(m):
            h = hipotesis(th0, th1,x[i])
            dervA += h - y[i]
            dervB += (h- y[i])*x[i] 
            arrayCoste.append(costeJ(th0,th1,x,y))
        th0 -=  (dervA / m) * alpha 
        th1 -=  (dervB / m) * alpha 
        
    return th0,th1, arrayCoste


#### Funciones para trazar las gráficas de h (función hipóteis) y J, (función error) #####

def graficarRecta(th0,th1,x1,x2, x, y):
    
    plt.scatter(x,y,marker='o')
    y1 = hipotesis(th0,th1,x1)
    y2 = hipotesis(th0,th1,x2)
    plt.plot([x1,x2],[y1,y2],color='r')
    plt.show()


def graficarError(arrayCoste):
    x = range(len(arrayCoste))
    plt.plot(x, arrayCoste)
    plt.show()
    
##################### El programa Principal ######################

# Cargamos los datos
nombreArchivo = 'Inmobiliaria.csv'

def loadData(nombreArchivo):
    datosArchivo = open(nombreArchivo)
    datosInmobiliaria = np.loadtxt(datosArchivo, delimiter=";",  dtype=np.float128)
    print(datosInmobiliaria.shape)
    print(datosInmobiliaria)
    x = datosInmobiliaria[:,0]
    y = datosInmobiliaria[:,1]
    return x,y

x,y = loadData(nombreArchivo)

# Gráfico del problema 
plt.scatter(x,y,marker='o')
plt.show()


# Generación de una hipótesis inicial
th0 = 100000
th1 = 20
graficarRecta(th0,th1,0,3000, x, y)

# establecemos alpha y el numero de iteraciones
alpha = 0.00000001
iterm = 500
th0, th1, arrayCoste = descensoPorGradiente(th0,th1,x,y, alpha, iterm)
print(th0)
print(th1)

graficarRecta(th0,th1,0,max(x),x,y)

graficarError(arrayCoste)





# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Cargamos el archivo de imagen

img1 = cv.imread('fire2.tif',1)

# convertimos la imagen a RGB

img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# Desplegamos la imagen
plt.imshow(img2)
plt.xticks([]), plt.yticks([])  # Oculta los valores en X e Y.
plt.show()

img2.shape

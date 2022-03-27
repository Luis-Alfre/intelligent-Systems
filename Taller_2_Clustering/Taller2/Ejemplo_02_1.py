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

img2 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)



# Desplegamos la imagen
plt.imshow(img2)
plt.xticks([]), plt.yticks([])  # Oculta los valores en X e Y.
plt.show()

colors = np.array([[0,0,255],
                   [255, 255, 0], 
                   [0, 255, 255]], 'uint8')

w, h, d = original_shape = img2.shape
img = img2.reshape(w*h, d) 

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(img)


cluster_center = kmeans.cluster_centers_
cluster_labels = kmeans.labels_


plt.figure(2)
plt.suptitle('k-means')
plt.imshow(colors[cluster_labels].reshape(w, h, d))
plt.show()



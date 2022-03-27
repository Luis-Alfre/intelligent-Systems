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

w, h, d = original_shape = img2.shape
img = img2.reshape(w*h, d) 


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(img)


cluster_center = kmeans.cluster_centers_
cluster_center = np.array(cluster_center, 'uint8')
cluster_labels  = kmeans.labels_

# Desplegamos la imagen
plt.imshow(img2)
plt.xticks([]), plt.yticks([])  # Oculta los valores en X e Y.
plt.show()

s = cluster_center[cluster_labels].reshape(w,h,d)
print ()
plt.figure(2)
plt.suptitle('k-means')
plt.imshow(cluster_center[cluster_labels].reshape(w,h,d))
plt.scatter(data[:50,0],data[:50,3],color ='red', label = 'Setosa')
plt.show()



# -*- coding: utf-8 -*-
"""
Universidad de Pamplona
Facultad de Ingenierías y Arquitectura
Programa de Ingenieria de Sistemas
Asignatura: Sistemas Inteligentes
Profesor: Jose Orlando Maldonado Bautista
#"""

import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#url = "D:\\Orlando\\2019-1\\ElectivaProfesional\\Python\\IntroML\\iris.txt"
# Se crea una lista de nombres para las etiquetas del frame
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# Se crea un data frame
dataset = pandas.read_csv(url, names=names)

# Se cre un arreglos que contenga solo los valores numericos de la tabla
array = dataset.values
# Las primeras 4 columnas que corresponden a las caracteristica se almacenan en X
X = array[:,0:4]
# La columna 5, que corresponde a la clase, se almacena en Y
Y = array[:,4]

# Se define el tamaño del conjunto de validación, quedando este en 20%
validation_size = 0.30
seed = 5
# Se realiza la división de los datos en los conjuntos de prueba y entrenamiento
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                    test_size=validation_size, random_state=seed)

scoring = 'accuracy'


# Hacer predicciones sobre el conjunto de prueba, independiente del conjunto de entrenamiento
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("El procentaje de acierto es: %s" % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))

############################# Validación cruzada #################

kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score( KNeighborsClassifier(), 
                                             X, Y, cv=kfold, scoring=scoring)
msg = "%s: \n porcentaje medio de acierto (%f)  - Desviacion estandar (%f)" % (
        "Resultado clasificador KNN", cv_results.mean(), cv_results.std())
print(msg) 





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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

url = "cmc.data"


def getData(url):
    # Se crea un data frame
    dataset = pandas.read_csv(url)
    names = ['wife-age', 'wife-aeducation', 'husband-edcation', 'number-children', 'wife-religion', 'wife-working', 'husband-occupation', 
             'standard-living','media-exposure', 'method']
    # Se crea un data frame
    dataset = pandas.read_csv(url, names=names)
    return dataset


def graph(dataset, caracteritica1, caracteristica2):

    fig = plt.figure('scatter')

    ax = fig.add_subplot(111)
    titulo = 'M1 vs M2 vs M3 con caract. ', caracteritica1,  'y', caracteristica2

    ax.set_title(titulo)

    c1 = dataset.loc[dataset['method'] == 1]
    carac1 = c1.loc[:, caracteritica1]
    carac2 = c1.loc[:, caracteristica2]
    ax.scatter(carac1, carac2, c='r', label='metodo 1')

    c2 = dataset.loc[dataset['method'] == 2]
    carac1 = c2.loc[:, caracteritica1]
    carac2 = c2.loc[:, caracteristica2]
    ax.scatter(carac1, carac2, c='b', label='metodo 2')
    
    
    c3 = dataset.loc[dataset['method'] == 3]
    carac1 = c3.loc[:, caracteritica1]
    carac2 = c3.loc[:, caracteristica2]
    ax.scatter(carac1, carac2, c='y', label='metodo 2')

    ax.set_xlabel(caracteritica1)
    ax.set_ylabel(caracteristica2)
    ax.legend(loc='best')
    plt.show()


def statistics(dataset):
    print(dataset.describe())
    print(dataset.groupby('status').mean())
    dataset.plot(kind='box', subplots=True, layout=(
        3, 8), sharex=False, sharey=False)
    plt.show()

    dataset.hist()
    plt.show()


def knnVsGaus(dataset, seed, validationSize):

    array = dataset.values
    # clases
    Y = array[:, 16]
    array = (dataset.drop(['status'], axis=1)).values
    # datos
    X = array
    # Se realiza la división de los datos en los conjuntos de prueba y entrenamiento
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,
                                                                                    test_size=validationSize, random_state=seed)
    scoring = 'accuracy'
    # Hacer predicciones sobre el conjunto de prueba, independiente del conjunto de entrenamiento
    knn(X_train, Y_train,X_validation,Y_validation,seed,X,Y,scoring)
    Gaus(X_train, Y_train,X_validation,Y_validation,seed,X,Y,scoring)


def knn(X_train, Y_train, X_validation, Y_validation, seed, X, Y, scoring):

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, Y_train)

    predictions = knn.predict(X_validation)

    print("======= VALIDACION MEDIANTE CONJUNTOS KNN: ENTRENAMIENTO - TEST =========\n")
    print("El procentaje de acierto mediante KNN es: %s" %
          accuracy_score(Y_validation, predictions))
    print(" \n Matriz de confusión: \n ")
    print(confusion_matrix(Y_validation, predictions))
    
    # Se obtienen métricas de clasificación
    print(classification_report(Y_validation, predictions))

    print("\n ============ VALIDACION CRUZADA KNN =========== \n")
    
    kfold = model_selection.StratifiedKFold(
        n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(KNeighborsClassifier(),
                                                 X, Y, cv=kfold, scoring=scoring)
    msg = "%s: \n porcentaje medio de acierto (%f)  \n Desviacion estandar (%f) \n" % (
        "Resultado clasificador KNN", cv_results.mean(), cv_results.std())
    print(msg)



def Gaus(X_train, Y_train, X_validation, Y_validation, seed, X, Y, scoring):

    # Se crea una instancia del clasificador
    NB = GaussianNB()
    # Se entrena el clasificador
    NB.fit(X_train, Y_train)
    # Se realiza una predicción sobre los datos de validación
    predictions = NB.predict(X_validation)
    print("\n ======= VALIDACION MEDIANTE CONJUNTOS GAUS: ENTRENAMIENTO - TEST =========")
    print("El procentaje de acierto  mediante GNB es: %s" %
          accuracy_score(Y_validation, predictions))
    print(" \n Matriz de confusión: \n ")
    print(confusion_matrix(Y_validation, predictions))
    
    # Se obtienen métricas de clasificación
    print(classification_report(Y_validation, predictions))
    
    print("\n ============ VALIDACION CRUZADA =========== \n")
    
    kfold = model_selection.StratifiedKFold(
        n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(
        NB, X, Y, cv=kfold, scoring=scoring)
    
    msg = "%s: \n porcentaje medio de acierto (%f)  \n Desviacion estandar (%f) \n" % (
        "Resultado clasificador GNB es ", cv_results.mean(), cv_results.std())
    print(msg)


dataset = getData(url)
print(dataset)
raph(dataset,'MDVP:Fhi(Hz)','MDVP:Flo(Hz)')
#graph(dataset,'MDVP:Jitter(Abs)','MDVP:Shimmer(dB)')
#graph(dataset,'MDVP:Shimmer(dB)','MDVP:Flo(Hz)')
#graph(dataset,'Shimmer:APQ5','MDVP:Flo(Hz)')
#graph(dataset,'MDVP:Fhi(Hz)','Shimmer:APQ5')
#graph(dataset,'spread1','MDVP:Flo(Hz)')
#graph(dataset,'MDVP:Fhi(Hz)','spread1')
#graph(dataset,'spread1','spread2')




#statistics(dataset)

#validationSize = 0.30
#seed = 400
#knnVsGaus(dataset, seed, validationSize)


"""
La matriz de confusión es una herramienta que permite analizar el desempeño de un clasificador. 
Permite evidenciar en que casos la matriz confunde la clases.

Apartir de ella se obtienen las siguientes métricas de desempeño:

La precisión es la relación tp / (tp + fp) 
donde tp es el número de positivos verdaderos y fp el número de falsos positivos. 
La precisión es intuitivamente la capacidad del clasificador de no etiquetar 
como positiva una muestra que es negativa.

El recuerdo es la relación tp / (tp + fn) 
donde tp es el número de verdaderos positivos y fn el número de falsos negativos. 
El recuerdo es intuitivamente la capacidad del clasificador para encontrar 
todas las muestras positivas.

El valor F se considera como una media armónica que combina los valores de la 
precisión y de la exhaustividad. F = 2(P*R)/(P+R).

El soporte es el número de ocurrencias de cada clase en y_true (conjunto de salida).

"""

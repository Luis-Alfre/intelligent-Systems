from operator import index
import pandas
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


url = "wine.data.txt"
# Se crea una lista de nombres para las etiquetas del frame
names = ['class', 'Alcohol', 'Malic-acid', 'Ash', 'Alcalinity-of-ash', 'Magnesium', 'Total-phenols', 'Flavanoids',
'Nonflavanoid-phenols', 'Proanthocyanins', 'Color-intensity', 'Hue', 'OD280/OD315', 'Proline']
# Se crea un data frame
dataset = pandas.read_csv(url, names=names)

# print(dataset.shape)
# # Se muestran los primeros n datos
# n = 10
# print(dataset.head(n))

def graficar(dataset, caracteritica1, caracteristica2):
    fig = plt.figure('scatter')

    ax = fig.add_subplot(111)
    titulo = 'c1 vs c2 vs c3 con caract. ' , caracteritica1,  'y' , caracteristica2

    ax.set_title(titulo) 

    cul1 = dataset.loc[dataset['class'] == 1]
    AlcoholCul1 = cul1.loc[:,caracteritica1]
    ColorCul1 = cul1.loc[:,caracteristica2]
    ax.scatter(AlcoholCul1,ColorCul1, c='r', label = 'cultivar 1')

    cul2 = dataset.loc[dataset['class'] == 2]
    AlcoholCul2 = cul2.loc[:,caracteritica1]
    ColorCul2 = cul2.loc[:,caracteristica2]
    ax.scatter(AlcoholCul2,ColorCul2, c='b', label = 'cultivar 2')

    cul3 = dataset.loc[dataset['class'] == 3]
    AlcoholCul3 = cul3.loc[:,caracteritica1]
    ColorCul3 = cul3.loc[:,caracteristica2]
    ax.scatter(AlcoholCul3,ColorCul3, c='y', label = 'cultivar 3')

    ax.set_xlabel(caracteritica1) 
    ax.set_ylabel(caracteristica2) 
    ax.legend(loc='best')
    plt.show()

def knn(dataset):

    array = dataset.values
    X = array[:,1:]
    Y = array[:,0]

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
    print(predictions)
    print(Y_validation)
    print("El procentaje de acierto es: %s" % accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))

def validacionCruzada(dataset):

    array = dataset.values
    X = array[:,1:]
    Y = array[:,0]

    validation_size = 0.30
    seed = 5
    # Se realiza la división de los datos en los conjuntos de prueba y entrenamiento
    scoring = 'accuracy'


    kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score( KNeighborsClassifier(), 
                                             X, Y, cv=kfold, scoring=scoring)
    msg = "%s: \n porcentaje medio de acierto (%f)  - Desviacion estandar (%f)" % (
        "Resultado clasificador KNN", cv_results.mean(), cv_results.std())
    print(msg) 

def kmeans(dataset):

    array = dataset.values
    data = array[:,1:]
    clases = array[:,0]

    # Hacer predicciones sobre el conjunto de prueba, independiente del conjunto de entrenamiento
    kmeans = KMeans(n_clusters=3,random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_ 
    datos = []

    for label in labels:
        datos.append(label+1)
    
    print("El procentaje de acierto es: %s" % accuracy_score(clases, (datos)))
    print('# Desaciertos ', np.sum((datos!=clases).astype(int)))
    error = (np.sum((datos!=clases).astype(int))/178)*100
    print('Porcentaje error ',error,'%')


    c1 = array[labels[:]==0]
    c2 = array[labels[:]==1]
    c3 = array[labels[:]==2]

    plt.figure(3)
    plt.suptitle('Agurpamiento por kmeans')
    plt.scatter(c1[:,1],c1[:,2],color ='red', label = 'Clase 1')
    plt.scatter(c2[:,1],c2[:,2],color ='blue', label = 'Clase 2')
    plt.scatter(c3[:,1],c3[:,2],color ='y', label = 'Clase 3')
    plt.legend()
    plt.show()

def estadisticas(dataset):
    print(dataset.describe())
    print(dataset.groupby('class').mean())
    dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
    plt.show()

    dataset.hist()
    plt.show()

  

# graficar(dataset,'Alcohol','Proanthocyanins')
# graficar(dataset,'Alcohol','Ash')
# graficar(dataset,'Alcohol','OD280/OD315')
# graficar(dataset,'Alcohol','Flavanoids')
# graficar(dataset,'Magnesium','Flavanoids')
# graficar(dataset,'Alcalinity-of-ash','Nonflavanoid-phenols')


# kmeans(dataset)

#estadisticas(dataset)

#knn(dataset)

#validacionCruzada(dataset)



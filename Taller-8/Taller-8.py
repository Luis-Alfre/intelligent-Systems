
# Importar las librerías necesarias
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# 1. Leer el conjunto de datos a un dataframe de Pandas
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
#df['target'] = housing.target_names
df.head()

df['Target'] = housing.target

print(df.head())
print(housing.target_names)
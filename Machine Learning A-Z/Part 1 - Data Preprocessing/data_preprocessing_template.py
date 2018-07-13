# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = LabelEncoder()
X[:,0] = encoder.fit_transform(X[:,0])


one_Hot_Encoder = OneHotEncoder(categorical_features = [0])
X = one_Hot_Encoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
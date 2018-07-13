import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from minisom import MiniSom
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


data_File_Path = 'Credit_Card_Applications.csv'
data_File_Path = os.path.join( 'Self_Organizing_Maps', data_File_Path)

dataset = pd.read_csv(data_File_Path)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

sc = MinMaxScaler( feature_range= (0,1))
X = sc.fit_transform(X)

###################### SOM ###################################
from minisom import MiniSom

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate (X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
        markers[y[i]],
        markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2
        )

show()


###################### SOM Frauds ###################################

mappings = som.win_map(X)

# frauds = np.concatenate( (mappings[(4,2)], mappings[(1,4)]), axis = 0)

frauds =  mappings[(2,1)]
frauds = sc.inverse_transform(frauds)



#################### Supervised Deep Learning #########################

customers = dataset.iloc[:,1:].values

is_fraud = np.zeros(len(dataset))


for i in range (0, len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
classifier = Sequential()

# Adding input layer and first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15 ))
classifier.add(Dropout(p = 0.10))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid' ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)


y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[ : ,0:1].values, y_pred), axis = 1)

y_pred = y_pred[y_pred[:,1].argsort()]








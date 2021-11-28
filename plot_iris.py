from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

#carica dataset
iris = load_iris()

#indici delle caratteristiche nel grafo
x_index = 0
y_index = 1

#formato colorbar con i nomi delle tipologie di iris
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

#plot del grafo
plt.figure(figsize=(5, 4))                                                 #dimensioni plot
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)   #dati
plt.colorbar(ticks=[0, 1, 2], format=formatter)                            #color bar
plt.xlabel(iris.feature_names[x_index])                                    #lunghezza sepalo etichetta
plt.ylabel(iris.feature_names[y_index])                                    #larghezza sepalo etichetta

plt.tight_layout()

plt.show()
#Algoritmo K-means con plot

from sklearn import datasets                       #per caricare i dati
import matplotlib.pyplot as plt                    #per visualizzare i dati
from sklearn.cluster import KMeans                 #per addestrare il modello
from sklearn.metrics import silhouette_score       #per determinare la bontà dell'algoritmo

#Caricare il dataset
iris = datasets.load_iris()

X = iris.data[:, :2]            #input
y = iris.target                 #output

#Visualizzare il dataset
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Lunghezza sepale', fontsize=18)
plt.ylabel('Larghezza sepale', fontsize=18)

#Costruzione del modello k-means
km = KMeans(n_clusters = 3, n_jobs = 4, random_state = 0)
km.fit(X)

#Analizzare i punti centrali dei vari cluster
centers = km.cluster_centers_
print(centers)                  #Punti centrali

#Etichette iris
new_labels = km.labels_
print("\nStampa etichette Predette")
print(new_labels)               #0 Setosa, 1 Versicolor, 2 virginica   Etichette Predette
print("\nStampa etichette Corrette")
print(y)                        #Etichette corrette

#Confronto tra il valore effettivo e il valore previsto dall'algoritmo k-means
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='brg', edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='cool', edgecolor='k', s=150)

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='yellow', marker='*', edgecolor='k', s=300)

axes[0].set_xlabel('Lunghezza sepale', fontsize=18)
axes[0].set_ylabel('Larghezza sepale', fontsize=18)

axes[1].set_xlabel('Lunghezza sepale', fontsize=18)
axes[1].set_ylabel('Larghezza sepale', fontsize=18)

axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

axes[0].set_title('Effettivo', fontsize=18)
axes[1].set_title('Previsto', fontsize=18)
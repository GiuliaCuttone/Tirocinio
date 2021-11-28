from sklearn import datasets                       #per caricare i dati
from sklearn.cluster import KMeans                 #per addestrare il modello
from yellowbrick.cluster import KElbowVisualizer   #per calcolare il punto di gomito

#Caricare il dataset
iris = datasets.load_iris()

X = iris.data[:, :2]            #input
y = iris.target                 #output

#Costruzione del modello k-means
km = KMeans(n_clusters = 3, n_jobs = 4, random_state = 0)
km.fit(X)

#Analizzare i punti centrali dei vari cluster
centers = km.cluster_centers_

#Etichette iris
new_labels = km.labels_

#Analisi Silhouette
visualizer = KElbowVisualizer(km, k=(2,6), metric='silhouette', timings=False)
visualizer.fit(X)
visualizer.poof()
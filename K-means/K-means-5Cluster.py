#Algoritmo K-means con 5 Cluster

from sklearn import datasets                       #per caricare i dati
from sklearn.cluster import KMeans                 #per addestrare il modello
import numpy as np
#per la matrice di confusione e classificazione
from sklearn.metrics import confusion_matrix, classification_report
import pylab as pl

def swap(array1, array2, start, size, r):
    #Scambia etichette
    label1 = etichette(array1, start, size)
    label2 = etichette(array2, start, size)
    
    if label1 != label2:
        for i in range (0, r):
            if array1[i] == label1:
                array1[i] = label2
            elif array1[i] == label2:
                array1[i] = label1
    else: return

def etichette(array, start, size):
    #Conta occorrenze
    zero = np.sum(np.array(array[start:size]) == 0)
    uno = np.sum(np.array(array[start:size]) == 1)
    due = np.sum(np.array(array[start:size]) == 2)
   
    #Restituisce l'occorrenza più frequente
    if zero > uno and zero > due:
        return 0
    elif uno > zero and uno > due:
        return 1
    else:
        return 2

def riduci(array, n, m):
    label = etichette(array, n, m)
    
    for j in range(n, m):
        if (new_labels[j] == 3) or (new_labels[j] == 4):
            new_labels[j] = label

#Caricare il dataset
iris = datasets.load_iris()

X = iris.data          #input
y = iris.target        #output

pattern = y.size       #numero di pattern
pattern_corretti = 0   #numero di pattern corretti

print(X)

#Costruzione del modello k-means
km = KMeans(n_clusters = 5, n_jobs = 4, random_state = 0)
km.fit(X)

#Analizzare i punti centrali dei vari cluster
centers = km.cluster_centers_
print("\nI punti centrali dei cluster sono:")
print(centers)                  #Punti centrali

#Etichette iris
new_labels = km.labels_

#Sistema etichette
n = 0
m = int(pattern/3)

for i in range(1, 3):
    m *= i
    swap(new_labels, y, n, m, pattern)
    n = m

#Riduce cluster
n = 0
m = int(pattern/3)
    
for i in range(1, 3):
    m *= i
    riduci(new_labels, n, m)
    n = m
riduci(new_labels, n, pattern) 
    
#Conta le etichette corrette
pattern_corretti = np.sum(np.array(y) == np.array(new_labels))

print("\nStampa etichette Predette")
print(new_labels)               #0 Setosa, 1 Versicolor, 2 Virginica   Etichette Predette
print("\nStampa etichette Corrette")
print(y)                        #Etichette corrette

count = 0
print("\n")
#Nomi etichette iris
for label in iris.target_names:
    print(count, "=", label)
    count += 1

accuratezza = pattern_corretti / pattern
errore = 1 - accuratezza

print("\n", pattern_corretti, "pattern corretti su", pattern)
print("Accuratezza:", accuratezza*100, "%")
print("Errore:", errore)

cm = confusion_matrix(y, new_labels)
pl.matshow(cm) 
pl.title('Matrice di confusione\n') 
pl.colorbar() 
pl.show()

print(classification_report(y, new_labels))
from sklearn.datasets import load_iris

#carica dataset
iris = load_iris()

#stampa i nomi delle caratteristiche
print(iris.feature_names)

#stampa i dati
print(iris.data)

#stampa nomi iris
print(iris.target_names)

#stampa i target
print(iris.target)
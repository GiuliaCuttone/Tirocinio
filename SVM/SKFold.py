#K-Folds Cross Validation

from sklearn import svm                               #per creare il vettore
from sklearn import datasets                          #per caricare i dati
from sklearn.model_selection import StratifiedKFold   #KFold
from sklearn.metrics import matthews_corrcoef         #MCC score
from sklearn.metrics import accuracy_score            #accuracy

#per la matrice di confusione e classificazione
from sklearn.metrics import confusion_matrix, classification_report
import pylab as pl


def calcola(X, y):
    for train_index, test_index in kf.split(X, y):
        #separa le variabili per l'addestramento da quelle per il test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
    
        score = accuracy_score(y_test, pred)
   
        MCC = matthews_corrcoef(y_test, pred)
    
        print("\n\nEtichette Corrette")    
        print("TRAIN:", y_train, "\nTEST:", y_test)
    
        print("\nEtichette Predette")
        print("TEST:", pred)                 #0 Setosa, 1 Versicolor, 2 Virginica   Etichette Predette
    
        cm = confusion_matrix(y_test, pred)
        pl.matshow(cm) 
        pl.title('Matrice di confusione\n') 
        pl.colorbar() 
        pl.show() 
    
        print(classification_report(y_test, pred))
    
        print("Accuracy:", score)
        print("Matthew Coefficient:", MCC)


#carica il dataset
iris = datasets.load_iris()
X = iris.data            #input
y = iris.target          #output

clf = svm.SVC(kernel='linear', C=1)     #SVM
kf = StratifiedKFold(n_splits=5)        #K-Folds

calcola(X, y)
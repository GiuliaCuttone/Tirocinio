#Dataset iris suddiviso in training e test
#40 per l'addestramento e 10 per il test

from sklearn import svm, datasets                     #per caricare i dati e creare il vettore
from sklearn.model_selection import train_test_split  #train e test
from sklearn.metrics import matthews_corrcoef         #MCC score
from sklearn.metrics import accuracy_score            #accuracy

#per la matrice di confusione e classificazione
from sklearn.metrics import confusion_matrix, classification_report
import pylab as pl


def evaluate(clf, X, y):
    pred = clf.predict(X)
    
    MCC = matthews_corrcoef(y, pred)
    
    stampa(clf, X, y, MCC, pred)

def stampa(clf, X, y, MCC, pred):
    print("\nStampa etichette Predette")
    print(pred)                          #0 Setosa, 1 Versicolor, 2 Virginica   Etichette Predette
    print("\nStampa etichette Corrette")
    print(y)                        #Etichette corrette

    cm = confusion_matrix(y, pred)
    pl.matshow(cm) 
    pl.title('Matrice di confusione\n') 
    pl.colorbar() 
    pl.show() 
    
    print(classification_report(y, pred))
    
    print("Accuracy:", accuracy_score(y, pred))
    print("Matthew Coefficient:", MCC)
    

#carica dataset
X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=11)

print("\nLINEARE")
clf1 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
evaluate(clf1, X_test, y_test)

print("\n\nRBF")

clf2 = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X_train, y_train)
evaluate(clf2, X_test, y_test)
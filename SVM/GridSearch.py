import dask_ml.model_selection as dcv
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split  #train e test
from sklearn.metrics import matthews_corrcoef         #MCC score
from sklearn.metrics import accuracy_score            #accuracy

#per la matrice di confusione e classificazione
from sklearn.metrics import confusion_matrix, classification_report
import pylab as pl


#carica il dataset
iris = datasets.load_iris()
X = iris.data            #input
y = iris.target          #output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, random_state=11)

parameters = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma':[0.1, 0.01], 'C': [1, 10]}
svc = svm.SVC()

clf = dcv.GridSearchCV(svc, parameters)
best_model = clf.fit(X_train, y_train)

print("\nAccuracy del modello migliore: %.4f" %best_model.best_score_)
print(best_model.best_params_)

pred = best_model.predict(X_test)

MCC = matthews_corrcoef(y_test, pred)

print("\nStampa etichette Predette")
print(pred)                          #0 Setosa, 1 Versicolor, 2 Virginica   Etichette Predette
print("\nStampa etichette Corrette")
print(y_test)                        #Etichette corrette

cm = confusion_matrix(y_test, pred)
pl.matshow(cm) 
pl.title('Matrice di confusione\n') 
pl.colorbar() 
pl.show() 
    
print(classification_report(y_test, pred))
    
print("Accuracy:", accuracy_score(y_test, pred))
print("Matthew Coefficient:", MCC)
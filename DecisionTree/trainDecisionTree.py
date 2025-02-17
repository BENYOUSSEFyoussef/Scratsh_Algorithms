import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from DecisionTree import DecisionTree
from sklearn.preprocessing import StandardScaler
data = datasets.load_breast_cancer()
X,y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = DecisionTree()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
def accuracy(y_test, predictions):
    return np.sum(predictions ==  y_test)/len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

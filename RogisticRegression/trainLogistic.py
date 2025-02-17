import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
bc =datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y, color ="b", marker="o", s=30)
plt.show()

clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred= clf.predict(X_test)
def accuracy (y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

print(accuracy(y_pred,y_test))
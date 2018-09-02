# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

#loading datasets
kyphosis = pd.read_csv('kyphosis.csv')
print(kyphosis[:5])

# extracting independent variables
X = kyphosis.drop('Kyphosis',axis=1) # removing column with axis=1 NB axis=0 removes rows

# extracting dependent variable
y = kyphosis['Kyphosis']

# visualize the data
sns.barplot(x='Kyphosis',y='Age',data=kyphosis)

sns.pairplot(kyphosis,hue='Kyphosis',palette='Set1')

sns.catplot(x='Start',y='Number',hue='Kyphosis',data=kyphosis)

plt.figure(figsize=(25,7))
sns.countplot(x='Age',hue='Kyphosis',data=kyphosis,palette='Set1')

# splitting dataset into training and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=40)

# loading model and training it
from sklearn.tree import DecisionTreeClassifier
decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)

# predicting with model
y_pred = decision_tree.predict(X_test)

# evaluating the model
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))

# confusion matrix
print(confusion_matrix(y_test,y_pred))

# Accuracy
accuracy = (15+2)/len(y_pred)
print(accuracy)

# misclassification rate
mis_cla_rate = (1+3)/len(y_pred)
print(mis_cla_rate)

# Using Random Forest Algorithm(RFC)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

rfc_pred  = rfc.predict(X_test)
print(rfc_pred)

# evaluating the RFC model
print(classification_report(y_test,rfc_pred))

# RFC confusion matrix
print(confusion_matrix(y_test,rfc_pred))

# RFC Accuracy
rfc_accuracy = (17+1)/len(rfc_pred)
print(rfc_accuracy)

# RFC misclassification rate
rfc_mis_cla_rate = (1+2)/len(rfc_pred)
print(rfc_mis_cla_rate)

plt.show()
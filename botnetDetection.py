# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('synthetic_botnet_dataset_large.csv', nrows=50)
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

# If labels are strings, encode them
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Base rate model
def base_rate_model(X):
    return np.zeros(X.shape[0])

y_base_rate = base_rate_model(X_test)

from sklearn.metrics import accuracy_score
print("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
print("Accuracy of logistic model %2.2f" % accuracy_score(y_test, y_pred))

# AUC and reports
from sklearn.metrics import roc_auc_score, classification_report

print("---Base Model---")
base_roc_auc = roc_auc_score(y_test, y_base_rate)
print("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, y_base_rate))

print("---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, y_pred)
print("Logistic Rate AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Relationship Between False Positives & True Positives')
plt.legend(loc='lower right')
plt.show()

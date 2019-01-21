import os
os.chdir('E:\Datacamp\Python\Supervised learning with sklearn')
import pandas as pd
df = pd.read_csv('diabetes.csv')
y = df['Outcome']
X = df.drop('Outcome', axis = 1)


#### METRIC FOR CLASSIFICATION
# Accuracy is not always an informative metric => confusion matrix
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
(tn, fp, fn, tp)

accuracy = (tp+tn)/(tp+tn+fp+fn)
accuracy

precision = tp/(tp + fp)
precision # => Over all PREDICTED positive, how many are predicted positive correct?

recall = tp/(tp+fn)
recall # => Over all REAL positive, how many are predicted positive ?
       # other names: sensitivity, True positive rate(TPR)
       # only depends on numerator because denominator is constant

f1 = 2*precision*recall/(precision + recall)
f1

print(classification_report(y_test, y_pred))


#### LOGISTIC REGRESSION VS KNN
# The same as KNN, just change the estimators
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# => Logreg outperform KNN


#### ROC CURVE
# http://www.navan.name/roc/
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
# predict_proba return the probability rather than 0 and 1
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
# tpr: see above = tp/real positive => depends on tp only
# fpr = fp/(fp+tn) = fp/real negative => depends on fp only
# as threshold decrease, more cases will be predicted positive => tpr and fpr increase
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#### AUC
# http://www.navan.name/roc/
# AUC = 0.5 => ROC is a diagonal line in which TPR = FPR => = random guessing
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv = 5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

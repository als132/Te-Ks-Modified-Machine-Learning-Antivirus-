# Malware Fearture Identification with Machine Learning and Python
# Credit for this code goes to Te-k https://github.com/Te-k)(originally written in Python 2) 
# Project: learning about machine learning and testing algorithms' speed in identifying malware based on feature selection.
# Tested on Windows 10 with python 3.7 and Anaconda using Spyder
# https://www.anaconda.com/
# https://www.python.org/
# https://scikit-learn.org
# Make sure to update Windows path environment and Python
# Updated dependecies and imports as of April 2019
# Updated lines 26-27
# Increased test size and n_estimators

import pandas as pd
import numpy as np
import sklearn.ensemble as ske
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Import data
# Columns separated by '|'
data = pd.read_csv('data.csv', sep='|')
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values


# Feature selection using Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.4)

features = []

print('%i Chosen Features:' % nb_features)

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

# XXX : feature order
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

#Algorithm comparison
algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
        "RandomForest": ske.RandomForestClassifier(n_estimators=100),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=100),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
        "GNB": GaussianNB()
    }
# Test function and svc
results = {}
print("\nTesting algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f%%" % (algo, score*100))
    results[algo] = score

Fastest_Algorithm = max(results, key=results.get)
print('\nFastest algorithm is %s with a %f%% success.' % (Fastest_Algorithm, results[Fastest_Algorithm]*100))


# Identify false and true positive rates
clf = algorithms[Fastest_Algorithm]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate: %f%%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate: %f%%' % ( (mt[1][0] / float(sum(mt[1]))*100)))

import numpy as np
from sklearn import tree
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import graphviz
from sklearn.model_selection import RandomizedSearchCV
from time import time
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit





# specify parameters and distributions to sample from
param_dist = {"max_depth": [1, None],
              "max_features": sp_randint(1, 12),
              "min_samples_split": sp_randint(100, 500),
              "min_samples_leaf": sp_randint(50, 200),
              "criterion": ["gini", "entropy"]}

param_dist_full = {"max_depth": [1, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(100, 500),
              "min_samples_leaf": sp_randint(50, 200),
              "criterion": ["gini", "entropy"]}

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

x = []
y = []
with open('bank-data.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(row[:len(row) - 1])
        y.append(row[len(row) - 1])
f.close()

#---------------this is the full parameter version------------
x_full = []
y_full = []
with open('bank-extra.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        x_full.append(row[:len(row) - 1])
        y_full.append(row[-1])
f.close()
#-----------------------------------------------------



def tree_12_hypertuned():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = tree.DecisionTreeClassifier().fit(x_train,y_train)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=20)
        random_search.fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        report(random_search.cv_results_)
        scores = cross_val_score(random_search.best_estimator_, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", random_search.best_estimator_.score(x_train, y_train)
        st = time()
        test_score = random_search.best_estimator_.score(x_test,y_test)
        print "Tree info: {} nodes.".format(random_search.best_estimator_.tree_.node_count)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def tree_12_default():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = tree.DecisionTreeClassifier().fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Tree info: {} nodes.".format(clf.tree_.node_count)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"


def tree_20_hypertuned():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = tree.DecisionTreeClassifier().fit(x_train,y_train)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist_full,
                                       n_iter=20)
        random_search.fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        report(random_search.cv_results_)
        scores = cross_val_score(random_search.best_estimator_, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", random_search.best_estimator_.score(x_train, y_train)

        st = time()
        test_score = random_search.best_estimator_.score(x_test,y_test)
        print "Tree info: {} nodes.".format(random_search.best_estimator_.tree_.node_count)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"


def tree_20_default():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = tree.DecisionTreeClassifier().fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Tree info: {} nodes.".format(clf.tree_.node_count)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"














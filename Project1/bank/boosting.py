import csv
import random
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn.model_selection import train_test_split




x = []
y = []
with open('bank-data.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(row[:len(row) - 1])
        y.append(row[len(row) - 1])
f.close()

x_full = []
y_full = []
with open('bank-extra.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        x_full.append(row[:len(row) - 1])
        y_full.append(row[len(row) - 1])
f.close()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x_full, y_full, test_size = 0.3)

base_est = tree.DecisionTreeClassifier(max_depth = 2, min_samples_leaf = 5000)

def boost_12():
    for i in range(1,350,50):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = i).fit(x_train, y_train)
        start = time()
        scores = cross_val_score(clf, x_train, y_train)
        print "AdaBoost with", i, "classifiers took", time() - start, "seconds to train. Mean: ", scores.mean()
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        start = time()
        print "Final score took ", time() - start, "seconds to calculate. Score: " + str(clf.score(x_test, y_test))
        print "------------------------------------------------------"
def boost_20():
    for i in range(1,350,50):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = i).fit(x_train, y_train)
        start = time()
        scores = cross_val_score(clf, x_train, y_train)
        print "AdaBoost with ", i, "classifiers took", time() - start, "seconds to train. Mean: ", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        start = time()
        print "Final score took ", time() - start, "seconds to calculate. Score: " + str(clf.score(x_test, y_test))
        print "------------------------------------------------------"

def boost_12_graph():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = 301).fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"
def boost_20_graph():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = 301).fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"







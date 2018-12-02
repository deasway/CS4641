import csv
import random
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn.model_selection import train_test_split


x = []
y = []
with open("winequality-white.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        line = row[0].split(";")
        x.append(line[:len(line) - 1])
        y.append(line[-1])
f.close()

x_simple = []
y_simple = []
with open("winequality-white-simplified.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        x_simple.append(row[:len(row) - 1])
        y_simple.append(row[-1])
f.close()

base_est = tree.DecisionTreeClassifier(max_depth = 2, min_samples_leaf = 500)

def boost():
    for i in range(1,350,50):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = i).fit(x_train, y_train)
        start = time()
        scores = cross_val_score(clf, x_train, y_train)
        print "AdaBoost with", i, "classifiers took", time() - start, "seconds to train. Mean: ", scores.mean()
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        start = time()
        print "Final score took ", time() - start, "seconds to calculate. Score: " + str(clf.score(x_test, y_test))
        print "------------------------------------------------------"
def boost_simple():
    for i in range(1,350,50):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = i).fit(x_train, y_train)
        start = time()
        scores = cross_val_score(clf, x_train, y_train)
        print "AdaBoost with ", i, "classifiers took", time() - start, "seconds to train. Mean: ", scores.mean()
        start = time()
        print "Final score took ", time() - start, "seconds to calculate. Score: " + str(clf.score(x_test, y_test))
        print "------------------------------------------------------"

def boost_graph():
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
        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"
def boost_simple_graph():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = AdaBoostClassifier(base_estimator = base_est, n_estimators = 301).fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"







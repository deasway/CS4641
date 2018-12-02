from sklearn import svm
import csv
import random
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.model_selection import train_test_split

x = []
y = []
with open("winequality-white.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        line = row[0].split(";")
        line = map(float, line)
        x.append(line[:len(line) - 1])
        y.append(line[-1])
f.close()

x_simple = []
y_simple = []
with open("winequality-white-simplified.csv", "rb") as f:
    reader = csv.reader(f)
    reader.next()
    for row in reader:
        row = map(float, row)
        x_simple.append(row[:len(row) - 1])
        y_simple.append(row[-1])
f.close()

def svm_rbf():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = svm.SVC(kernel = "rbf").fit(x_train, y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)

        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)
        print "--------------------------------------------------------"
def svm_lin():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = svm.LinearSVC().fit(x_train, y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)

        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def svm_simple_rbf():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = svm.SVC(kernel = "rbf").fit(x_train, y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)

        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)
        print "--------------------------------------------------------"
def svm_simple_lin():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = svm.LinearSVC().fit(x_train, y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)

        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)
        print "--------------------------------------------------------"

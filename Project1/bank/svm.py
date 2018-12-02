from sklearn import svm
import csv
import random
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.model_selection import train_test_split


x = []
y = []
with open('bank-data.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        row = map(float, row)
        x.append(row[:len(row) - 1])
        y.append(row[len(row) - 1])
f.close()

x_full = []
y_full = []
with open('bank-extra.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        row = map(float, row)
        x_full.append(row[:len(row) - 1])
        y_full.append(row[len(row) - 1])
f.close()


def svm_12_rbf():
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
def svm_12_lin():
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


def svm_20_rbf():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
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
def svm_20_lin():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
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






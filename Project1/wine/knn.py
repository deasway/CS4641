from sklearn import svm
import csv
import random
from sklearn.model_selection import cross_val_score
from time import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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


def knn_graph():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]
        st = time()
        clf = KNeighborsClassifier(n_neighbors=150).fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()

        print "Training score: ", clf.score(x_train, y_train)
        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"


def knn_simple_graph():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = KNeighborsClassifier(n_neighbors = 150).fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def knn():
    for i in range(1, 250, 25):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

        st = time()
        clf = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
        print "Took", time() - st, "to train knn with",i,"neighbors."
        st = time()
        score = clf.score(x_test, y_test)
        print "Took",time() -st, "to score:",score
def knn_simple():
    for i in range(1, 250, 25):
        x_train, x_test, y_train, y_test = train_test_split(x_simple, y_simple, test_size = 0.3)

        st = time()
        clf = KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
        print "Took", time() - st, "to train knn with",i,"neighbors."
        st = time()
        score = clf.score(x_test, y_test)
        print "Took",time() -st, "to score:",score

import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.model_selection import cross_val_score



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
with open('bank-data-extra.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        row = map(float, row)
        x_full.append(row[:len(row) - 1])
        y_full.append(row[len(row) - 1])
f.close()


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


# specify parameters and distributions to sample from
param_dist = {
              "solver": ["sgd"],
              "activation": ["identity", "logistic", "tanh", "relu"],
              "learning_rate": ["constant", "invscaling", "adaptive"],
              "learning_rate_init": stats.uniform(0.001, 0.1),
              "momentum": stats.uniform(0.1, 0.9)
              }

def net_12_hypertuned():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = MLPClassifier().fit(x_train,y_train)
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
        print "Net info: {} layers.".format(random_search.best_estimator_.n_layers_)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def net_12():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = MLPClassifier().fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Net info: {} layers.".format(clf.n_layers_)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def net_20_hypertuned():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = MLPClassifier().fit(x_train,y_train)
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
        print "Net info: {} layers.".format(random_search.best_estimator_.n_layers_)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def net_20():
    for i in range(1, 11, 1):
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
        split = len(x) / 10 * i
        x_train = x_train[:split]
        y_train = y_train[:split]

        st = time()
        clf = MLPClassifier().fit(x_train,y_train)
        print "Training time with {} samples: {}.".format(split, time() - st)
        scores = cross_val_score(clf, x_train, y_train, cv=5)
        print "Cross Val score across 5 folds:", scores.mean()
        print "Training score: ", clf.score(x_train, y_train)

        st = time()
        test_score = clf.score(x_test,y_test)
        print "Net info: {} layers.".format(clf.n_layers_)
        print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

        print "--------------------------------------------------------"

def layers():
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
    st = time()
    clf = MLPClassifier(hidden_layer_sizes = (100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100)).fit(x_train,y_train)
    print "Training time with {} samples: {}.".format(len(x_full), time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    st = time()
    test_score = clf.score(x_test,y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)


def test_12():
    print "-----------------net_12_hypertuned--------------------------"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
    st = time()
    clf = MLPClassifier().fit(x_train,y_train)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=20)
    random_search.fit(x_train,y_train)
    print "Training time with {} samples: {}.".format(len(x), time() - st)
    report(random_search.cv_results_)
    scores = cross_val_score(random_search.best_estimator_, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", random_search.best_estimator_.score(x_train, y_train)

    st = time()
    test_score = random_search.best_estimator_.score(x_test,y_test)
    print "Net info: {} layers.".format(random_search.best_estimator_.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"
    print "-------------------net_12_default-----------------------"
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

    st = time()
    clf = MLPClassifier().fit(x_train,y_train)
    print "Training time with {} samples: {}.".format(len(x), time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test,y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

def test_20():
    print "--------------------net_20_hypertuned-------------------"
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
    st = time()
    clf = MLPClassifier().fit(x_train,y_train)
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=20)
    random_search.fit(x_train,y_train)
    print "Training time with {} samples: {}.".format(len(x_full), time() - st)
    report(random_search.cv_results_)
    scores = cross_val_score(random_search.best_estimator_, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", random_search.best_estimator_.score(x_train, y_train)

    st = time()
    test_score = random_search.best_estimator_.score(x_test,y_test)
    print "Net info: {} layers.".format(random_search.best_estimator_.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"
    print "------------------net_20_default------------------------"
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)

    st = time()
    clf = MLPClassifier().fit(x_train,y_train)
    print "Training time with {} samples: {}.".format(len(x_full), time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test,y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

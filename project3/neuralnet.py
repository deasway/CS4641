import csv
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy




# specify parameters and distributions to sample from
param_dist = {
              "solver": ["sgd"],
              "activation": ["identity", "logistic", "tanh", "relu"],
              "learning_rate": ["constant", "invscaling", "adaptive"],
              "learning_rate_init": stats.uniform(0.001, 0.1),
              "momentum": stats.uniform(0.1, 0.9)
}

def bank():
    x = []
    y = []
    with open("bank-data.csv", "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            row = map(float, row)
            x.append(row[:len(row) - 1])
            y.append(row[-1])
    f.close()

    # n = Normalizer()
    # x = n.transform(x)
    print "Default------------------------------"
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nPCA------------------"
    pca = IncrementalPCA(3)
    x_pca = pca.fit_transform(x)
    param = x_pca

    x_train, x_test, y_train, y_test = train_test_split(param,y,test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nICA------------------"
    ica = FastICA(3)
    x_ica = ica.fit_transform(x)
    param = x_ica

    x_train, x_test, y_train, y_test = train_test_split(param,y,test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nRandom Projection------------------"
    rp = GaussianRandomProjection(3)
    x_rp = rp.fit_transform(x)
    param = x_rp

    x_train, x_test, y_train, y_test = train_test_split(param,y,test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nK best------------------"
    kb = SelectKBest(k=3)
    kb.fit(x, y)
    x_kb = kb.transform(x)
    param = x_kb

    x_train, x_test, y_train, y_test = train_test_split(param,y,test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nNormalized----------------------------"

    n = Normalizer()
    x = n.transform(x)

    print "\n\nDefault------------------------------"
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)


    print "\n\nPCA------------------"
    pca = IncrementalPCA(3)
    x_pca = pca.fit_transform(x)
    param = x_pca

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nICA------------------"
    ica = FastICA(3)
    x_ica = ica.fit_transform(x)
    param = x_ica

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nRandom Projection------------------"
    rp = GaussianRandomProjection(3)
    x_rp = rp.fit_transform(x)
    param = x_rp

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nK best------------------"
    kb = SelectKBest(k=3)
    kb.fit(x, y)
    x_kb = kb.transform(x)
    param = x_kb

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"


def bank_cluster():
    x = []
    y = []
    with open("bank-data.csv", "rb") as f:
        reader = csv.reader(f)
        for row in reader:
            row = map(float, row)
            x.append(row[:len(row) - 1])
            y.append(row[-1])
    f.close()



    print "K-means-------------------------"

    print "\n\nPCA------------------"
    pca = IncrementalPCA(3)
    x_pca = pca.fit_transform(x)
    param = x_pca

    y_cluster = KMeans(n_clusters=3).fit_predict(x_pca)
    x_pca = x_pca.tolist()
    for i in range(len(x)):
        x_pca[i].append(y_cluster[i])

    param = Normalizer().transform(param)


    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nICA------------------"
    ica = FastICA(3)
    x_ica = ica.fit_transform(x)
    param = x_ica

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nRandom Projection------------------"
    rp = GaussianRandomProjection(3)
    x_rp = rp.fit_transform(x)
    param = x_rp

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nK best------------------"
    kb = SelectKBest(k=3)
    kb.fit(x, y)
    x_kb = kb.transform(x)
    param = x_kb

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "EM-------------------------"
    print "\n\nPCA------------------"
    pca = IncrementalPCA(3)
    x_pca = pca.fit_transform(x)
    param = x_pca

    y_cluster = KMeans(n_clusters=3).fit_predict(x_pca)
    x_pca = x_pca.tolist()
    for i in range(len(x)):
        x_pca[i].append(y_cluster[i])

    param = Normalizer().transform(param)


    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nICA------------------"
    ica = FastICA(3)
    x_ica = ica.fit_transform(x)
    param = x_ica

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nRandom Projection------------------"
    rp = GaussianRandomProjection(3)
    x_rp = rp.fit_transform(x)
    param = x_rp

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"

    print "\n\nK best------------------"
    kb = SelectKBest(k=3)
    kb.fit(x, y)
    x_kb = kb.transform(x)
    param = x_kb

    y_cluster = KMeans(n_clusters=3).fit_predict(param)
    param = param.tolist()
    for i in range(len(x)):
        param[i].append(y_cluster[i])

    param = Normalizer().transform(param)

    x_train, x_test, y_train, y_test = train_test_split(param, y, test_size=0.3)

    st = time()
    clf = MLPClassifier().fit(x_train, y_train)
    print "Training time: {}".format(time() - st)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print "Cross Val score across 5 folds:", scores.mean()
    print "Training score: ", clf.score(x_train, y_train)

    st = time()
    test_score = clf.score(x_test, y_test)
    print "Net info: {} layers.".format(clf.n_layers_)
    print "Testing time: {} || Test Accuracy: {}".format(time() - st, test_score)

    print "--------------------------------------------------------"


bank_cluster()
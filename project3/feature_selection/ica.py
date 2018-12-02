from sklearn.decomposition import FastICA
import csv
import numpy
from scipy.stats import kurtosis
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt





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

    n = Normalizer()
    x = n.transform(x)

    for i in range(len(x[0])):
        print kurtosis(x[:,i])

    comp = 2
    ica = FastICA(comp)
    ica.fit(x)
    x = ica.transform(x)

    print "-------------------------"
    for i in range(comp):
        print kurtosis(x[:,i])

    plt.figure(figsize=(12,12))
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])
    plt.title("Distribution of Banking after ICA")
    plt.show()

def wine():
    x = []
    y = []
    with open("winequality-white.csv", "rb") as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            line = row[0].split(";")
            line = map(float, line)
            x.append(line[:len(line) - 1])
            y.append(int(line[-1]))
    f.close()

    n = Normalizer()
    x = n.transform(x)

    for i in range(len(x[0])):
        print kurtosis(x[:,i])

    comp = 2
    ica = FastICA(comp)
    x = ica.fit_transform(x)
    print "-------------------------"

    for i in range(comp):
        print kurtosis(x[:,i])
    plt.figure(figsize=(12,12))
    plt.title("Distribution of Wine after ICA")
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])

    plt.show()

bank()
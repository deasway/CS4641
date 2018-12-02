from sklearn.random_projection import GaussianRandomProjection
import csv
import numpy
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

    comp = 2

    rp = GaussianRandomProjection(comp)
    x = rp.fit_transform(x)

    plt.figure(figsize=(12,12))
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])
    plt.title("Distribution of Banking after Random Projection")
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

    comp = 2

    rp = GaussianRandomProjection(comp)
    x = rp.fit_transform(x)

    plt.figure(figsize=(12,12))
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])
    plt.title("Distribution of Wine after Random Projection")
    plt.show()



wine()
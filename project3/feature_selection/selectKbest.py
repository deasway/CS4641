from sklearn.feature_selection import SelectKBest
import csv
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


    # n = Normalizer()
    # x = n.transform(x)


    skb = SelectKBest(k=2)
    skb.fit(x, y)
    x = skb.transform(x)
    print skb.scores_

    plt.figure(figsize=(12,12))
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])
    plt.title("Distribution of Banking after Select K-best")
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

    # n = Normalizer()
    # x = n.transform(x)

    skb = SelectKBest(k=2)
    skb.fit(x,y)
    x = skb.transform(x)

    print skb.scores_

    plt.figure(figsize=(12,12))
    plt.scatter(x[:, 0], x[:, 1],c=y, color=[0.5,0.5,0.5])
    plt.title("Distribution of Wine after Select k-best")
    plt.show()


bank()
wine()

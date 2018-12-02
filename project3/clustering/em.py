from sklearn.mixture import GaussianMixture
import csv
from sklearn.preprocessing import Normalizer
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score
import numpy



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


    normal = Normalizer()
    x = normal.transform(x)

    tot = 0
    for rate in y:
        tot += rate
    print "avg score across all wine:", (tot + 0.0) / len(y)
    for i in range(2,10):
        print "----------------clustering for ", i, "components-----------"

        em = GaussianMixture(n_components=i)
        y_cluster = em.fit(x).predict(x)

        clusters=[]
        for j in range(i):
            clusters.append([])
        for j in range(len(y_cluster)):
            clusters[y_cluster[j]].append((y_cluster[j], y[j]))
        for k in range(len(clusters)):

            tot = 0
            for j in range(len(clusters[k])):
                tot += clusters[k][j][1]
            print "mean for cluster", k, (tot + 0.0) / len(clusters[k])
        print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                         "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual","Silhouette")
        print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, y_cluster),
                         homogeneity_score(y,y_cluster), v_measure_score(y, y_cluster), adjusted_rand_score(y, y_cluster),
                         adjusted_mutual_info_score(y, y_cluster),
                         silhouette_score(x,y_cluster,sample_size=6000))

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

    # normal = Normalizer()
    # x = normal.transform(x)
    for i in range(2,10):
        print "----------------clustering for ", i, "components-----------"

        em = GaussianMixture(i)
        y_cluster = em.fit(x).predict(x)

        clusters=[]
        for j in range(i):
            clusters.append([])
        for j in range(len(y_cluster)):
            clusters[y_cluster[j]].append((y_cluster[j], y[j]))

        print str.format("{:<10}{:<10}{:<10}{:<20}","No","Yes","Size","Ratio")
        for j in range(i):
            no_cnt = 0
            yes_cnt = 0
            for k in range(len(clusters[j])):
                if clusters[j][k][1] == 1:
                    yes_cnt += 1
                else:
                    no_cnt += 1
            print str.format("{:<10}{:<10}{:<10}{:<10}",no_cnt, yes_cnt,
                             no_cnt+ yes_cnt, (yes_cnt + 0.0) / (no_cnt + yes_cnt))

        print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                         "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual","Silhouette")
        print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, y_cluster),
                         homogeneity_score(y,y_cluster), v_measure_score(y, y_cluster), adjusted_rand_score(y, y_cluster),
                         adjusted_mutual_info_score(y, y_cluster),
                         silhouette_score(x,y_cluster,sample_size=1000))



wine()
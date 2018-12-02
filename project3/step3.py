from sklearn.mixture import GaussianMixture
import csv
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.feature_selection import SelectKBest
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import silhouette_score

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

    comp = 3

    pca = IncrementalPCA(comp)
    ica = FastICA(comp)
    rp = GaussianRandomProjection(comp)
    kb = SelectKBest(k=comp)

    x_pca = pca.fit_transform(x)
    x_ica = ica.fit_transform(x)
    x_rp = rp.fit_transform(x)
    x_kb = kb.fit_transform(x, y)

    print "Kmeans \n\n"
    kmeans = KMeans(3)
    y_pca = kmeans.fit(x_pca).predict(x_pca)
    param = y_pca
    print("\n\n\n---PCA---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(3)
    y_ica = kmeans.fit(x_ica).predict(x_ica)
    param = y_ica
    print("\n\n\n---ICA---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(3)
    y_rp = kmeans.fit(x_rp).predict(x_rp)
    param = y_rp
    print("\n\n\n---Random Projection---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(3)
    y_kb = kmeans.fit(x_kb).predict(x_kb)
    param = y_kb
    print("\n\n\n---K best---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print "EM \n\n"
    print("\n\n\n---PCA---")
    em = GaussianMixture(3)
    y_pca = em.fit(x_pca).predict(x_pca)
    param = y_pca
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---ICA---")
    em = GaussianMixture(3)
    y_ica = em.fit(x_ica).predict(x_ica)
    param = y_ica
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---Random Projection---")
    em = GaussianMixture(3)
    y_rp = em.fit(x_rp).predict(x_rp)
    param = y_rp
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---K best---")
    em = GaussianMixture(3)
    y_kb = em.fit(x_kb).predict(x_kb)
    param = y_kb
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

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

    comp = 3

    pca = IncrementalPCA(comp)
    ica = FastICA(comp)
    rp = GaussianRandomProjection(comp)
    kb = SelectKBest(k=comp)

    x_pca = pca.fit_transform(x)
    x_ica = ica.fit_transform(x)
    x_rp = rp.fit_transform(x)
    x_kb = kb.fit_transform(x, y)

    print "Kmeans \n\n"
    kmeans = KMeans(7)
    y_pca = kmeans.fit(x_pca).predict(x_pca)
    param = y_pca
    print("\n\n\n---PCA---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(7)
    y_ica = kmeans.fit(x_ica).predict(x_ica)
    param = y_ica
    print("\n\n\n---ICA---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(7)
    y_rp = kmeans.fit(x_rp).predict(x_rp)
    param = y_rp
    print("\n\n\n---Random Projection---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    kmeans = KMeans(7)
    y_kb = kmeans.fit(x_kb).predict(x_kb)
    param = y_kb
    print("\n\n\n---K best---")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", "Inertia",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", kmeans.inertia_, completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print "EM \n\n"
    print("\n\n\n---PCA---")
    em = GaussianMixture(7)
    y_pca = em.fit(x_pca).predict(x_pca)
    param = y_pca
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---ICA---")
    em = GaussianMixture(7)
    y_ica = em.fit(x_ica).predict(x_ica)
    param = y_ica
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---Random Projection---")
    em = GaussianMixture(7)
    y_rp = em.fit(x_rp).predict(x_rp)
    param = y_rp
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))

    print("\n\n\n---K best---")
    em = GaussianMixture(7)
    y_kb = em.fit(x_kb).predict(x_kb)
    param = y_kb
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}",
                     "Completeness", "Homogeneity", "V-measure", "Adj Rand", "Adj Mutual", "Silhouette")
    print str.format("{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}", completeness_score(y, param),
                     homogeneity_score(y, param), v_measure_score(y, param), adjusted_rand_score(param, y),
                     adjusted_mutual_info_score(y, param),
                     silhouette_score(x, param, sample_size=10000))




wine()
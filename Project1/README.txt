Name: Alan Li

To run code, install:
Python 2.7.x
scikit-learn : http://scikit-learn.org/stable/install.html
numpy & scipy : https://www.scipy.org/scipylib/download.html

All the required data files are in their respective folders.
the other .txt files include information about the datasets themselves. They were written by the original authors:

Citations for datasets:
bank: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In press, http://dx.doi.org/10.1016/j.dss.2014.03.001
Available at: [pdf] http://dx.doi.org/10.1016/j.dss.2014.03.001
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2014-dss.txt


wine:  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

To run Decision Tree code: go to the desired dataset and run dtree.py. Then call the function that will run the algorithm you want to see.
bank:
bank-simple with parameter tuning: tree_12_hypertuned()
bank-simple with default params: tree_12_default()
bank with parameter tuning: tree_20_hypertuned()
bank with default params: tree_20_default()
wine:
wine-simple with parameter tuning: tree_simple_hypertuned()
wine-simple with default params: tree_simple_default()
wine with parameter tuning: tree_hypertuned()
wine with default params: tree_default()

To run Neural net code, go to the desired dataset and run neural_net.py. Then call the function you want to see.
bank:
bank-simple with parameter tuning: net_12_hypertuned()
bank-simple with default params: net_12()
bank with parameter tuning: net_20_hypertuned()
bank with default params: net_20()
bank with increased layers: layers()
bank with just the final results: test_12() 
bank-simple with just the final results: test_20() 
wine:
wine-simple with parameter tuning: net_simple_hypertuned()
wine-simple with default params: net_simple()
win with parameter tuning: net_hypertuned()
wine with default params: net()

Boosting, go to the desired folder and run boosting.py.
bank:
bank-simple with iteratively increasing training sizes: boost_12_graph()
bank-simple with iteratively increasing number of learners: boost_12()
bank with iteratively increasing training sizes: boost_20_graph()
bank with iteratively increasing number of learners: boost_20()
wine:
wine-simple with iteratively increasing training sizes: boost_simple_graph()
wine-simple with iteratively increasing number of learners: boost_simple()
wine with iteratively increasing training sizes: boost_graph()
wine with iteratively increasing number of learners: boost()

SVMs, go to the desired folder and run svm.py
bank:
bank-simple with iteratively increasing training sizes and rbf kernel: svm_12_rbf()
bank-simple with increasing training sizes and linear kernel: svm_12_lin()
bank with increasing training sizes and rbf kernel: svm_20_rbf()
bank with increasing training sizes and linear kernel: svm_20_lin()
wine:
wine with increasing training sizes and rbf kernel: svm_rbf()
wine with increasing training sizes and linear kernel: svm_lin()
wine-simple with increasing training sizes and rbf kernel: svm_simple_rbf()
wine-simple with increasing training sizes and linear kernel: svm_simple_lin()

kNN, go to desired folder and run knn.py
bank:
bank-simple with increasing training sizes: knn_12_graph()
bank with increasing training sizes: knn_20_graph()
bank-simple with increasing neighbors: knn_12()
bank with increasing neighbors: knn_20()
wine:
wine with increasing training sizes: knn_graph()
wine-simple with increasing training sizes: knn_simple_graph()
wine with increasing amount of neighbors: knn()
wine-simple with increasing amount of neighbors: knn_simple()







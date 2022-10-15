from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def getKNN(X, Y):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, Y)
    return knn


def getDecisionTree(X, Y):
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree.fit(X, Y)
    return decisionTree


def getSVM(X, Y):
    svmModel = svm.SVC()
    svmModel.fit(X, Y)
    return svmModel

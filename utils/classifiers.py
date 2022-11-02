from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load


def splitDataSet(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, shuffle=True
    )
    return (X_train, X_test, Y_train, Y_test)


def getKNN(X, Y):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, Y)
    return knn


def getDecisionTree(X, Y):
    decisionTree = DecisionTreeClassifier(max_depth=10)
    decisionTree.fit(X, Y)
    return decisionTree


def getSVM(X, Y):
    svmModel = SVC()
    svmModel.fit(X, Y)
    return svmModel


def saveClassifier(classifier, fileName):
    dump(classifier, fileName)


def retrieveClassifier(fileName):
    return load(fileName)

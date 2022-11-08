from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


def splitDataSet(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, shuffle=True
    )
    return (X_train, X_test, Y_train, Y_test)


def getKNN(X, Y):
    knn = retrieveObject("./objects/knn.joblib")
    knn = None

    if knn == None:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, Y)

    saveObject(knn, "./objects/knn.joblib")
    return knn


def getDecisionTree(X, Y):
    decisionTree = retrieveObject("./objects/decisionTree.joblib")
    decisionTree = None

    if decisionTree == None:
        decisionTree = RandomForestClassifier(max_depth=200, random_state=0)
        decisionTree.fit(X, Y)

    saveObject(decisionTree, "./objects/decisionTree.joblib")
    return decisionTree


def getSVM(X, Y):
    svmModel = retrieveObject("./objects/svm.joblib")

    if svmModel == None:
        svmModel = SVC()
        svmModel.fit(X, Y)

    saveObject(svmModel, "./objects/svm.joblib")
    return svmModel


def saveObject(object, fileName):
    dump(object, fileName)


def retrieveObject(fileName):
    object = None

    try:
        object = load(fileName)
    except:
        print("There is no file with the following name: " + fileName)

    return object

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from sklearn.neural_network import MLPClassifier


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
        decisionTree = RandomForestClassifier(max_depth=200, n_estimators=100)
        decisionTree.fit(X, Y)

    saveObject(decisionTree, "./objects/decisionTree.joblib")
    return decisionTree


def getSVM(X, Y, name):
    svmModel = retrieveObject("./objects/" + name + ".joblib")

    if svmModel == None:
        svmModel = BaggingClassifier(SVC(), n_estimators=20)
        svmModel.fit(X, Y)

    saveObject(svmModel, "./objects/" + name + ".joblib")
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


def getCNN():
    cnn = load_model("./cnn/Digits")
    return cnn


def getANN():
    cnn = load_model("./cnn/ANN-Trial")
    return cnn


def decodeResult(results):

    character = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    decoded = 0
    print(results)
    for i in range(len(results)):
        if results[i] > 0.9:
            decoded = character[i]
    return decoded


def getIndex(array):
    index = 0
    for i in range(len(array)):
        if int(array[i]) == 1:
            index = i
            break
    return index


def getMLP(X, Y, name):
    clf = retrieveObject("./objects/" + name + ".joblib")

    if clf == None:
        clf = MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(200,),
            random_state=1,
        )
        clf.fit(X, Y)
        saveObject(clf, "./objects/" + name + ".joblib")

    return clf

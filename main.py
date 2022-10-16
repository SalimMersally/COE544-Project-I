from utils.classifiers import *

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 1, 0, 1]

(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

knn = getKNN(X_train, Y_train)
decisionTree = getDecisionTree(X_train, Y_train)
svmModel = getSVM(X_train, Y_train)

saveClassifier(knn, "./classifiers/knn.joblib")
saveClassifier(decisionTree, "./classifiers/decisionTree.joblib")
saveClassifier(svmModel, "./classifiers/svm.joblib")

Y_predict1 = knn.predict(X_test)
Y_predict2 = decisionTree.predict(X_test)
Y_predict3 = svmModel.predict(X_test)

from utils.classifiers import *

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 1, 0, 1]

knn = getKNN(X, Y)
decisionTree = getDecisionTree(X, Y)
svmModel = getSVM(X, Y)

newX = [[0, 1]]

print(knn.predict(newX))
print(decisionTree.predict(newX))
print(svmModel.predict(newX))

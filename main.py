from configparser import MAX_INTERPOLATION_DEPTH
from utils.classifiers import *

# X = [[0, 0], [1, 1], [0, 1], [1, 0]]
# Y = [0, 1, 0, 1]

# (X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

# knn = getKNN(X_train, Y_train)
# decisionTree = getDecisionTree(X_train, Y_train)
# svmModel = getSVM(X_train, Y_train)

# saveClassifier(knn, "./classifiers/knn.joblib")
# saveClassifier(decisionTree, "./classifiers/decisionTree.joblib")
# saveClassifier(svmModel, "./classifiers/svm.joblib")

# Y_predict1 = knn.predict(X_test)
# Y_predict2 = decisionTree.predict(X_test)
# Y_predict3 = svmModel.predict(X_test)


# import cv2
# import os
# import csv
# import numpy as np

# path = os.path.join(os.getcwd(), "dataSet", "english.csv")

# file = open(path)
# fileCSV = csv.reader(file)

# X = []
# Y = []
# for row in fileCSV:
#     if row[1] == "9":
#         break
#     if row[1] == "label":
#         continue
#     print(row[0])
#     Y.append(row[1])
#     imagePath = os.path.join(os.getcwd(), "dataSet", row[0].replace("/", "\\"))
#     img = cv2.imread(imagePath)
#     list = np.array(img)
#     flat_list = list.flatten()
#     X.append(flat_list)

# print("reading done")
# (X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
# print("split done")
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
# )

# knn = getKNN(X_train, Y_train)
# # print("knn done")
# # decisionTree = getDecisionTree(X_train, Y_train)
# print("tree done")
# # svmModel = getSVM(X_train, Y_train)
# # print("svm done")
# Y_predict1 = knn.predict(X_test)
# print("prediction 1 done")
# # Y_predict2 = decisionTree.predict(X_test)
# # print("prediction 2 done")
# # Y_predict3 = svmModel.predict(X_test)
# # print("prediction 3 done")

# print(accuracy_score(Y_test, Y_predict1))
# # print(accuracy_score(Y_test, Y_predict2))
# # print(accuracy_score(Y_test, Y_predict3))

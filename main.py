from configparser import MAX_INTERPOLATION_DEPTH
from utils.classifiers import *
import os
import csv
import cv2
from utils.preprocess import *
from utils.features import *
from utils.featureChoice import *
from pprint import pprint

X, Y = get_dataSet_or_process_images()

# X, Y = process_images()
# for row in fileCSV:
#     # if row[1] == "9":
#     #     break
#     if row[1] == "label":
#         continue
#     print(row[0])
#     print(row[1])
#     Y.append(row[1])
#     imagePath = os.path.join(os.getcwd(), "dataSet", row[0].replace("/", "\\"))
#     img = cv2.imread(imagePath)
#     chracters, processedImg = find_bounding_box(img)
#     print(processedImg)
#     print(cv2.imwrite("CroppedImages/" + row[0], processedImg))
#     feature1 = average_distance_from_center(processedImg)
#     feature2 = number_of_inner_closed_loops(processedImg)
#     feature3, feature4, feature5, feature6 = calc_perc_whitepx_quadrants(processedImg)
#     features = get_proj_histogram(processedImg)
#     features.append(feature1)
#     features.append(feature2)
#     features.append(feature3)
#     features.append(feature4)
#     features.append(feature5)
#     features.append(feature6)
#     print(features)

#     X.append(features)

# print("reading done")

# print(cor_selector(X, Y, len(X)))

(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
# print("split done")
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
# )

# knn = getKNN(X_train, Y_train)
# print("knn done")
# decisionTree = getDecisionTree(X_train, Y_train)
# print("tree done")
# svmModel = getSVM(X_train, Y_train)
# print("svm done")
# Y_predict1 = knn.predict(X_test)
# print("prediction 1 done")
# Y_predict2 = decisionTree.predict(X_test)
# print("prediction 2 done")
# Y_predict3 = svmModel.predict(X_test)
# print("prediction 3 done")

# print(accuracy_score(Y_test, Y_predict1))
# print(accuracy_score(Y_test, Y_predict2))
# print(accuracy_score(Y_test, Y_predict3))

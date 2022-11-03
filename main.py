from configparser import MAX_INTERPOLATION_DEPTH
from utils.classifiers import *
import os
import csv
import cv2
from utils.preprocess import *
from utils.features import *

path = os.path.join(os.getcwd(), "dataSet", "english.csv")
file = open(path)
fileCSV = csv.reader(file)

X = []
Y = []
for row in fileCSV:
    # if row[1] == "9":
    #     break
    if row[1] == "label":
        continue
    print(row[0])
    print(row[1])
    Y.append(row[1])
    imagePath = os.path.join(os.getcwd(), "dataSet", row[0].replace("/", "\\"))
    img = cv2.imread(imagePath)
    chracters, processedImg = find_bounding_box(img)
    print(cv2.imwrite("CroppedImages/" + row[0], processedImg))
    # feature1 = average_distance_from_center(processedImg)
    # feature2 = number_of_inner_closed_loops(processedImg)
    feature3, feature4, feature5, feature6 = calc_perc_whitepx_quadrants(processedImg)
    features = [feature3, feature4, feature5, feature6]
    print(features)
    # list = np.array(processedImg)
    # flat_list = list.flatten()
    X.append(features)

print("reading done")
(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
print("split done")
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

knn = getKNN(X_train, Y_train)
print("knn done")
decisionTree = getDecisionTree(X_train, Y_train)
print("tree done")
svmModel = getSVM(X_train, Y_train)
print("svm done")
Y_predict1 = knn.predict(X_test)
print("prediction 1 done")
Y_predict2 = decisionTree.predict(X_test)
print("prediction 2 done")
Y_predict3 = svmModel.predict(X_test)
print("prediction 3 done")

print(accuracy_score(Y_test, Y_predict1))
print(accuracy_score(Y_test, Y_predict2))
print(accuracy_score(Y_test, Y_predict3))

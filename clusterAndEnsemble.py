from utils.preprocess import *
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint
from sklearn.metrics import accuracy_score
from utils.features import *


def cluster(X, n_clusters):
    y_pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    return y_pred


def divide(X, Y, Y_cluster, n_clusters):
    X_groups = []
    Y_groups = []
    for i in range(n_clusters):
        X_groups.append([])
        Y_groups.append([])
    for i in range(len(X)):
        cluster = Y_cluster[i]
        X_groups[cluster].append(X[i])
        Y_groups[cluster].append(Y[i])

    return X_groups, Y_groups


def cluster_and_get_SVM(X, Y, X_feature):
    Y_cluster = cluster(X_feature, 2)
    X_groups, Y_groups = divide(X, Y, Y_cluster, 2)

    svmModel = get_SVM(X_feature, Y_cluster)

    return X_groups, Y_groups, svmModel


def get_SVM(X, Y):
    (X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
    svmModel = getSVM(X_train, Y_train)
    Y_predict = svmModel.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_predict)
    print("accuracy of SVM: " + str(accuracy))
    return svmModel


X, Y = get_Clustered_No_Feature()
(X, X_test_final, Y, Y_test_final) = splitDataSet(X, Y)
X_feature_1 = []
X_feature_2 = [[], []]
X_feature_3 = [[], [], [], []]
X_feature_4 = [[], [], [], [], [], [], [], []]


for x in X:
    X_feature_1.append(get_proj_histogram_horz(x))

print("1")
X_groups_1, Y_groups_1, svmModel_1 = cluster_and_get_SVM(X, Y, X_feature_1)

svmModels_2 = []

for x in X_groups_1[0]:
    X_feature_2[0].append(get_proj_histogram_vert(x))
for x in X_groups_1[1]:
    X_feature_2[1].append(get_proj_histogram_vert(x))

print("2-1")
X_groups_2_1, Y_groups_2_1, svmModels_2_1 = cluster_and_get_SVM(
    X_groups_1[0], Y_groups_1[0], X_feature_2[0]
)
print("2-2")
X_groups_2_2, Y_groups_2_2, svmModels_2_2 = cluster_and_get_SVM(
    X_groups_1[0], Y_groups_1[0], X_feature_2[0]
)

svmModels_2.append(svmModels_2_1)
svmModels_2.append(svmModels_2_2)

svmModels_3 = [[], []]

for x in X_groups_2_1[0]:
    X_feature_3[0].append(get_dense_SIFT(x))
for x in X_groups_2_1[1]:
    X_feature_3[1].append(get_dense_SIFT(x))
for x in X_groups_2_2[0]:
    X_feature_3[2].append(get_dense_SIFT(x))
for x in X_groups_2_2[1]:
    X_feature_3[3].append(get_dense_SIFT(x))


print("3-1")
X_groups_3_1, Y_groups_3_1, svmModel_3_1 = cluster_and_get_SVM(
    X_groups_2_1[0], Y_groups_2_1[0], X_feature_3[0]
)
print("3-2")
X_groups_3_2, Y_groups_3_2, svmModel_3_2 = cluster_and_get_SVM(
    X_groups_2_1[1], Y_groups_2_1[1], X_feature_3[1]
)
print("3-3")
X_groups_3_3, Y_groups_3_3, svmModel_3_3 = cluster_and_get_SVM(
    X_groups_2_2[0], Y_groups_2_2[0], X_feature_3[2]
)
print("3-4")
X_groups_3_4, Y_groups_3_4, svmModel_3_4 = cluster_and_get_SVM(
    X_groups_2_2[1], Y_groups_2_2[1], X_feature_3[3]
)

svmModels_3[0].append(svmModel_3_1)
svmModels_3[0].append(svmModel_3_2)
svmModels_3[1].append(svmModel_3_3)
svmModels_3[1].append(svmModel_3_4)

for x in X_groups_3_1[0]:
    X_feature_4[0].append(get_HOG(x))
for x in X_groups_3_1[1]:
    X_feature_4[1].append(get_HOG(x))
for x in X_groups_3_2[0]:
    X_feature_4[2].append(get_HOG(x))
for x in X_groups_3_2[1]:
    X_feature_4[3].append(get_HOG(x))
for x in X_groups_3_3[0]:
    X_feature_4[4].append(get_HOG(x))
for x in X_groups_3_3[1]:
    X_feature_4[5].append(get_HOG(x))
for x in X_groups_3_4[0]:
    X_feature_4[6].append(get_HOG(x))
for x in X_groups_3_4[1]:
    X_feature_4[7].append(get_HOG(x))

svmModels_4 = [[[], []], [[], []]]


print("4-1")
svmModels_4[0][0].append(get_SVM(X_feature_4[0], Y_groups_3_1[0]))
print("4-2")
svmModels_4[0][0].append(get_SVM(X_feature_4[1], Y_groups_3_1[1]))
print("4-3")
svmModels_4[0][1].append(get_SVM(X_feature_4[2], Y_groups_3_2[0]))
print("4-4")
svmModels_4[0][1].append(get_SVM(X_feature_4[3], Y_groups_3_2[1]))
print("4-5")
svmModels_4[1][0].append(get_SVM(X_feature_4[4], Y_groups_3_3[0]))
print("4-6")
svmModels_4[1][0].append(get_SVM(X_feature_4[5], Y_groups_3_3[1]))
print("4-7")
svmModels_4[1][1].append(get_SVM(X_feature_4[6], Y_groups_3_4[0]))
print("4-8")
svmModels_4[1][1].append(get_SVM(X_feature_4[7], Y_groups_3_4[1]))

Y_predict_final = []

for x_test in X_test_final:
    x_proj_horz = get_proj_histogram_horz(x_test)
    x_proj_vert = get_proj_histogram_vert(x_test)
    x_hog = get_HOG(x_test)
    x_sift = get_dense_SIFT(x_test)

    y_predict_1 = svmModel_1.predict([x_proj_horz])[0]

    svm_2 = svmModels_2[y_predict_1]
    y_predict_2 = svm_2.predict([x_proj_vert])[0]

    svm_3 = svmModels_3[y_predict_1][y_predict_2]
    y_predict_3 = svm_3.predict([x_sift])[0]

    svm_4 = svmModels_4[y_predict_1][y_predict_2][y_predict_3]
    y_predict_4 = svm_4.predict([x_hog])[0]

    Y_predict_final.append(y_predict_4)

accuracy = accuracy_score(Y_test_final, Y_predict_final)
print("accuracy of: " + str(accuracy))

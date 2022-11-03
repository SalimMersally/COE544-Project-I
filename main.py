from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.featureChoice import *
from pprint import pprint
from sklearn.metrics import accuracy_score


X, Y = get_dataSet_or_process_images()

# for image in X:
#     feature1 = average_distance_from_center(image)
#     feature2 = number_of_inner_closed_loops(image)
#     feature3, feature4, feature5, feature6 = calc_perc_whitepx_quadrants(image)
#     features = get_proj_histogram(image)
#     features.append(feature1)
#     features.append(feature2)
#     features.append(feature3)
#     features.append(feature4)
#     features.append(feature5)
#     features.append(feature6)
#     print(features)


for i in range(len(X)):
    print(i)
    X[i] = np.reshape(X[i], (1024))

(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)

print("data ready")

knn = getKNN(X_train, Y_train)
print("done knn")
decisionTree = getDecisionTree(X_train, Y_train)
print("done tree")
svmModel = getSVM(X_train, Y_train)
print("done svm")
Y_predict1 = knn.predict(X_test)
print("done predict 1")
Y_predict2 = decisionTree.predict(X_test)
print("done predict 2")
Y_predict3 = svmModel.predict(X_test)
print("done predict 3")

print(accuracy_score(Y_test, Y_predict1))
print(accuracy_score(Y_test, Y_predict2))
print(accuracy_score(Y_test, Y_predict3))

img1 = cv2.imread("./testImage1.png")
dummy, x1 = find_bounding_box(img1)
cv2.imwrite("./TestImage1Processed.png", x1)
x1 = np.reshape(x1, (1024))

img2 = cv2.imread("./testImage2.png")
dummy, x2 = find_bounding_box(img2)
cv2.imwrite("./TestImage2Processed.png", x2)
x2 = np.reshape(x2, (1024))

img3 = cv2.imread("./testImage3.png")
dummy, x3 = find_bounding_box(img3)
cv2.imwrite("./TestImage3Processed.png", x3)
x3 = np.reshape(x3, (1024))

print(knn.predict([x1, x2, x3]))
print(decisionTree.predict([x1, x2, x3]))
print(svmModel.predict([x1, x2, x3]))

from utils.classifiers import *
from utils.preprocess import *
from utils.features import *
from utils.featureChoice import *
from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


X, Y = get_dataSet_or_process_images()
X_features = []

#for image in X:
#     feature1 = average_distance_from_center(image)
#     feature2 = number_of_inner_closed_loops(image)
#     feature3, feature4, feature5, feature6 = calc_perc_whitepx_quadrants(image)
# path = os.path.join(os.getcwd(), "dataSet","Img","img002-010.png")
# feature7 =  hough_lines(path)  
# print("feature = " + str(feature7))
#     features = get_proj_histogram(image)
#     features.append(feature1)
#     features.append(feature2)
#     features.append(feature3)
#     features.append(feature4)
#     features.append(feature5)
#     features.append(feature6)
#     X_features.append(features)


# importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt


image = imread("./img001-002.png")
imageResized = resize(image, (32, 32))
fd, hogRes = hog(
    imageResized,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    multichannel=True,
)

print(np.shape(fd))
print(np.shape(hogRes))

for i in range(len(X)):
    # print(np.shape(X[i]))

    fd = hog(
        X[i],
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        multichannel=False,
    )

    X[i] = np.reshape(fd, (288))


# selectedX = SelectKBest(f_classif, k=250).fit_transform(X, Y)

(X_train, X_test, Y_train, Y_test) = splitDataSet(X, Y)
svmModel = getSVM(X_train, Y_train)
Y_predict = svmModel.predict(X_test)
accuracy = accuracy_score(Y_test, Y_predict)


img1 = cv2.imread("./testImage1.png")
dummy, x1 = find_bounding_box(img1)
fd1 = hog(
    x1,
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    multichannel=False,
)

img2 = cv2.imread("./testImage2.png")
dummy, x2 = find_bounding_box(img2)
cv2.imwrite("./TestImage2Processed.png", x2)
fd2 = hog(
    x2,
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    multichannel=False,
)

img3 = cv2.imread("./testImage3.png")
dummy, x3 = find_bounding_box(img3)
cv2.imwrite("./TestImage3Processed.png", x3)
fd3 = hog(
    x3,
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    multichannel=False,
)

# print(knn.predict([fd1, fd2, fd3]))
# print(decisionTree.predict([fd1, fd2, fd3]))
print(svmModel.predict([fd1, fd2, fd3]))

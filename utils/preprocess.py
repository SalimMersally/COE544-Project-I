import imutils
from imutils.contours import sort_contours
import numpy as np
import cv2
import os
import csv
from utils.classifiers import *
from utils.features import *

##the function bellow aims to find and sort contours input images##


def find_contours(img):

    # preprocess the image by blurring to remove noise and converting to grey scale
    to_grey_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    to_blurred = cv2.GaussianBlur(to_grey_scale, (5, 5), 0)

    # detect the edges
    # compute the lower and upper threshold for canny method
    sigma = 0.23  # default value
    median_pix = np.median(img)
    lower = int(max(0, (1 - sigma) * median_pix))
    upper = int(min(255, (1 + sigma) * median_pix))
    # get the edged image
    to_edged = cv2.Canny(to_blurred, lower, upper)

    # find and sort contours from left to right
    contours = cv2.findContours(
        to_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    return contours


##The function below finds bounding boxes of an input image


def find_bounding_box(img):

    characters = []

    contours = find_contours(img)
    paddedArray = []
    for contour in contours:
        # get the bounding box for each contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # extract the character and threshold it

        to_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped = to_gray[y : y + h, x : x + w]
        thresholded = cv2.threshold(
            cropped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        (H, W) = thresholded.shape

        # make 32x32 pixels
        # output = cv2.resize(thresholded, (32,32))
        if W > H:
            thresholded = imutils.resize(thresholded, width=28)
        # otherwise, resize along the height
        else:
            thresholded = imutils.resize(thresholded, height=28)

        # pad instead of extending
        (H, W) = thresholded.shape
        X_padding = int(
            max(0, 28 - W) / 2.0
        )  # if W =32 we won't need to pad on the X-axis
        Y_padding = int(
            max(0, 28 - H) / 2.0
        )  # if H = 32 we won't need to padd on the y-axis

        # pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(
            thresholded,
            top=Y_padding,
            bottom=Y_padding,
            left=X_padding,
            right=X_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        padded = cv2.resize(padded, (28, 28))
        paddedArray.append(padded)
        # add character image and dimension from original image into the characters array
        characters.append((padded, (x, y, w, h)))

    return characters, paddedArray


# the method bellow creates a visual representation of the bounding box on original images
def draw_bounding_boxes(img):

    characters, img = find_bounding_box(img)

    boxes = [b[1] for b in characters]
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(img)
    cv2.waitKey(0)


# this method process all images from dataSet folder
# it also saves X and Y
def process_images():
    X = []
    Y = []
    path = os.path.join(os.getcwd(), "dataSet", "cluster.csv")
    file = open(path)
    fileCSV = csv.reader(file)
    for row in fileCSV:
        # if row[1] =='a':
        #     break
        if row[1] == "label":
            continue
        print(row[0])
        imagePath = os.path.join(os.getcwd(), "dataSet/ImgClustered", row[0])
        img = cv2.imread(imagePath)
        characters, processedImg = find_bounding_box(img)
        processedImg = processedImg[0]  # trained images have only one letter

        sift = get_dense_SIFT(processedImg)
        hog = get_HOG(processedImg)

        X.append(np.concatenate((sift, hog), axis=0))
        Y.append(row[1])

    saveObject(X, "./objects/X.joblib")
    saveObject(Y, "./objects/Y.joblib")
    return X, Y


def get_dataSet_or_process_images():
    X = retrieveObject("./objects/X.joblib")
    Y = retrieveObject("./objects/Y.joblib")

    if X == None or Y == None:
        X, Y = process_images()

    return X, Y


def get_dataSet():
    X = []
    Y = []
    path = os.path.join(os.getcwd(), "dataSet", "english.csv")
    file = open(path)
    fileCSV = csv.reader(file)
    for row in fileCSV:
        # if row[1] =='A':
        #    break
        if row[1] == "label":
            continue
        print(row[0])
        imagePath = os.path.join(os.getcwd(), "dataSet", row[0].replace("/", "\\"))
        img = cv2.imread(imagePath)

        X.append(img)
        Y.append(row[1])

    return X, Y


def get_Clustered_No_Feature():
    X = retrieveObject("./objects/X_noFeature.joblib")
    Y = retrieveObject("./objects/Y_noFeature.joblib")

    if X == None or Y == None:
        X = []
        Y = []
        # path = os.path.join(os.getcwd(), "dataSet", "cluster.csv")
        path = os.path.join(os.getcwd(), "dataSet", "english.csv")
        file = open(path)
        fileCSV = csv.reader(file)
        for row in fileCSV:
            if row[1] == "label":
                continue
            print(row[0])
            # imagePath = os.path.join(os.getcwd(), "dataSet/ImgClustered", row[0])
            imagePath = os.path.join(os.getcwd(), "dataSet", row[0].replace("/", "\\"))
            img = cv2.imread(imagePath)
            characters, processedImg = find_bounding_box(img)
            processedImg = processedImg[0]  # trained images have only one letter

            X.append(processedImg)
            Y.append(row[1])

    saveObject(X, "./objects/X_noFeature.joblib")
    saveObject(Y, "./objects/Y_noFeature.joblib")

    return X, Y


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


def get_character_vs_number(Y):
    Y_differ = []
    for y in Y:
        index = 0
        for i in range(len(character)):
            if y == character[i]:
                index = i
                break
        if index <= 9:
            Y_differ.append(0)
        else:
            Y_differ.append(1)
    return Y_differ

import numpy as np
from sklearn.cluster import KMeans
from utils.preprocess import *
import os
import cv2
from os import listdir
import csv


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


def cluster_dataSet():
    X, Y = get_dataSet()
    number_clusters = 2

    targetdir = "./dataSet/ImgClustered"
    try:
        os.mkdir(targetdir)
    except:
        print("Directory already exists")

    for k in range(len(character)):
        X_letter = []
        X_flat = []
        for i in range(len(X)):
            if Y[i] == character[k]:
                characters, processedImg = find_bounding_box(X[i])
                processedImg = processedImg[0]  # trained images have only one letter
                X_letter.append(X[i])
                X_flat.append(np.reshape(processedImg, (1024)))
        # Clustering
        kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit_predict(X_flat)

        dir = ""
        if character[k].isupper():
            dir = "/" + character[k] + "-U-"
        else:
            dir = "/" + character[k] + "-"

        count0 = 0
        count1 = 0
        for x in kmeans:
            if x == 0:
                count0 += 1
            else:
                count1 += 1

        for i in range(len(kmeans)):
            if (kmeans[i] == 0 and count0 > count1) or (
                kmeans[i] == 1 and count1 > count0
            ):
                cv2.imwrite(targetdir + dir + str(i) + ".png", X_letter[i])
        print(character[k] + " done")


def get_excel():
    f = open("dataSet/cluster.csv", "w", newline="")
    writer = csv.writer(f)

    writer.writerow(["image", "label"])

    folder_dir = "./dataSet/ImgClustered"
    for image in os.listdir(folder_dir):
        row = [image, image[0]]
        writer.writerow(row)

    f.close()

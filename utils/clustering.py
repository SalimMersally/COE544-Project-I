import numpy as np
from sklearn.cluster import KMeans
from utils.preprocess import *
import os
import cv2
from os import listdir
import csv
from pprint import pprint
import os


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

paths = [
    "img001-003.png",
    "img002-043.png",
    "img003-043.png",
    "img004-050.png",
    "img005-043.png",
    "img006-012.png",
    "img007-014.png",
    "img008-012.png",
    "img009-012.png",
    "img010-040.png",
    "img011-012.png",
    "img012-030.png",
    "img013-009.png",
    "img014-012.png",
    "img015-002.png",
    "img016-024.png",
    "img017-030.png",
    "img018-028.png",
    "img019-023.png",
    "img020-012.png",
    "img021-025.png",
    "img022-047.png",
    "img023-018.png",
    "img024-003.png",
    "img025-025.png",
    "img026-025.png",
    "img027-001.png",
    "img028-023.png",
    "img029-001.png",
    "img030-020.png",
    "img031-024.png",
    "img032-001.png",
    "img033-003.png",
    "img034-023.png",
    "img035-020.png",
    "img036-014.png",
    "img037-022.png",
    "img038-012.png",
    "img039-003.png",
    "img040-020.png",
    "img041-003.png",
    "img042-020.png",
    "img043-025.png",
    "img044-014.png",
    "img045-037.png",
    "img046-050.png",
    "img047-032.png",
    "img048-040.png",
    "img049-003.png",
    "img050-003.png",
    "img051-025.png",
    "img052-020.png",
    "img053-047.png",
    "img054-022.png",
    "img055-050.png",
    "img056-003.png",
    "img057-018.png",
    "img058-018.png",
    "img059-026.png",
    "img060-028.png",
    "img061-019.png",
    "img062-050.png",
]


def cluster_dataSet():
    centroid = []
    for path in paths:
        imagePath = os.path.join(os.getcwd(), "centroid", path)
        img = cv2.imread(imagePath)
        characters, processedImg = find_bounding_box(img)
        if len(processedImg) == 1:
            processedImg = processedImg[0]
        else:
            processedImg = processedImg[1]
        centroid.append(np.reshape(processedImg, (784)))

    X, Y = get_dataSet()

    X_flat = []
    X_letter = []
    for i in range(len(X)):
        characters, processedImg = find_bounding_box(X[i])
        X_letter.append(X[i])
        if len(processedImg) == 1:
            processedImg = processedImg[0]
        else:
            processedImg = processedImg[1]
        X_flat.append(np.reshape(processedImg, (784)))
    print("feature done")
    kmeans = KMeans(
        init=centroid, random_state=0, n_clusters=62, max_iter=1
    ).fit_predict(X_flat)
    print("kmeans done")
    targetdir = "./dataSet/ImgClustered"
    try:
        os.mkdir(targetdir)
    except:
        print("Directory already exists")

    for k in range(len(character)):
        dir = ""
        if character[k].isupper():
            dir = "/" + character[k] + "-U-"
        else:
            dir = "/" + character[k] + "-"

        count = 0
        for i in range(len(Y)):
            if str(Y[i]) == character[k]:
                cluster = kmeans[i]
                if cluster == k:
                    cv2.imwrite(targetdir + dir + str(i) + ".png", X_letter[i])
                    count += 1
        print(character[k] + ": " + str(count))


def get_excel():
    f = open("dataSet/cluster.csv", "w", newline="")
    writer = csv.writer(f)

    writer.writerow(["image", "label"])

    folder_dir = "./dataSet/ImgClustered"
    for image in os.listdir(folder_dir):
        row = [image, image[0]]
        writer.writerow(row)

    f.close()

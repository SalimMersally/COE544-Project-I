from keras.preprocessing import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.utils as image
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image

from utils.preprocess import *
image.LOAD_TRUNCATED_IMAGES = True 
model = VGG16(weights='imagenet', include_top=False)

X,Y =  get_dataSet_or_process_images()
number_clusters = 2



character = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P", "Q", "R","S","T","U","V","W","X","Y","Z",
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
]
for k in range(len(character)):
    X_letter = []
    X_flat = []
    for i in range(len(X)):
        if Y[i] == character[k]:
            X_letter.append(X[i])
            X_flat.append(np.reshape(X[i], (1024)))
# Clustering
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit_predict(X_flat)


# Copy with cluster name
    print("\n")
    targetdir1 = "./dataSet/ImgClustered1"
    targetdir2 = "./dataSet/ImgClustered2"
    dir=""
    if character[k].isupper():
        dir = "/"+ character[k]+ "-U-"
    else:
        dir = "/"+ character[k]+ "-"


    for i in range(len(kmeans)):
        if kmeans[i] == 0:
            cv2.imwrite( targetdir1 + dir + str(i) + ".png", X_letter[i])
        else:
            cv2.imwrite( targetdir2 + dir + str(i) + ".png", X_letter[i])
    print(character[k] + " done")

print("Done :)")



##############    
# Variables
#imdir = 'C:/Users/ahmad/OneDrive/Desktop/IEA Project 1/COE544-Project-I/dataSet/Img'
# Loop over files and get features
# filelist = glob.glob(os.path.join(imdir, 'img011*.png'))
# filelist.sort()
# featurelist = []
# for i, imagepath in enumerate(filelist):
#     print("    Status: %s / %s" %(i, len(filelist)), end="\r")
#     img = image.load_img(imagepath, target_size=(224, 224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data, axis=0)
#     img_data = preprocess_input(img_data)
#     features = np.array(model.predict(img_data))
#     featurelist.append(features.flatten())
# Copy images renamed by cluster 
# Check if target dir exists
# try:
#     os.makedirs(targetdir)
# except OSError:
#     pass
#create 3 arrays
#########
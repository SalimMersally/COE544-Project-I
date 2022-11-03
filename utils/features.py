import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
import cv2
from matplotlib import pyplot as plt


# in these methods all images are already processes


def average_distance_from_center(img):
    dist = []
    for x in range(0, 32):
        for y in range(0, 32):
            if img[x][y] == 255:
                temp = ((x - 16) ** 2 + (y - 16) ** 2) ** 0.5
                dist.append(temp)
    # print(dist)
    return sum(dist) / len(dist)


def number_of_inner_closed_loops(img):

    # get contours
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = contours[1] if len(contours) == 2 else contours[2]
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get the actual inner list of hierarchy descriptions
    hierarchy = hierarchy[0]

    # count inner contours
    count = 0
    for component in zip(contours, hierarchy):
        cntr = component[0]
        hier = component[1]
        # discard outermost no parent contours and keep innermost no child contours
        # hier = indices for next, previous, child, parent
        # no parent or no child indicated by negative values
        if (hier[3] > -1) & (hier[2] < 0):
            count = count + 1
    return count


def get_proj_histogram(img):
    img = img / 255
    to_one_dimension = []
    for i in range(32):
        su = 0
        for j in range(32):
            val = img[j][i]
            if val < 0.5:
                val = 0
            else:
                val = 1
            su += val
        to_one_dimension.append(su)
    return to_one_dimension


def calc_perc_whitepx_quadrants(img):
    perc_white_upper_left = (
        np.sum(sum(img[x][y] == 255 for x in range(0, 16) for y in range(0, 16)))
        / 256.0
        * 100
    )
    perc_white_upper_right = (
        np.sum(sum(img[x][y] == 255 for x in range(0, 16) for y in range(16, 32)))
        / 256.0
        * 100
    )
    perc_white_lower_left = (
        np.sum(sum(img[x][y] == 255 for x in range(16, 32) for y in range(0, 16)))
        / 256.0
        * 100
    )
    perc_white_lower_right = (
        np.sum(sum(img[x][y] == 255 for x in range(16, 32) for y in range(16, 32)))
        / 256.0
        * 100
    )

    return (
        perc_white_lower_left,
        perc_white_lower_right,
        perc_white_upper_left,
        perc_white_upper_right,
    )


def hough_lines(path):
    
    img = cv2.imread(path,0)
    img = ~img
    
    
    # Set a precision of 1 degree. (Divide into 180 data points)
    # You can increase the number of points if needed. 
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)

    # Perform Hough Transformation to change x, y, to h, theta, dist space.
    hspace, theta, dist = hough_line(img, tested_angles)
    
    #Now, to find the location of peaks in the hough space we can use hough_line_peaks
    h, q, d = hough_line_peaks(hspace, theta, dist)
    
    return len(h)
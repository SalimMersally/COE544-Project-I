import numpy as np
import cv2
import matplotlib.pyplot as plt


#in these methods all images are already processes

def average_distance_from_center(img):
    dist = []
    for x in range(0,32):
        for y in range (0,32):
            if img[x][y]==255:
                temp = ((x-16)**2 + (y-16)**2)**0.5
                dist.append(temp)
    print(dist)
    return sum(dist)/len(dist)


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
  width =32
  height =32
  
  # Sum the value lines 
  vertical_px = np.sum(img, axis=0)
  # Normalize
  normalize = vertical_px/255
  # create a black image with zeros 
  blankImage = np.zeros_like(img)
  # Make the vertical projection histogram
  for idx, value in enumerate(normalize):
     cv2.line(blankImage, (idx, 0), (idx, height-int(value)), (255,255,255), 1)
  # Concatenate the image
  img_concate = cv2.vconcat(
    [cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGB)])
  plt.imshow(img_concate)
  plt.show()
  
  
  to_one_dimension=[]
  for i in range(32):
    su = 0;
    for j in range(32):
        su += img_concate[j][i]
    to_one_dimension.append(su, end = " ")    
  
  return to_one_dimension

def calc_perc_whitepx_quadrants(img):
    perc_white_upper_left = np.sum(sum (img[x][y] ==255 for x in range(0,16) for y in range(0,16)))/256.0*100
    perc_white_upper_right = np.sum(sum (img[x][y] ==255 for x in range(0,16) for y in range(16,32)))/256.0*100
    perc_white_lower_left = np.sum(sum (img[x][y] ==255 for x in range(16,32) for y in range(0,16)))/256.0*100
    perc_white_lower_right = np.sum(sum (img[x][y] ==255 for x in range(16,32) for y in range(16,32)))/256.0*100
    
    return perc_white_lower_left,perc_white_lower_right,perc_white_upper_left,perc_white_upper_right
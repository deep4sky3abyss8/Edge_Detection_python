#-------------library box-----------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data,filters

#--------img input box--------------------------------------
img = cv2.imread('F:/univercity/projects/dmath/hh/hh.JPG')
height , width , channels = img.shape

#--------gray color filter box------------------------------
grayimg=np.zeros((height,width),dtype=np.uint8)

for i in range(height):
    for j in range(width):
        b,g,r = img[i,j]
        graying=int(0.2989*r + 0.5870*g + 0.1140*b)
        grayimg[i,j]=graying

#-------------edge finding box------------------------------
#edges = filters.sobel(img)
#plt.imshow(edges, cmap='gray')

#----------output box---------------------------------------
cv2.imshow('grayImage',grayimg)
cv2.waitKey(0)
cv2.destroyAllWindows

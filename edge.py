#-------------library box-----------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.filters import gaussian

#--------img input box--------------------------------------
img = cv2.imread('F:/univercity/projects/dmath/hh/hh.JPG')
height , width , channels = img.shape

#--------gray color filter box------------------------------
grayimg=np.zeros((height,width),dtype=np.uint8)    # that's a new img in size of our original img 

for i in range(height):
    for j in range(width):
        b,g,r = img[i,j]
        grayval=int(0.2989*r + 0.5870*g + 0.1140*b)
        grayimg[i,j]=grayval   # already we can apply this all on original img but we do it to check changes by steps

#---------------gowse filter to reduce noise box------------
guimg= gaussian(grayimg,sigma=0.5)

#---------------sobel filter box----------------------------
#sobelx=cv2.Sobel()

#-------------edge finding box------------------------------
#edges = filters.sobel(img)
#plt.imshow(edges, cmap='gray')

#----------output box---------------------------------------
#cv2.imshow('orginal',img)
#cv2.imshow('grayImage',grayimg)
cv2.imshow('bullued',guimg)
cv2.waitKey(0)
cv2.destroyAllWindows
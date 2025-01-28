#-------------library box-----------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import io
from skimage.filters import gaussian


#----------------------function box----------------------------

def gradian(i,j,matrix):                                             # function for sobel filter
    Gx = 2*(matrix[i,j+1]-matrix[i,j-1]) + (matrix[i-1,j+1]+matrix[i+1,j+1]-matrix[i-1,j-1]-matrix[i+1,j-1])
    Gy = 2*(matrix[i+1,j]-matrix[i-1,j]) + (matrix[i+1,j-1]+matrix[i+1,j+1]-matrix[i-1,j-1]-matrix[i-1,j+1])
    return int(((Gx)**2 +(Gy)**2 )**0.5)
#                                                here we calculate gradiant of 3x3 matrix which central object [i,j]

#--------img input box--------------------------------------

img = cv2.imread('F:/univercity/projects/dmath/hh/hh.JPG')
height , width , channels = img.shape                              # for loops need this line


#--------gray color filter box------------------------------

grayimg=np.zeros((height,width),dtype=np.uint8)                   # that's a new img in size of our original img 

for i in range(height):
    for j in range(width):
        b,g,r = img[i,j]
        grayval=int(0.2989*r + 0.5870*g + 0.1140*b)
        grayimg[i,j]=grayval                                      # already we can apply this all on original img but we do it to check changes by steps


#---------------gowse filter to reduce noise box------------

guimg= gaussian(grayimg,sigma=5)

#---------------sobel filter box----------------------------

sobimg=np.zeros((height,width),dtype=np.uint8)

#+++++++++++++++++++++++++++++
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#+++++++++++++++++++++++++++++

for i in range(1,height-1):
    for j in range(1,width-1):
        #sobimg[i,j] = gradian(i,j,grayimg)   # we can alse use this way but its dont work good
        
        #+++++++++++++++++++++++++++++++++
        region = grayimg[i-1:i+2, j-1:j+2]
        
        gx = np.sum(sobel_x * region)  
        gy = np.sum(sobel_y * region) 
        
        sobimg[i, j] = (gx**2 + gy**2)**0.5
        #+++++++++++++++++++++++++++++++++


#-------------edge finding box------------------------------
#edges = filters.sobel(img)
#plt.imshow(edges, cmap='gray')

#----------output box---------------------------------------
cv2.imshow('orginal',img)
cv2.imshow('grayImage',grayimg)
cv2.imshow('bullued',guimg)
cv2.imshow('sobel filtered',sobimg)
cv2.waitKey(0)
cv2.destroyAllWindows
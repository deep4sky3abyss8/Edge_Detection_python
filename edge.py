#-------------library box-----------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data,filters

#--------img input box--------------------------------------
img = cv2.imread('F:/univercity/projects/dmath/hh/hh.JPG')
height , width , channels = img.shape

#--------gray color filter box------------------------------
edges = filters.sobel(img)  
plt.imshow(edges, cmap='gray')





#----------output box------------------
cv2.imshow('Image',img)


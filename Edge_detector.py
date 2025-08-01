#----------------------library box---------------------------------

import cv2
import numpy as np

from skimage import io
from skimage.filters import gaussian


#----------------------function box--------------------------------

#def gradian(i,j,matrix):                                             # function for sobel filter write by myself
    #Gx = 2*(matrix[i,j+1]-matrix[i,j-1]) + (matrix[i-1,j+1]+matrix[i+1,j+1]-matrix[i-1,j-1]-matrix[i+1,j-1])
    #Gy = 2*(matrix[i+1,j]-matrix[i-1,j]) + (matrix[i+1,j-1]+matrix[i+1,j+1]-matrix[i-1,j-1]-matrix[i-1,j+1])
    #return int(((Gx)**2 +(Gy)**2 )**0.5)
#                                                here we calculate gradiant of 3x3 matrix which central object [i,j]

#------------------------img input box------------------------------

address=input()
#img = cv2.imread('F:/univercity/projects/dmath/hh/hh.JPG')
img = cv2.imread(address)

height , width , channels = img.shape                              # for loops need this line


#----------------------gray color filter box-------------------------

grayimg=np.zeros((height,width),dtype=np.uint8)                   # that's a new img in size of our original img 

for i in range(height):
    for j in range(width):
        b,g,r = img[i,j]
        grayval=int(0.2989*r + 0.5870*g + 0.1140*b)
        grayimg[i,j]=grayval                                      # already we can apply this all on original img but we do it to check changes by steps


#---------------------gowse filter to reduce noise box------------------

guimg= gaussian(grayimg,sigma=0.2)


#--------------------edge finding sobel filter box------------------

sobimg=np.zeros((height,width),dtype=np.uint8)

#+++++++++++++++++++++++++++++
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])         # sobel array to product matrises
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobelgradian=[]
#+++++++++++++++++++++++++++++

for i in range(1,height-1):
    for j in range(1,width-1):
        #sobimg[i,j] = gradian(i,j,grayimg)                      # we can alse use this way but its not best way
        
        #+++++++++++++++++++++++++++++++++
        region = guimg[i-1:i+2, j-1:j+2]
        
        gx = np.sum(sobel_x * region)  
        gy = np.sum(sobel_y * region) 
        
        sobimg[i, j] = (gx**2 + gy**2)**0.5
        sobelgradian.append((gx**2 + gy**2)**0.5)
        #+++++++++++++++++++++++++++++++++

midgradian = sum(sobelgradian)/len(sobelgradian)

for i in range(1,height-1):
    for j in range(1,width-1):
        if sobimg[i,j] > midgradian+midgradian//2 :
            sobimg[i,j]=255
        else :
            sobimg[i,j]=0

#======================================================================================================================#
#top way is ok to find edges but because of handwriting code , edges are not monolithic so program cant find true shape#
# so if we use numpy and cv2 functions it will be completely  !  so i comment it                                                         #
#sobel_x = cv2.Sobel(grayimg, cv2.CV_64F, 1, 0, ksize=3)                                                                #
#sobel_y = cv2.Sobel(grayimg, cv2.CV_64F, 0, 1, ksize=3)                                                                #
#sobimg = cv2.magnitude(sobel_x, sobel_y)                                                                               #
#======================================================================================================================#

_ , sobimg = cv2.threshold(sobimg,127,255,cv2.THRESH_BINARY)    # change img to binary for function findcounters


#---------------------biggest container finding box----------------------

sobimg = cv2.convertScaleAbs(sobimg)
contour_img = np.zeros_like(img)                                 # making a black img in size of original img

counters, _ = cv2.findContours(sobimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # finding counters
largest=max(counters,key=cv2.contourArea)                                        # finding largest
cv2.drawContours(contour_img,[largest],-1,(0,0,255),2)                             # drawing on black page


#-----------------------output box---------------------------------------

cv2.imshow('orginal',img)
cv2.imshow('grayImage',grayimg)
cv2.imshow('bullued',guimg)
cv2.imshow('sobel filtered',sobimg)
cv2.imshow('ok',contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows

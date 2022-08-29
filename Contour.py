# difference between contour and edge detection is contour detection detect object that has a closed shape, whereas edge detection not necessarily detect a closed shape but it detects all intensity of pixels that changes dramatically

import cv2 as cv
import numpy as np

frame=cv.imread('nc.webp')
frame=  cv.resize(frame,(400,400))
cv.imshow('Original',frame)

blank = np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)
cv.imshow('Blank',blank)

# to reduce noise and decrease the number of contours detected
blur = cv.GaussianBlur(frame,(9,9),1)
cv.imshow("Gaussian Blur",blur)

gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

canny = cv.Canny(gray,50,90)
cv.imshow('Canny',canny)

# in order to find the contours, we should input the binary image, that is the return of either treshold or canny
# hierarchy contains set of certain contours that are child of certain contour
# RETR_LIST returns all contours, RETR_TREE returns all contours that has hierarchy, RETR_EXTERNAL returns only the contours that external/outside any contours (not child)
contours, hierarchy = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

# the drawed contours seems not much difference compared to canny. but now there are sets of pixels that are continous an
cv.drawContours(blank,contours,-1,(255,255,0),1)
cv.imshow('Contours',blank)
# the size of array contours represents how many contours in it
print(len(contours))
print(hierarchy)
cv.waitKey(0)

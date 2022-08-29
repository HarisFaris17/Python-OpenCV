import cv2 as cv
import numpy as np

# opencv stores pixels as BGR
frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

# change to grayscaled
# using the formula 0.3R+0.59G+0.11B since the human eye more sensitive to green color, following red and finally blue, hence the weighted for green is greatest, following red and finally blue.
gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

print('Original : {frame}'.format(frame=frame))
print(gray)

# since gray contain only the luminosity, hence it can't be restored to the original image
grayToBGR=  cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
# if the matrix/array/frame containing the image is single channel (only 1 elements not 3 elements like BGR), the method imshow will consider the matrix/array/frame is a grayscale image
cv.imshow('Gray to BGR',grayToBGR)

BGRtoHSV = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL)
# method imshow will consider the three channels matrix/array/frame is in BGR format, hence when an image in HSV formatted inputted, the display of the image looks weird
cv.imshow('BGR to HSV',BGRtoHSV)


cv.waitKey(0)


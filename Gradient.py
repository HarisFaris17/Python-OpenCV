import cv2 as cv
import numpy as np

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

laplacian = cv.Laplacian(gray,cv.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv.imshow('Laplacian',laplacian)

cv.waitKey(0)
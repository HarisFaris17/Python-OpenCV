import cv2 as cv
import numpy as np


frame = cv.imread('cat.jpg')

blank = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)

# pixels in OpenCV is stored as BGR not RGB
# split the image from three channel frame to three single channels frame. Each Splitted frame representing blue, green, and red channels
blue, green, red = cv.split(frame)


cv.imshow('Original',frame)
cv.imshow('Blue',blue)
cv.imshow('Green',green)
cv.imshow('Red',red)

blue = cv.merge([blue,blank,blank])
green = cv.merge([blank,green,blank])
red = cv.merge([blank,blank,red])

cv.imshow('Bluish Blue Channel',blue)
cv.imshow('Greenish Green Channel',green)
cv.imshow('Redish Red Channel',red)

cv.waitKey(0)
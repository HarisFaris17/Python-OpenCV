import cv2 as cv
import numpy as np

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

blank = np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)
cv.imshow('Blank',blank)

circle=cv.circle(blank.copy(),(255,100),100,(255,255,255),cv.FILLED)
cv.imshow('Circle',circle)

# masking is basically taking bitwise AND operator between mask image and real image
frameANDcircle = cv.bitwise_and(frame,circle)
cv.imshow('ANDed',frameANDcircle)

rectangle = cv.rectangle(blank.copy(),(255,100),(355,200),(255,255,255),cv.FILLED)
cv.imshow('Rectangle',rectangle)

# make weird shape
circleORrectangle = cv.bitwise_or(circle,rectangle)
cv.imshow('Circle OR rectangle',circleORrectangle)

# And take circleORrectangle as mask image
maskedFrame = cv.bitwise_and(frame,circleORrectangle)
cv.imshow('Frame AND weird shape',maskedFrame)

cv.waitKey(0)
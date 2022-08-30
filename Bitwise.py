import cv2 as cv
import numpy as np

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

blank = np.zeros((frame.shape[0],frame.shape[1],3),np.uint8)
cv.imshow('Blank',blank)

circle=cv.circle(blank.copy(),(255,100),100,(255,255,255),cv.FILLED)
cv.imshow('Circle',circle)

# bitwise and operator simply multiply (AND operator) bit by bit to each channels of frame and circle
# we can use anded to implement masking concept
# the white template is the mask
frameANDblank = cv.bitwise_and(frame,circle)
cv.imshow('ANDed',frameANDblank)

rectangle = cv.rectangle(blank.copy(),(255,100),(355,200),(255,255,255),cv.FILLED)
cv.imshow('Rectangle',rectangle)

# find the intersection between those two
circleANDrectangle =cv.bitwise_and(circle,rectangle)
cv.imshow('Circle AND Rectangle',circleANDrectangle)

# if we ORing rectangle/circle with frame the rectangle/circle shape will be painted on frame since the rectangle/circle shape has maximum value of a byte (255), since any byte (pixel) ORed with maximum value of a byte (255) only return maximum value of a byte also. Which means the return pixels on coordinate same as rectangle/circle shape has value 255.
frameORrectangle = cv.bitwise_or(frame,rectangle)
cv.imshow('Frame oR Rectangle',frameORrectangle)

# bitwise or operator simply add (OR operator) bit by bit to each channels of circle and rectangle
circleORrectangle = cv.bitwise_or(circle,rectangle)
cv.imshow('Circle OR rectangle',circleORrectangle)

# not simply inverse bit by bit of the image 0 become 1 and 1 become 0
NotCircle = cv.bitwise_not(circle)
cv.imshow('Not Circle',NotCircle)

NotFrame = cv.bitwise_not(frame)
cv.imshow('Not Frame',NotFrame)

# XOR bitwise operator will take XOR operator bit by bit between images
circleXORRectangle = cv.bitwise_xor(circle,rectangle)
# display the regions that only one of the shape allowed
cv.imshow('Circle XOR Rectangle',circleXORRectangle)

cv.waitKey(0)
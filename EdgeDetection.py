import cv2 as cv
import numpy as np

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

blur = cv.GaussianBlur(frame,(7,7),0)
cv.imshow('Blur',blur)

gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

canny = cv.Canny(gray,90,150)
cv.imshow('Canny',canny)

cv.waitKey(0)
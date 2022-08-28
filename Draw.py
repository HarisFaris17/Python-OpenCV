import numpy as np
import cv2 as cv

# np.zeros will return nd.array that has a shape of rows and column specified from the first 2 element of the argument shape. SInce Mat in cv not only it should a 2D array, but its element should contains 3 channel of color BGR, therefore each element in Mat is a array that has 3 element. Therefore it should be specified in the 3rd element in tuple of shape argument that it is an array of 3 element
frame=np.zeros((500,500,3),dtype=np.uint8)
cv.imshow('Blank',frame)

# draw a line on frame starting from pt1 to pt2 with line color color of certain thickness
cv.line(frame,(0,0),(300,100),(90,255,140),3)
print(frame)
cv.imshow('With line',frame)



cv.circle(frame,(255,255),10,(134,25,190),cv.FILLED,cv.LINE_AA)
cv.imshow('With circle',frame)

cv.rectangle(frame,(355,355),(450,450),(120,120,190),10,cv.LINE_8)
cv.imshow('With rectangle',frame)


cv.putText(frame,'Hey my name is Haris',(100,100),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,100),0,cv.LINE_4)
cv.imshow('With text',frame)
cv.waitKey(0)
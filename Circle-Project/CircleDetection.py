from turtle import goto
import cv2 as cv
import numpy as np
from cv2 import imshow


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret : break
    imshow('Original',frame)

    blur = cv.GaussianBlur(frame,(9,9),1)
    imshow('Blur',blur)

    gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    imshow('Gray',gray)

    canny = cv.Canny(gray,25,50)
    imshow('Canny',canny)

    circles=cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,50,param1=50,param2=30,minRadius=10,maxRadius=90)
    if(not isinstance(circles,type(None))):
        circles=np.uint16(np.around(circles))
        print('Matrix of circles : ')
        print(circles)
        for i in circles[0,:]:
            cv.putText(frame,'# of circles : {numberofcircles}'.format(numberofcircles=circles[0].shape[0]),(frame.shape[0]-100,50),cv.FONT_HERSHEY_TRIPLEX,0.7,(255,0,255),1)
            # draw the outer circle
            cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    imshow('Original with circled circle',frame)
    if(cv.waitKey(10)&0xFF == ord('d')): break

import cv2 as cv
from cv2 import CascadeClassifier
import numpy as np
def resizing(frame,scale):
    # src is the input frame, fx is the scaling number for width, fy is the scaling number for height
    return cv.resize(src=frame,dsize=(0,0),fx=scale,fy=scale,interpolation=cv.INTER_AREA)

# frame = cv.imread('Face-Detection-Project/faces2.jpg')
cap = cv.VideoCapture(0)
faceDetector = cv.CascadeClassifier()
faceDetector.load('Face-Detection-Project/FaceDetection.xml')
while True:
    ret, frame = cap.read()
    if(not ret) :continue
    # resizedFrame = resizing(frame,0.4)
    cv.imshow('Original',frame)

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',gray)

    faces = faceDetector.detectMultiScale(gray)

    print('# of faces:')
    print(len(faces))

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv.imshow('Detected',frame)
    if(cv.waitKey(25)&0xFF == ord('e')) : break
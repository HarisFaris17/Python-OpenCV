import cv2 as cv
import numpy as np
def resizing(frame,scale):
    # src is the input frame, fx is the scaling number for width, fy is the scaling number for height
    return cv.resize(src=frame,dsize=(0,0),fx=scale,fy=scale,interpolation=cv.INTER_AREA)

# frame = cv.imread('Face-Detection-Project/faces2.jpg')
cap = cv.VideoCapture(0)
faceDetector = cv.CascadeClassifier()
eyeDetector = cv.CascadeClassifier()
faceDetector.load('Face-Eye-Detection-Project/FaceDetection.xml')
eyeDetector.load('Face-Eye-Detection-Project/EyeDetection.xml')
while True:
    ret, frame = cap.read()
    if(not ret) :continue
    # resizedFrame = resizing(frame,0.4)
    cv.imshow('Original',frame)

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',gray)

    faces = faceDetector.detectMultiScale(gray)
    eyes = eyeDetector.detectMultiScale(gray)

    print(f'# of faces: {len(faces)}')
    print(f'# of eyes: {len(eyes)}')

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    for (x,y,w,h) in eyes:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv.imshow('Detected',frame)
    if(cv.waitKey(25)&0xFF == ord('e')) : break
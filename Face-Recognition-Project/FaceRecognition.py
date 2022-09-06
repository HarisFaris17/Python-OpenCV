import os
import cv2 as cv
import numpy as np

datasetPath = 'E:\Dataset'
trainingPath = f'{datasetPath}\Training'
people = os.listdir(trainingPath)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,1024)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.read('Face-Recognition-Project/TrainedModel.xml')

faceDetector = cv.CascadeClassifier()
faceDetector.load('Face-Recognition-Project/FaceDetection.xml')

while True:
    ret, frame = cap.read()
    frame=cv.resize(frame,(0,0),fx=0.5,fy=0.5)
    if (not ret) : exit(0)
    cv.imshow('Original',frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow('Gray',gray)
    faces = faceDetector.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        label, confidence=faceRecognizer.predict(gray[y:y+h,x:x+w])
        if(confidence<35):
            cv.putText(frame,f'Unknown:{np.around(confidence,2)}%',(x,y),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),1)
        else: cv.putText(frame,f'{people[label]}:{np.around(confidence,2)}%',(x,y),cv.FONT_HERSHEY_TRIPLEX,1,(0,0,255),1)
    cv.imshow('Detected',frame)
    if(cv.waitKey(25)==ord('d')):break
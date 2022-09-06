import cv2 as cv
import mediapipe as mp
import time

trainingDirectory = './training'

# get the 
doCar = ['turnLeft','turnRight','straight','backLeft','backRight']

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
hands = mp.solutions.hands.Hands()
now, prevTime = 0,0
while True:
    ret, frame = cap.read()
    if(not ret): break
    frame=cv.resize(frame,(0,0),fx=0.5,fy=0.5)
    cv.imshow('Originial',frame)
    # for better output, flip the frame (since the frame is mirrored by default) with respect to y-axis
    frame=cv.flip(frame,1)
    cv.imshow('Flipped',frame)
    now = time.time()
    cv.putText(frame,f'Frame {1/(now-prevTime+0.01)}',(frame.shape[1]-250,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    prevTime=time.time()
    cv.imshow('Originial with FPS',frame)
    # since mediapipe working in RGB format not BGR format, we need to convert the frame to RGB format
    results = hands.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    print(results.multi_hand_landmarks)
    # for handLandmarks in results.multi_hand_landmarks:
        # mp.solutions.drawing_utils.draw_landmarks()
    if (cv.waitKey(25)&0xFF==ord('e')):break
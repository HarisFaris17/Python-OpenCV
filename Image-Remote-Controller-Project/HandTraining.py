import cv2 as cv
import mediapipe as mp
import time

trainingDirectory = './training'

# what the remote-controlled car can do
doCar = ['turnLeft','turnRight','straight','backLeft','backRight','back']

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
hands = mp.solutions.hands
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
    cv.putText(frame,f'Frame {int(1/(now-prevTime))}',(frame.shape[1]-250,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    prevTime=time.time()
    cv.imshow('Originial with FPS',frame)
    # since mediapipe working in RGB format not BGR format, we need to convert the frame to RGB format
    results = hands.Hands().process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    multiHandLandmarks = results.multi_hand_landmarks
    print(multiHandLandmarks)
    if multiHandLandmarks:
        for handLandmarks in multiHandLandmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame,handLandmarks,hands.HAND_CONNECTIONS)
    cv.imshow('Frame with hand landmarks',frame)
    if (cv.waitKey(25)&0xFF==ord('e')):break
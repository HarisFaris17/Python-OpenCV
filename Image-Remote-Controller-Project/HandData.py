import cv2 as cv
import mediapipe as mp
import time
import pandas

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
    height, width, channels = frame.shape
    cv.imshow('Originial',frame)
    # for better output, flip the frame (since the frame is mirrored by default) with respect to y-axis
    frame=cv.flip(frame,1)
    cv.imshow('Flipped',frame)
    # since mediapipe working in RGB format not BGR format, we need to convert the frame to RGB format
    results = hands.Hands(True).process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    print(results.multi_handedness)
    multiHandLandmarks = results.multi_hand_landmarks
    print(multiHandLandmarks)
    if multiHandLandmarks:
        # x0 = min(handLandmarks.landmark.x for handLandmarks in multiHandLandmarks)

        for handLandmarks in multiHandLandmarks:
            landmarks = handLandmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(frame,handLandmarks,hands.HAND_CONNECTIONS)
            # landmark represented as normalized coordinate that's normalized by its frame's width and height
            normx0 = min([landmark.x for landmark in landmarks])
            normy0 = min([landmark.y for landmark in landmarks])
            normx1 = max([landmark.x for landmark in landmarks])
            normy1 = max([landmark.y for landmark in landmarks])
            # therefore to crop the hand region we should multiply it with width/height to get the exact coordinate
            # the result of multiplication of width/height with normalized coordinate is float, but since the frame is a np array, and np array can only be accessed with int slice
            x0 = int(width*normx0)
            x1 = int(width*normx1)
            y0 = int(height*normy0)
            y1 = int(height*normy1)
            print((x0,y0,x1,y1))
            croppedHand = frame[y0:y1,x0:x1]
            cv.imshow('Hand',croppedHand)

            # we have to find the min-max normalized data for training, hence we should find the scale for every x and y coordinate
            scalexAxis = (normx1-normx0)
            scaleyAxis = (normy1-normy0)
            for landmark in landmarks:
                scaledX = landmark.x*scalexAxis
                scaledY = landmark.y*scaleyAxis
    
    # add fps to the frame
    now = time.time()
    cv.putText(frame,f'Frame {int(1/(now-prevTime))}',(frame.shape[1]-250,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    prevTime=time.time()
    cv.imshow('Frame with hand landmarks and fps',frame)
    if (cv.waitKey(25)&0xFF==ord('e')):break


def modeTraining():
    
import os
import cv2 as cv
import mediapipe as mp
import time
import pandas as pd
import numpy as np
from CarAction import actionCar
from HandTraining import trainingDirectory,dataTrainingDirectory

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
hands = mp.solutions.hands
now, prevTime = 0,0

while True:
    print('Choose what action data will be collected')
    [print(f'{id}. {action}') for id,action in enumerate(actionCar)]
    action=int(input())
    # try to get the last index of png from the <action car> folder
    indexLastPNG = None
    try:
        namePNGs = os.listdir(trainingDirectory+'/'+actionCar[action])
        print(namePNGs)
        indexLastPNG = namePNGs[-1].split('.')[0] if len(namePNGs)>0 else 0
    except:
        # if the folder doesn't exist, create the folder
        os.makedirs(trainingDirectory+'/'+actionCar[action])
        indexLastPNG = 0

    # try to open the existing training data
    file = None
    try:
        # if the file previously doesn\'t exist
        # this will open the file with write mode but it will only write after the last character already exist
        file=open(trainingDirectory+dataTrainingDirectory,'a')
    except:
        # this will open the file with write mode but if the file dosnt exist, then create the file
        file=open(trainingDirectory+dataTrainingDirectory,'w')

    while True:
        ret, frame = cap.read()
        if(not ret): break
        frame=cv.resize(frame,(0,0),fx=0.5,fy=0.5)
        height, width, channels = frame.shape
        cv.imshow('Originial',frame)
        # for better output, flip the frame (since the frame is mirrored by default) with respect to y-axis
        frame=cv.flip(frame,1)
        cv.imshow('Flipped',frame)
        # used to save the frame to PNG
        copyFrame = frame.copy()
        # since mediapipe working in RGB format not BGR format, we need to convert the frame to RGB format
        results = hands.Hands(False,1).process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
        print(results.multi_handedness)
        multiHandLandmarks = results.multi_hand_landmarks
        print(multiHandLandmarks)
        normx0, normy0, normx1, normy1, normz0, normz1 = 0,0,0,0,0,0
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
                # sometimes when the hand is out of the frame, the detected coordinate of landmarks can be negative , hence the  slicing with below code will be wrong
                # hence we should specify that when the coordinate is negative, the value should be 0
                croppedHand = frame[y0 if y0>0 else 0:y1 if y1>0 else 0,x0 if x0>0 else 0:x1 if x0>0 else 0]
                cv.imshow('Hand',croppedHand)

                # we have to find the min-max normalized data for training, hence we should find the scale for every x and y coordinate
                scalexAxis = (normx1-normx0)
                scaleyAxis = (normy1-normy0)
                for landmark in landmarks:
                    scaledX = landmark.x*scalexAxis
                    scaledY = landmark.y*scaleyAxis

                # save the image to correct direcories
        
        # add fps to the frame
        now = time.time()
        cv.putText(frame,f'Frame {int(1/(now-prevTime))}',(frame.shape[1]-250,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        prevTime=time.time()
        cv.imshow('Frame with hand landmarks and fps',frame)
        keyPressed = cv.waitKey(25)
        # if the landmark want to be captured, hence the captured landmark can be used as training data
        if(keyPressed==ord('c')) : 
                if multiHandLandmarks:
                    for handLandmarks in multiHandLandmarks:
                        landmarks = handLandmarks.landmark

                        # we have to find the min-max normalized data for training, hence we should find the scale for every x and y coordinate
                        # normalized coordinate x,y already calculated above, hence we only have to calculate normalized in z-axis
                        normz0 = min([landmark.z for landmark in landmarks])
                        normz1 = max([landmark.z for landmark in landmarks])
                        scalexAxis = (normx1-normx0)
                        scaleyAxis = (normy1-normy0)
                        scalezAxis = (normz1-normz0)
                        for id,landmark in enumerate(landmarks):
                            scaledX = landmark.x*scalexAxis
                            scaledY = landmark.y*scaleyAxis
                            scaledZ = landmark.z*scalezAxis
                            file.write(f'{scaledX},{scaledY},{scaledZ}')
                            print(len(landmarks))
                            # if 0its the last landmark ini the captured, then write newline hence each data training can be separated. If otherwise (which means ) then write ,
                            # TODO:
                            if id==len(landmarks)-1 : 
                                print(str(action))
                                file.write(',')
                                file.write(str(action))
                                file.write('\n')
                            else : file.write(',')
                indexLastPNG+=1
                cv.imwrite(f'{trainingDirectory}/{actionCar[action]}/{indexLastPNG}.png',copyFrame)
        if (keyPressed==ord('e')):
            cv.destroyAllWindows()
            file.close()
            break


# def modeData():
    
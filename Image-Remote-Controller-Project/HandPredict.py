import pickle
import cv2 as cv
import mediapipe as mp
import numpy as np
from HandTraining import trainingDirectory,modelHandDirectory
from sklearn.ensemble import RandomForestClassifier

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_WIDTH,720)

# the input that will be passed should be a 2D array and have 63 columns. The reasons there are 63 columns is that the there are 21 landmarks of hand and each landmark contains 3 data (x,y,z) hence there are 21*3=63 features/columns
x=np.zeros((1,63),float)
print(x)

# Load the model
classifier = pickle.load(open(trainingDirectory+modelHandDirectory,'rb'))
hands = mp.solutions.hands
while True:
    ret, frame = cap.read()
    cv.resize(frame,(0,0),fx=0.5,fy=0.5)
    frame=cv.flip(frame,1)
    cv.imshow('Original',frame)
    multiHandLandmarks = hands.Hands(False,1).process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)).multi_hand_landmarks
    if multiHandLandmarks:
        for handLandmarks in multiHandLandmarks:
            landmarks = handLandmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(frame,handLandmarks,hands.HAND_CONNECTIONS)
            cv.imshow('Landmarked frame',frame)
            flatLandmark=np.array([],float)
            # since the trained data of landmarks were normalized with min-max, hence the input of production/test data should be min-max normalized also
            normx0 = min([landmark.x for landmark in landmarks])
            normy0 = min([landmark.y for landmark in landmarks])
            normz0 = min([landmark.z for landmark in landmarks])
            normx1 = max([landmark.x for landmark in landmarks])
            normy1 = max([landmark.y for landmark in landmarks])
            normz1 = max([landmark.z for landmark in landmarks])
            scalexAxis = (normx1-normx0)
            scaleyAxis = (normy1-normy0)
            scalezAxis = (normz1-normz0)
            for idx,landmark in enumerate(landmarks):
                # print(flatLandmark)
                scaledX = landmark.x*scalexAxis
                scaledY = landmark.y*scaleyAxis
                scaledZ = landmark.z*scalezAxis

                x[0,idx*3+0]=scaledX
                x[0,idx*3+1]=scaledY
                x[0,idx*3+2]=scaledZ

                # flatLandmark=np.append(flatLandmark,landmark.y)
                # flatLandmark=np.append(flatLandmark,landmark.z)
                # print(flatLandmark)
            # x=np.insert(x,0,[flatLandmark],axis=0)
        print(flatLandmark)
        print(f'flatlandmarks : {x}')

        print(classifier.predict(x))
    if(cv.waitKey(25)==ord('e')): break
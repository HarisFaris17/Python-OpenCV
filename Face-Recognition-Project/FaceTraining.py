import os
import cv2 as cv
import numpy as np
datasetPath = 'E:\Dataset'
trainingPath = f'{datasetPath}\Training'
people = os.listdir(trainingPath)

faceRecognizer = cv.face.LBPHFaceRecognizer_create()

faceDetector = cv.CascadeClassifier()
faceDetector.load('Face-Recognition-Project/FaceDetection.xml')

labels = []
features = []

for i,person in enumerate(people):
    label = i
    print(label)
    for j in os.listdir(trainingPath+'\\'+person):
        frame = cv.imread(trainingPath+'\\'+person+'\\'+j)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        for (x,y,w,h) in faces:
            faceCropped = gray[y:y+h,x:x+w]
            cv.imshow(f'person{i} {j}',faceCropped)
            features.append(faceCropped)
            labels.append(label)

cv.waitKey(0)
print(f'labels : {labels}')
print(f'frame : {len(features)}')
features = np.array (features,'object')
labels = np.array(labels,'int')
faceRecognizer.train(features,labels)
faceRecognizer.save('Face-Recognition-Project/TrainedModel.xml')
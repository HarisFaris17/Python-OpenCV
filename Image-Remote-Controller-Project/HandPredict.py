import pickle
import cv2 as cv
from HandTraining import trainingDirectory,modelHandDirectory
from sklearn.ensemble import RandomForestClassifier

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_WIDTH,720)

# Load the model
classifier = pickle.load(open(trainingDirectory+modelHandDirectory,'rb'))

# while True:

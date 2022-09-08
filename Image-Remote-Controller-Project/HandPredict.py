import pickle
import cv2 as cv
import 
from sklearn.ensemble import RandomForestClassifier

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_WIDTH,720)

# Load the model
# classifier = pickle.load()

# while True:

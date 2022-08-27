import cv2 as cv
from cv2 import resizeWindow
from cv2 import INTER_AREA

def resizing(frame,scale):
    # src is the input frame, fx is the scaling number for width, fy is the scaling number for height
    return cv.resize(src=frame,dsize=(0,0),fx=scale,fy=scale,interpolation=INTER_AREA)


cap = cv.VideoCapture('file.mp4')

while True:
    ret, frame = cap.read()
    frame= resizing(frame,0.1)
    cv.imshow('Resized video',frame)
    if( cv.waitKey(10) & 0xFF==ord('d')):break



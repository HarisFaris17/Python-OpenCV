import cv2 as cv
import numpy as np
from cv2 import imshow

img = cv.imread('cat.jpg')
print(img.shape[0])
imshow('Cat',img)

def shift(img,x,y):
    # the wardAffine only receive transformation marix which that matrix only contain float. Therefore the transformation matrix should be floated using function np.float. The first two column of the transformation matrix is the matrix multiplication and the last column of the transformation matrix is the addition vector, therefore each row in transformation matrix needs 3 element. When warpAffine executed, program will multiply the src with multiplication matrix. The result then translated by addition vector.
    shiftTransformationMatrix = np.float32([[1,0, x],[0,1,y]])
    # img.shape[1] represent the number of row whereas img.shape[0] represent the number of columns
    return cv.warpAffine(img,shiftTransformationMatrix,(img.shape[1],img.shape[0]))

shiftedImgRightBelow = shift(img,50,50)
# negative value in x-axis means will translated to the left
shiftedImgLeftBelow = shift(img,-50,50)
# negative value in y-axis means will translated to the top
shiftedImgLeftTop = shift(img,-50,-50)
imshow('Shifted Right Below Image',shiftedImgRightBelow)
imshow('Shifted Left Below Image',shiftedImgLeftBelow)
imshow('Shifted Left Top Image',shiftedImgLeftTop)


def rotate(img,rot,x,y):
    rotationTransformationMatrix = cv.getRotationMatrix2D((x,y),rot,1)
    return cv.warpAffine(img,rotationTransformationMatrix,(img.shape[1],img.shape[0]))

rotatedImage = rotate(img,90,img.shape[1]/2,img.shape[0]/2)
imshow("Rotated image",rotatedImage)

flippedHorizontalVertical = cv.flip(img,-1)
flippedHorizontal = cv.flip(img,1)
flippedVertical = cv.flip(img,0)
imshow('Flipped horizontal and vertical',flippedHorizontalVertical)
imshow('Flipped horizontal',flippedHorizontal)
imshow('Flipped vertical',flippedVertical)
cv.waitKey(0)
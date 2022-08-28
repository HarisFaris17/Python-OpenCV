import cv2 as cv
from cv2 import imshow

img = cv.imread('cat.jpg')
# the order of channels in each element of matrix 2D of frame is actually reversed becomes BGR. Therefore here to convert the image to gray, it should be specified that the conversion takes on BGR ordered image to gray scaled image
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray2=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
# but there are not much difference between the conversion from BGR and RGB
cv.imshow('Gray scaled',gray)
imshow('Gray scaled 2',gray2)
imshow('Cat',img)


# blurred image is often used compared non-blurred image to detect edge detection, since the blurred image will get rid of the small edges or even noise
# we can increase the blur by increasing the kernel
blur = cv.GaussianBlur(gray,(3,3),sigmaX=0)
blur2 = cv.GaussianBlur(gray,(5,5),sigmaX=0)
blur3 = cv.GaussianBlur(gray,(7,7),sigmaX=0)

imshow('Blur',blur)
imshow('Blur 2',blur2)
imshow('Blur 3',blur3)

# find edges
# trying difference treshold
# canny function will convert the gray scaled image into an image that can only represent black or white pixels.
canny=cv.Canny(blur,5,9,4)
canny2=cv.Canny(blur2,2,3,4)
canny3=cv.Canny(blur3,5,15,4)
imshow('Edges',canny)
imshow('Edges 2',canny2)
imshow('Edges 3',canny3)


# we can grow (dilate) or shrink the edges (erode) by using dilate and erode function
dilated = cv.dilate(canny,(3,3),iterations=3)
dilated2 = cv.dilate(canny2,(5,5),iterations=1)
dilated3= cv.dilate(canny3,(7,7),iterations=1)
imshow("Dilated", dilated)
imshow("Dilated 2", dilated2)
imshow("Dilated 3", dilated3)

erode = cv.erode(canny,(3,3),iterations=3)
erode2 = cv.erode(canny2,(5,5),iterations=1)
erode3 = cv.erode(canny3,(3,3),iterations=1)
imshow("Erode", erode)
imshow("Erode 2", erode2)
imshow("Erode 3", erode3)

# since img is a ndarray we can crop certain pixels by assigning only for certain range of columns and certain range of rows
crop = img[250:,250:400]
imshow('Cropped image',crop)

cv.waitKey(0)
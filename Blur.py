import cv2 as cv 

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

# the higher the kernel, the more blur the image
# blur is average blur. Basically it taks kernal[0] as the number of pixels in x-axis and kernel[1] as the number of pixelsin y-axis, hence the total of surrounding pixels calculated for each pixels is kernel[0]*kernel[1]
blur = cv.blur(frame,(3,3))
cv.imshow('Casual blur',blur)

# gaussian blur method preferred in edge detection
gaussian = cv.GaussianBlur(frame,(3,3),0)
cv.imshow('Gaussian',gaussian)

# median blur usually used to reduce the noise
median = cv.medianBlur(frame,3)
cv.imshow('Median',median)

# bilateral keeps edges
bilateral = cv.bilateralFilter(frame,3,2,2)
cv.imshow('Bilateral',bilateral)

cv.waitKey(0)
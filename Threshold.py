import cv2 as cv
import matplotlib.pyplot as plt

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

ret, thresholdedFrame = cv.threshold(gray,90,255,cv.THRESH_BINARY)
cv.imshow('Threshold',thresholdedFrame)
print(thresholdedFrame)

histGray = cv.calcHist([gray],[0],None,[256],[0,256])

# apply masking to gray with mask image is thresholdFrame
maskedGray = cv.bitwise_and(gray,thresholdedFrame)
cv.imshow('Masked Gray',maskedGray)

histMaskedGray = cv.calcHist([maskedGray],[0],None,[256],[0,256])
print(histMaskedGray)


plt.plot(histGray)
plt.title('Histogram')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')

plt.figure(2)
plt.plot(histMaskedGray[1:])
plt.title('Histogram Masked')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')

plt.show()



cv.waitKey(0)
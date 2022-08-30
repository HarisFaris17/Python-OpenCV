import cv2 as cv
import matplotlib.pyplot as plt

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# input threshold must be grayscaled
# function threshold will convert all pixels of grayscaled image that have intensities (binary represented) greater than thresh and less than maxVal to maxVal. And convert all pixels that have intensities less than tresh and greater than maxVal to 0.
ret, thresholdedGray = cv.threshold(gray,90,255,cv.THRESH_BINARY)
cv.imshow('Threshold',thresholdedGray)
print(thresholdedGray)

# cv.THRESH_BINARY_INV will inverse the above thresholdedGray
retInv, thresholdedGrayInv = cv.threshold(gray,90,255,cv.THRESH_BINARY_INV)
cv.imshow('Threshold Inverse',thresholdedGrayInv)

# Adaptive thresholding simply find the best thresh value
AdaptiveThreshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
print('Adaptive')
print(AdaptiveThreshold)
ThreeChannelsAdaptiveThreshold = cv.merge([AdaptiveThreshold,AdaptiveThreshold,AdaptiveThreshold])
print(ThreeChannelsAdaptiveThreshold)
cv.imshow('Adaptive Thresholding',AdaptiveThreshold)

ThresholdedFrameByAdaptiveThreshold = cv.bitwise_and(frame,ThreeChannelsAdaptiveThreshold)
cv.imshow('Haa',ThresholdedFrameByAdaptiveThreshold)

histGray = cv.calcHist([gray],[0],None,[256],[0,256])

# apply masking to gray with mask image is thresholdFrame
maskedGray = cv.bitwise_and(gray,thresholdedGray)
cv.imshow('Masked Gray',maskedGray)

histMaskedGray = cv.calcHist([maskedGray],[0],None,[256],[0,256])
print(histMaskedGray)


plt.plot(histGray)
plt.title('Histogram')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')

plt.figure(2)
# since the pixels that have 0 intensity is much more than pixels that have other intensities, therefore we shouldn't included it so that the histogram can be seen clearly of other intensities
plt.plot(histMaskedGray[1:])
plt.title('Histogram Masked')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')

plt.show()



cv.waitKey(0)
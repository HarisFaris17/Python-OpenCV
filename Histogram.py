import cv2 as cv
import matplotlib.pyplot as plt

# Histogram simply graph the frequencies of intervals.

frame = cv.imread('cat.jpg')
cv.imshow('Original',frame)

# There are three ways calculating histogram. First, calculate using plt.hist. Second using the value return by plt.hist index [0]. and last using cv.calcHist

# Histogram calculator can only take one flat array 1D array (alias, in image processing is single channel array)
gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

# Histogram calculator can only take one flat array 1D array, hence array/matrix/image gray flattened to 1D array
# if the array/matrix/image gray not flattened, the function plt.hist will only take index 0 of gray into calculation of histogram
# bins denote how many intervals of binary representation of pixels. Each binary representation needs 8-bit. Which means there are 2^8=256 possible values of binary representations. We want to see individual frequency of binary representation of pixels, therefore we should assign 256 to bins. Hence each intervals only contain 1 binary representation of pixels.
# plt.hist automatically create figure. If no figure defined before, hence the figure will be Figure 1.
hist=plt.hist(gray.flatten(),256)
print(hist)
# plt.figure(figsize=(19,19),dpi=300,facecolor=[1,1,1],edgecolor=[0.5,0.5,0.5])
plt.title('Histogram')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')

# same histogram as above but with plt.plot so that the graph looks continue
# specify the order of figure, otherwise the previous figure will be replaced
plt.figure(2)
plt.title('Histogram')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')
# function plt.plot can take one 1D array as input. The plt.plot will graph index as x-axis whereas the array[index] as y-axis
plt.plot(hist[0])


histogramGray = cv.calcHist([gray],[0],None,[256],[0,256])
print(histogramGray)
plt.figure(3)
plt.title('Histogram')
plt.xlabel('Binary representation of pixels')
plt.ylabel('# of pixels')
plt.plot(histogramGray)
# plt.show() acts same as cv.waitKey, they stopped the program. The only difference is cv.waitKey will be over if we press any key or the time specified as argument finished whereas plt.show() will only be over if all windows of figure closed.
plt.show()

cv.waitKey(0)
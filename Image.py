import cv2 as cv
from cv2 import imshow
image = cv.imread('NC.webp');
imshow('An image',image);

# will wait any key for infinite time. When a key is pressed then the window displaying the image will be closed
cv.waitKey(0);

# if we change the wait to delay ms, then when the execution of that frame has been windows then the windows will be closed
# will wait any key for 2000 ms, if there is no key pressed within 2000ms after the windows displayed, then the windows will be closed but if any key pressed within that period then the windows will be closed immedietly
# cv.waitKey(2000)
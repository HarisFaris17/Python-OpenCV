import cv2 as cv
capture = cv.VideoCapture(0)
if(capture.isOpened()==False) : print("error opening the file")

while True:
    ret, frame=capture.read()
    if (not ret) :break
    cv.imshow('Video',frame)
    # method waitKey will wait within the argument ms. If in that interval time there is a key pressed, the method waitKey will return the unicode code of the pressed key. Otherwise if there is no key pressed then method waitKey will return -1, alias in 8 bit binary data is 255, alias in the hexadecimal numbering is 0xFF.
    # Therefore when cv.waitKey() executed, the below code cv.waitKey(10) & 0xFF will inspect if there is key pressed, then do the logical bitwise AND, alias multiply for each bit from the return value of cv.waitKey(10) and 0xFF, the return value cv.waitKey(10) & 0xFF will be the unicode code of the pressed key, or if there is no key pressed within 10 ms, the return value must be 0xFF in hexadecimal or 255 in decimal. Hence when the return value of cv.waitKey(10) & 0xFF compared with ord('d'), it will return true if the method waitKey(10) return the unicode code 'd' since the ord('d') will also return the univode code 'd'. The purpose of this if statement is the program will break outside while loop if the key 'd' is pressed
    if( cv.waitKey(10) & 0xFF==ord('d')):break

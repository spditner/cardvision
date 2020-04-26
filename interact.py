from imutils import contours
import imutils
import code
import cv2
import numpy

#ref = cv2.imread("3-red-shaded-oval.jpg")
ref = cv2.imread("2-green-solid-diamond.jpg")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 100, 255, cv2.THRESH_BINARY_INV)[1]
#window = cv2.namedWindow("cam-test",cv2.WINDOW_AUTOSIZE)
#code.interact(local=locals())
cv2.imwrite("test.jpg", ref)

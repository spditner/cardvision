import cv2
import sys
cam = cv2.VideoCapture(0)
s, img = cam.read()
cv2.imwrite(sys.argv[1], img)
cam.release()

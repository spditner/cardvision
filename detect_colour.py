# https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

import sys

import imutils
import cv2

from ShapeDetector import ShapeDetector
from ColourDetector import ColourDetector

def look(image):
#    return
    cv2.imshow("image", image)
    cv2.waitKey(0)

image = cv2.imread(sys.argv[1])
look(image)
resized = imutils.resize(image, width=1200)
ratio = image.shape[0] / float(resized.shape[0])
look(resized)

# convert the resized image to grayscale, blur it slightly,
# and threshold it
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
look(blurred)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
look(gray)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
look(lab)
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
look(thresh)
# Invert
#thresh = cv2.bitwise_not(thresh)
#cv2.imshow("image", thresh); cv2.waitKey(0)

#cv2.imwrite(sys.argv[2], thresh)

# find contours in the thresholded image and initialize the shape detector
# (white=object, black=background)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
cl = ColourDetector()

for c in cnts:
    # Calculate center of contours
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    # Detect shape
    shape = sd.detect(c)
    # Label colours
    color = cl.label(lab, c)

    # scale contours back up to original image size
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")

    # Draw the contours around the shapes
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    # Labels
    text = "{} {}".format(color, shape)
    cv2.putText(
        image, text,
        (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2
    )

look(image)
cv2.imwrite("final.jpg", image)
cv2.waitKey(0)

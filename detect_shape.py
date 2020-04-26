# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

from ShapeDetector import ShapeDetector
import imutils
import cv2
import sys
import numpy as np

def look(image):
#    return
    cv2.imshow("image", image)
    cv2.waitKey(0)

image = cv2.imread(sys.argv[1])
look(image)
resized = imutils.resize(image, width=400)
ratio = image.shape[0] / float(resized.shape[0])
look(resized)

brightness = float('50')
contrast = float('50')
adjusted = imutils.adjust_brightness_contrast(
    resized,
#    contrast=contrast,
    brightness=brightness,
)
look(adjusted)


# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
look(gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
look(blurred)

thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
look(thresh)

# Invert
thresh = cv2.bitwise_not(thresh)
look(thresh)

# wiggle it around
kernel = np.ones((5,5), np.uint8)  # note this is a horizontal kernel
dilate = cv2.dilate(thresh, kernel, iterations=1)
look(dilate)


target = dilate
look(target)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
if circles:
    print("Circles!")
else:
    print("No Circles")

#cv2.imwrite(sys.argv[2], thresh)

# find contours in the thresholded image and initialize the shape detector
# (white=object, black=background)
cnts = cv2.findContours(target.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

for c in cnts:
    # Calculate center
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    # Detect shape
    shape = sd.detect(c)

    # scale contours back up to original image size
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(
        image, shape,
        (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2
    )

look(image)
cv2.imwrite("final.jpg", image)
cv2.waitKey(0)

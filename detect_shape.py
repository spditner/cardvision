# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

import sys

import cv2
import numpy as np

import imutils

from ShapeDetector import ShapeDetector

def look(image):
    cv2.imshow("image", image)
    cv2.waitKey(delay=0)

image = cv2.imread(sys.argv[1])
look(image)
resized = image.copy()
#resized = imutils.resize(image, width=400)
ratio = image.shape[0] / float(resized.shape[0])
look(resized)

def chain(img, methods):
    for method in methods:
        img = method(img)
        look(img)
    return img

def adjust(img):
    brightness = float('50')
    contrast = float('-10')
    img = imutils.adjust_brightness_contrast(
        img,
        contrast=contrast,
        brightness=brightness,
    )
    return img

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def threshold(img):
    return cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

def invert(img):
    return cv2.bitwise_not(img)

def wiggle(img):
    kernel = np.ones((5,5), np.uint8)  # note this is a horizontal kernel
    return cv2.dilate(img, kernel, iterations=1)

def find_circles(img):
    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
#    params.filterByArea = True
#    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Set Convexity filtering parameters
#    params.filterByConvexity = True
#    params.minConvexity = 0.1

    # Set inertia filtering parameters
#    params.filterByInertia = True
#    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(
        image, keypoints, blank, (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    number_of_blobs = len(keypoints)
    print("Number of Circular Blobs: {}".format(number_of_blobs))

def find_circles2(img):
    # detect circles in the image
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 40)
    if circles is not None:
        print("Circles!")
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for circle in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            circle = circle.astype("float")
            circle *= ratio
            circle = circle.astype("int")
            (x, y, r) = circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    else:
        print("No Circles")


# find contours in the thresholded image and initialize the shape detector
# (white=object, black=background)
def find_shapes(img):
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

target = chain(
    resized,
    [
        adjust,
        grayscale,
        blur,
        threshold,
        invert,
        wiggle,
    ],
)

#target2 = chain(
#    resized,
#    [
#        adjust,
#        grayscale,
#    ],
#)

#find_circles(target)
#find_circles2(target2)
find_shapes(target)

cv2.imwrite("final.jpg", image)
cv2.imshow("image", image)
cv2.waitKey(0)

# https://www.pyimagesearch.com/2016/02/15/determining-object-color-with-opencv/

import sys

import imutils
import cv2
import numpy as np

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(img, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def look(img):
    return
    cv2.imshow("image", img)
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
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
look(thresh)

# find contours in the thresholded image and initialize the shape detector
# (white=object, black=background)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for i, c in enumerate(cnts):
    # Calculate center of contours
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    if len(approx) == 4:
        #pts = np.array([(0,0),(0,0),(0,0),(0,0)], dtype = "float32")
        # apply the four point tranform to obtain a "birds eye view" of
        # the image
        warped = four_point_transform(image, approx.reshape(4, 2) * ratio)
        cv2.imwrite("cards/%02d-four-sided.png" % i, warped)
    else:
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cropped = image[y: y+h, x: x+w]
        cv2.imwrite("cards/%02d-reject.png" % i, cropped)

for i, c in enumerate(cnts):
    # scale contours back up to original image size
    # Why is this not needed for the transform above??
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")

    # Draw the contours around the shapes
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    if len(approx) == 4:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    else:
        cv2.drawContours(image, [c], -1, (255, 255, 0), 2)

    # extract contour
    # https://stackoverflow.com/questions/44830110/copy-area-inside-contours-to-another-image
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    # Draw a box around the contour
    # Clip out the image
    box = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
#    cropped = image[y: y+h, x: x+w]
#    cv2.imshow("Show Boxes", cropped)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    cv2.imwrite("cards/card-%02d.png" % i, cropped)

cv2.imshow("image", image)
cv2.imwrite("final.jpg", image)
cv2.waitKey(0)

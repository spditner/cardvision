import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, contour):
        # Init shape and approximate contours
        shape = "unidentified"
        # Determine perimeter of contour
        perimeter = cv2.arcLength(contour, True)
        # Create an approximate polygon
        # is normal 1-5%
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # compute boudning box, and compute aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle {}".format(len(approx))

        return shape

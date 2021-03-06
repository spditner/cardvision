from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import sys

class ColourDetector:
    def __init__(self, colors=None):

        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        if not colors:
            colors = OrderedDict({
                # Use "Digital Color Meter" app to help you find colours on MacOS
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                #"blue": (0, 0, 255),
                "violet": (140, 60, 140),
            })

        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        # ΔL* (L* sample minus L* standard) = difference in lightness and darkness (+ = lighter, – = darker)
        # Δa* (a* sample minus a* standard) = difference in red and green (+ = redder, – = greener)
        # Δb* (b* sample minus b* standard) = difference in yellow and blue (+ = yellower, – = bluer)
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
                minDist = (d, i)

        return self.colorNames[minDist[1]]

    def find(self, color):
        #import pdb; pdb.set_trace()
        lcolor = np.zeros((1, 1, 3), dtype="uint8")
        lcolor[0] = color
        # Seems to want it in some funny array shape
        color = cv2.cvtColor(lcolor, cv2.COLOR_RGB2LAB)[0][0]
        minDist = (np.inf, None)
        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], color)
            if d < minDist[0]:
                minDist = (d, i)
        sys.stderr.write(
            "minDist: %r / %r / %r\n" % (
                minDist[0],
                tuple(color),
                tuple(self.lab[minDist[1]][0]),
            )
        )
        if minDist[0] > 80:
            return None
        return self.colorNames[minDist[1]]

import sys
from collections import OrderedDict

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-i2", "--image2", required=True, help="Path to the image")
#ap.add_argument("-o", "--output", required=True, help="Path to output image")
ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters")
ap.add_argument("-s", "--show", required=False, type=bool, help="Show result", default=False)
args = vars(ap.parse_args())

def kmeans(image, clusters, calibrate=None):
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to LAB format 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # show our image
    if False:
        plt.figure()
        plt.axis("off")
        plt.imshow(image)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=clusters)
    clt.fit(image)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)

    def sorted_idx(array):
        return sorted(range(len(array)), key=lambda k: array[k])

    s_hist = []
    s_clusters = []
    discarded = 0.0
    s_discarded = []
    for idx in sorted_idx(hist):
        # Discard any bin that is > 50%
        if hist[idx] > 0.5:
            # Assume top discarded is the background
            if not s_discarded:
                discarded = hist[idx]
                s_discarded = clt.cluster_centers_[idx]
            continue
        s_hist.append(hist[idx])
        s_clusters.append(clt.cluster_centers_[idx])

    # Probably a card... If a calibration colour provided, try to match it
    # RESULT: Did not improve the situation
    if len(s_hist) == (clusters-1) and len(s_discarded) and calibrate is not None:
        adj = [None] * len(calibrate)
        for idx, color in enumerate(calibrate):
            adj[idx] = color / s_discarded[idx]
        for idx, _ in enumerate(s_clusters):
            s_clusters[idx] *= adj[idx]
        sys.stderr.write(">>> Calibrate %s" % adj)

    # Rescale
    if discarded:
        for i, _ in enumerate(s_hist):
            s_hist[i] /= (1-discarded)

    #bar = utils.plot_colors(hist, clt.cluster_centers_)
    bar = utils.plot_colors(s_hist, s_clusters)
    # show our color bart
    if False:
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()
    return OrderedDict(zip(s_hist, s_clusters)), s_discarded

import glob
from ColourDetector import ColourDetector
def compare_many(src, compare_dir, clusters):
    image = cv2.imread(src)
    km1, calibrate = kmeans(image, clusters)
    detector = ColourDetector(colors=km1)

    filenames = glob.glob(compare_dir)
    filenames.sort()
    for filename in filenames:
        image2 = cv2.imread(filename)
        same_colour = False
        try:
            km2, _ = kmeans(image2, clusters) #, calibrate)
        except ValueError:
            sys.stderr.write("Not enough colours")
        else:
            same_colour = any([detector.find(c) for c in km2.values()])

        sys.stderr.write("%s\n%s\n%s\n" % (km1, km2, same_colour))
        sys.stdout.write("%s:%d\n" % (filename, not same_colour))

    sys.stderr.write("%s\n" % calibrate)

if __name__ == "__main__":
    compare_many(
        args["image"],
        args["image2"],
        args["clusters"]
    )


##same_colour = True
##for pct, colour in km2.items():
##    matched_pct = detector.find(colour)
##    if not matched_pct:
##        same_colour = False
#same_colour = any([detector.find(c) for c in km2.values()])
#print("%s\n%s\n%s" % (km1, km2, same_colour))
#if same_colour:
#    sys.exit(0)
#sys.exit(-1)

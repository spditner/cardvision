# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
#ap.add_argument("-o", "--output", required = True, help = "Path to output image")
ap.add_argument("-c", "--clusters", required = True, type = int, help = "# of clusters")
ap.add_argument("-s", "--show", required=False, type=bool, help="Show result", default=False)
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)

def sorted_idx(array):
    return sorted(range(len(array)), key=lambda k: array[k])

s_hist = []
s_clusters = []
discarded = 0
for idx in sorted_idx(hist):
    # Discard any bin that is > 50%
    if hist[idx] > 0.5:
        discarded += hist[idx]
        continue
    s_hist.append(hist[idx])
    s_clusters.append(clt.cluster_centers_[idx])

# Rescale
if discarded > 0:
    for i, _ in enumerate(s_hist):
        s_hist[i] /= (1-discarded)

#bar = utils.plot_colors(hist, clt.cluster_centers_)
bar = utils.plot_colors(s_hist, s_clusters)
for a, b in zip(s_hist, s_clusters):
    print(a,tuple(b))
#    print(s_hist)
#    print(s_clusters)

# show our color bart
if args["show"]:
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

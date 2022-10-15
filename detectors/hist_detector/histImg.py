# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

#https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
# load the image and show it
image = cv2.imread(args["image"])
#cv2.imshow("image", image)
#cv2.waitKey(0)

'''
# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
cv2.waitKey(0)

#cv2.calcHist(images, channels, mask, histSize, ranges)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
'''

# grab the image channels, initialize the tuple of colors,
# the figure and the flattened feature vector
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
features = []
# loop over the image channels
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and
	# concatenate the resulting histograms for each
	# channel
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	#print(type(hist))
	features.extend(hist)
	# plot the histogram
	plt.plot(hist, color = color)
	plt.xlim([0, 256])
plt.show()
# here we are simply showing the dimensionality of the
# flattened color histogram 256 bins for each channel
# x 3 channels = 768 total values -- in practice, we would
# normally not use 256 bins for each channel, a choice
# between 32-96 bins are normally used, but this tends
# to be application dependent
print("flattened feature vector size: %d" % (np.array(features).flatten().shape))


dif = hist - hist2

#https://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html
from scipy import stats
#stats.ttest_ind(hist,hist2)
#stats.ttest_rel(hist,hist2)
#stats.chisquare(f_obs=hist, f_exp=hist2)
stats.ks_2samp(hist.flatten(),hist2.flatten())
#https://stats.stackexchange.com/questions/57885/how-to-interpret-p-value-of-kolmogorov-smirnov-test-python

# equals to t-test_independent
t = (np.mean(hist)-np.mean(hist2)) / (np.sqrt(np.std(hist)**2 + np.std(hist2)**2) + np.sqrt(1/len(hist)))
#https://stats.stackexchange.com/questions/189362/comparing-z-scores-from-different-data-sets
tp = stats.ttest_ind(hist,hist2)
print(tp.statistic[0])
print(tp.pvalue[0])

#GET THE AVERAGE OF HISTOGRAMS
histAvg = (hist1+hist2+hist3)/3


#compare histograms
#https://theailearner.com/tag/cv2-comparehist/
# Load the images
img1 = cv2.imread('my_pictures/yellow_bmw.jpg')
img2 = cv2.imread('my_pictures/yellow_power.jpg')
# Convert it to HSV
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# Calculate the histogram and normalize it
hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
# find the metric value
metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)  #or HISTCMP_CHISQR



#https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/



#save the histograms and then load them back
with open('histo.txt', 'w') as file:
   file.write(list(hist))  # save 'hist' as a list string in a text file
   
with open('histo.txt', 'r') as file:
   hist = np.array(eval(file.read()) # read list string and convert to array
   
np.savetxt('test1.txt', a, fmt='%d')
b = np.loadtxt('test1.txt', dtype=int)
   
   
#iterate through a folder
for name in os.listdir('.'):   
   print(name)


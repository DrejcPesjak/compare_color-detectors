from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import argparse
import cv2
import os


colors = ["blue","green","black","red","white", "yellow"]
hsv = [180,256,256]

if __name__ == "__main__":
	# main
	ap = argparse.ArgumentParser()
	# only one should be present
	ap.add_argument("-i", "--image", help = "Path to the image")
	ap.add_argument("-d", "--dataset", help = "Path to training dataset")
	args = vars(ap.parse_args())
	
	if(args["image"]):
		# PREDICTION
		# load saved average histgrams for each color
		histAvg = []
		for name in colors:
			histAvg.append(np.loadtxt(name+'.txt'))
		
		# load the image and show it
		image = cv2.imread(args["image"])
		im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		chans = cv2.split(im_hsv)
		hists = np.zeros([256,3])
		#	calculate image histogram
		for i in range(len(chans)):
			hists[0:hsv[i],i] += cv2.calcHist([chans[i]], [0], None, [hsv[i]], [0, hsv[i]]).flatten()
		cv2.normalize(hists, hists, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
		
		#compare image and average histogram of each color
		colorDists = []
		for hist in histAvg:
			i = 0
			# get Hue histogram
			hist1 = hists[1:hsv[i],i].reshape(-1,1)
			hist2 = hist[1:hsv[i],i].reshape(-1,1)
			# put both on the same measure
			cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
			cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
			# compare
			metric_val = cv2.compareHist(np.float32(hist1),np.float32(hist2),cv2.HISTCMP_BHATTACHARYYA)
			colorDists.append(metric_val)
			
			#plt.plot(hist1, 'g')
			#plt.plot(hist2, 'r')
			#plt.show()
		
		#print(colorDists)
		#print([(1-x)/sum(colorDists) for x in colorDists])
		#print(colors,"\nDistances: ", end =" ")
		#print(', '.join(["{0:0.3f}".format(x) for x in colorDists]))
		print(colors[colorDists.index(min(colorDists))])
		
		
	elif(args["dataset"]):
		# TRAINING
		# make 1 histogram for all images in the given dataset 
		# (input dataset should be path to a folder which contains only images of the same color) 
		hists = np.zeros([256,3])
				
		files = 0
		for filename in os.listdir(args["dataset"]):
			if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
				#print(filename)
				image = cv2.imread(args["dataset"] + filename)
				im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
				chans = cv2.split(im_hsv)
				hist_tmp = np.zeros([256,3])
				for i in range(len(chans)):
					hist_tmp[0:hsv[i],i] = cv2.calcHist([chans[i]], [0], None, [hsv[i]], [0, hsv[i]]).flatten()
					
				cv2.normalize(hist_tmp, hist_tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
				hists += hist_tmp
				files += 1
				
		# average it out
		hists /= files
		
		# get the color name from folder name, and use it as filename
		name = 'histo.txt'
		for color in colors:
			if color in args["dataset"]:
				name = color + '.txt'
				
		#save average histograms for each channel
		np.savetxt(name, hists, fmt='%1.3f')


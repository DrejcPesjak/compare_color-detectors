#https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
#https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
# load the image
imageR = cv2.imread(args["image"])
image = cv2.cvtColor(imageR, cv2.COLOR_BGR2HSV)

# define the list of boundaries
# red, blue, yellow, gray
boundariesBGR = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250])
	#,([103, 86, 65], [145, 133, 128])
]
#yellow, green, black, white, blue, red*2
boundariesHSV = [
	([24,80,20],[32,255,255]),
	([36,25,25],[80,255,255]),
	([0,0,0],[180,255,50]),
	#([0,0,168],[172,111,255]),
	([88,120,25],[133,255,255]),
	([0,100,20],[10,255,255]),
	([160,100,20],[180,255,255])
]


count = []
# loop over the boundaries
for (lower, upper) in boundariesHSV:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	
	count.append(np.sum(mask)/255)
	# show the images
	#outputrgb = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
	#cv2.imshow("images", np.hstack([imageR, outputrgb]))
	#cv2.waitKey(0)
	

count[-2] += count[-1]
count.pop()
#print("yellow, green, black, white, blue, red")
colors = ["yellow", "green", "black", "blue", "red"]
if(sum(count) > 0): #zero division
	probs = [float("{:.2f}".format(i/sum(count))) for i in count]
	print(colors[probs.index(max(probs))])
	#print(colors)
	#print(probs)




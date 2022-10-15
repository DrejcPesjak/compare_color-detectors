# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

#https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()
	
	

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())


# initialize the the features matrix, and labels list
features = []
labels = []
class_names = ['red','blue','green', 'yellow', 'white','black']

if(args["dataset"]):
	#create NEW dataset and save it
	d_path = args["dataset"]
	for name in os.listdir(d_path):
		ix = [color in name for color in class_names]
		#is it a directory and does it contain any of the class names
		if(os.path.isdir(name) and any(ix)):
			label = np.where(ix)[0][0]
			print(label)
			for filename in os.listdir(name):
				if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
					#print(name+'/'+filename)
					image = cv2.imread(name+'/'+filename)
					# extract a color histogram to characterize the color distribution of the pixels in the image
					hist = extract_color_histogram(image)
					# update the features, and labels matricies, respectively
					features.append(hist)
					labels.append(label)
					
	features = np.array(features)
	labels = np.array(labels)
	np.save('features', features)
	np.save('class_labels', labels)
else:
	features = np.load('features.npy')
	labels = np.load('class_labels.npy')

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
	
# train and evaluate a k-NN classifer on the histogram representations
print("building classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)

if(args["image"]):
	#predict class for one image
	image = cv2.imread(args["image"])
	hist = extract_color_histogram(image)
	pred = model.predict([hist])[0]
	print("prediction: ",class_names[pred])
else:
	print("evaluating histogram accuracy...")
	acc = model.score(testFeat, testLabels)
	print("histogram accuracy: {:.2f}%".format(acc * 100))
	#print("knn=", args["neighbors"]," -> acc={:.2f}%".format(acc * 100))
	#print("{:.3f}".format(acc), end=",")



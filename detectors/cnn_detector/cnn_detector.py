import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import cv2


class_names = ['red','blue','green', 'yellow', 'white','black']
size = 128
fname = 'data'
d_path = "."
split = False
all_images = np.empty((0,size,size,3),float)
all_ys = []

ap = argparse.ArgumentParser()
# Available argument combinations:
	# newD  + trainD + eval     > python cnn_detector.py -d path_to-data -m train
	#         loadD  + predict  > python cnn_detector.py -i image_pathname
	# loadD + loadM  + eval		> python cnn_detector.py -m eval
	# newD  + loadM  + eval		> python cnn_detector.py -d path_to-data -m eval
ap.add_argument("-i", "--image", help = "Path to the image")
ap.add_argument("-d", "--dataset", help = "Path to training dataset")
ap.add_argument("-m", "--mode", help = "Modes: train/eval")
args = vars(ap.parse_args())


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
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					re_image = cv2.resize(image, (size,size))
					all_images = np.append(all_images, [re_image/255.0], axis=0)
					all_ys.append(label)

	all_ys = np.array(all_ys)
	np.save(fname, all_images)
	np.save('labels', all_ys)
	split=True
	
elif(not args["image"]):
	#LOAD the dataset
	#if(os.path.isfile(fname+'.npy')):
	all_images = np.load(fname+'.npy')
	all_ys = np.load('labels.npy')
	split=True

if(split):
	#train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
	X_train, X_test, y_train, y_test =  train_test_split(all_images,all_ys ,test_size=0.2, random_state=42, shuffle=True, stratify=all_ys)


if(args["image"] or args["mode"]=="eval"):
	#LOAD cnn model
	model = tf.keras.models.load_model('model')
elif(args["mode"]=="train"):
	#TRAIN new seq. cnn model
	model = models.Sequential()
	model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(size, size, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(6))

	model.summary()

	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
	model.save('model')
	
	#show learning curve (aka. accuracy through epochs)
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()
	
	
if(not args["image"]):
	#EVAL model on given dataset
	test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
	print(test_acc)

else:
	#PREDICTION
	image = cv2.imread(args["image"])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	re_image = cv2.resize(image, (size,size))
	im_pred = np.reshape(re_image/255.0, (1,size,size,3))
	class_prob = model.predict(im_pred, batch_size=1)
	i = np.argmax(class_prob, axis=-1)[0]
	print(class_names[i])
	



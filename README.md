# Introduction

The second homework from perception is to implement and evaluate a reliable colour recognition method.
The goal is to train a classifier to recognize at least six colours: red, green, blue, yellow, white, black.

# Dataset

The dataset of 2900 images was retrieved from the google images site using the automated method described [here] (https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-
using-google-images/).

# Color detectors

We’ve experimented with 5 classifiers, of which 4 were implemented by us.

## kNN classifier by Ahmet Özlü

A k-Nearest Neighbors method on RGB color histograms; the code can be found in this [repo](https://github.com/ahmetozlu/color_recognition).
Accuracy: 69.0%

## Hard-coded color range detector

The predefined color ranges in HSV color space can be seen in this image:
![HSV color ranges](/detectors/color_range_detector/hsvColorRange.png)
Code: [detect_color.py](detectors/color_range_detector/detect_color.py)
Accuracy: 90.17%

## Comparison of histograms

Calculating Bhattacharyya distance between hue histograms of an image and the average histogram for each color.
Code: [hist_detector.py](detectors/hist_detector/hist_detector.py)
Accuracy: 77.0%

## kNN classifier

A kNN classifier comparing 3D color histograms of HSL images. (k=5)
Code: [knn_detector.py](detectors/knn_detector/knn_detector.py)
Accuracy: 80.69%

## Convolutional neural network

A CNN trained on 128x128 RGB images.
Code: [cnn_detector.py](detectors/cnn_detector/cnn_detector.py)
Accuracy: 93.81%


You can find a more detailed report in [homework2.pdf](/report/homework2.pdf).

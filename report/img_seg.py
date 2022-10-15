from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#import cv2
#from scipy import ndimage



#https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/

image = plt.imread('red_heart.jpeg')
print(image.shape)
#plt.imshow(image)
#plt.show()

gray = rgb2gray(image)
#plt.imshow(gray, cmap='gray')
#plt.show()


ix = gray<0.4  # numpy.ndarray of true and falses
plt.imshow(ix, cmap='gray')
plt.show()

im_th = image[ix]
m = np.mean(im_th, axis=0)  # get mean rgb values in thresholded image
print(m)
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(im_th[:,0],im_th[:,1],im_th[:,2])
ax.set_xlabel('Red value')
ax.set_ylabel('Green value')
ax.set_zlabel('Blue value')
plt.show()
'''

#https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
import cv2
im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
im_hsv.shape
imHSVth = im_hsv[ix]
m = np.mean(imHSVth, axis=0)

'''
# HSV picture with segmentation (by thresholding)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(imHSVth[:,0],imHSVth[:,1],imHSVth[:,2])
plt.show()
# HSV double thresholded
inx = imHSVth[:,0] < m[0]-5
betterHSV = imHSVth[inx]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(betterHSV[:,0], betterHSV[:,1], betterHSV[:,2])
plt.show()

#two blobs on the graph of HSV image, one item and other background 
I = im_hsv.reshape([378*672,3])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(I[:,0], I[:,1], I[:,2])
plt.show()
'''

# segmentation by HSV color
ixes = im_hsv[:,:,0]<20
plt.imshow(ixes, cmap='gray')
plt.show()

# cut out red object
iii = image[:,:,0] * ~ixes
jjj = image[:,:,1] * ~ixes
kkk = image[:,:,2] * ~ixes
newImage = cv2.merge((iii,jjj,kkk))
plt.imshow(newImage)
plt.show()





#OTSU - thresholding on grayscale image
from skimage.filters import threshold_otsu
th = threshold_otsu(gray)
gray_th = gray<=th
plt.imshow(gray_th, cmap='gray')
plt.show()


#OTSU - on Hue  in HSV image
th = threshold_otsu(im_hsv[:,:,0])
ixes = im_hsv[:,:,0]<th
plt.imshow(ixes,cmap='gray')
plt.show()


#HSV color histogram, using Hue
image = plt.imread('red_heart.jpeg')
im_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
plt.hist(im_hsv[:,:,0].ravel(), bins=180)
plt.show()


plt.hist(gray.ravel(), bins=255)
plt.show()




# This module is used for Data Preprocessing
# In this module we use OPENCV to apply LBP and face detection to each image.
# Here we have totally 2000 photos, 500 for everyone.

try:
    import os
except ImportError:
    import OS as os

from os import listdir
from os.path import isfile, join
import numpy

try:
    import cv2
except ImportError:
    print'Please install OPENCV to proceed.'
    exit(0)

# the file of OPENCV for face detection
path = '/Users/abhi/code/lib/python2.7/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')

# Two functions for LBP
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx, idy]
    except IndexError:
        return default

# Apply LBP and face detection to all photos and save the new photos
for i in range(1,5):# that means i = 1,2,3,4
	if i == 1:
	    mypath = '/Users/abhi/Desktop/ANN/changedphoto/abhi/'
	elif i == 2:
	    mypath = '/Users/abhi/Desktop/ANN/changedphoto/adi/'
	elif i == 3:
	    mypath = '/Users/abhi/Desktop/ANN/changedphoto/charu/'
	elif i == 4:
	    mypath = '/Users/abhi/Desktop/ANN/changedphoto/kru/'

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	images = numpy.empty(len(onlyfiles), dtype=object)
	for n in range(0, len(onlyfiles)):
	    images[n] = cv2.imread(join(mypath, onlyfiles[n]),0)
	    # read the image in grayscale
	    newgray = images[n]
	    # apply face detection to this image
	    faces = face_cascade.detectMultiScale(
		newgray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	    )
# The cropped image is still in size of w*h, we need an image in 48*48, so change it
	    for (x, y, w, h) in faces:
	        x = x
	        cropped = newgray[y:y + h, x:x + w]  
	        result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)

	    # Apply LBP to this image
	    # copy result as transformed_img
	    transformed_img = cv2.copyMakeBorder(result, 0, 0, 0, 0, cv2.BORDER_REPLICATE)

	    for x in range(0, len(result)):
		for y in range(0, len(result[0])):
		    center = result[x, y]
		    top_left = get_pixel_else_0(result, x - 1, y - 1)
		    top_up = get_pixel_else_0(result, x, y - 1)
		    top_right = get_pixel_else_0(result, x + 1, y - 1)
		    right = get_pixel_else_0(result, x + 1, y)
		    left = get_pixel_else_0(result, x - 1, y)
		    bottom_left = get_pixel_else_0(result, x - 1, y + 1)
		    bottom_right = get_pixel_else_0(result, x + 1, y + 1)
		    bottom_down = get_pixel_else_0(result, x, y + 1)

		    values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
		                                  bottom_down, bottom_left, left])

		    weights = [1, 2, 4, 8, 16, 32, 64, 128]
		    res = 0
		    for a in range(0, len(values)):
		        res += weights[a] * values[a]

		    transformed_img.itemset((x, y), res)

	    # we only use the part (1,1) to (46,46) of the result img.
	    # original img: 0-47, after resize: 1-46
	    lbp = transformed_img[1:47, 1:47]  # here 1 included, 47 not included

	    # save the final image
	    if i == 1:
		name = './CNNdata/1_abhi_' + str(n) + '.png'
	    elif i == 2:
		name = './CNNdata/2_adi_' + str(n) + '.png'
	    elif i == 3:
		name = './CNNdata/3_charu_' + str(n) + '.png'
	    elif i == 4:
		name = './CNNdata/4_kru_' + str(n) + '.png'

	    cv2.imwrite(name, lbp)



# Module for testing the model using some existing photos

from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils

import cv2
import numpy as np

from skimage import io
from sklearn.cross_validation import train_test_split

import os
from os import listdir
from os.path import isfile, join


# Functions for LBP
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

# Using OPENCV for face detection
path = '/Users/abhi/code/lib/python2.7/site-packages/cv2/data/'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_default.xml')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# load the photos paths firstly
DatasetPath = []
for i in os.listdir('./testPhotos/'):
    DatasetPath.append(os.path.join('./testPhotos/', i))

imageData = []
imageName = []

# Read the photos, apply face detection and LBP to test
for i in DatasetPath:
    imgRead = cv2.imread(i,0)
    imageName.append(str(i))
    faces = face_cascade.detectMultiScale(
        imgRead,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        x=x
        cropped = imgRead[y:y + h, x:x + w]
        result = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_LINEAR)  # OPENCV 3.x
    
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
    lbp = transformed_img[1:47, 1:47]

    imageData.append(lbp)

# Apply the model to the test photos
# Print the result on the original photo
for i in range(0,len(imageData)):
	c = np.array(imageData[i])
	c = np.array(c)
	c = c.reshape(1, 46, 46, 1)
	c = c.astype('float32')
	c /= 255

	predictions = loaded_model.predict(c)
	img = cv2.imread(imageName[i],1)
	faces = face_cascade.detectMultiScale(
	    img,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
	)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	if predictions[0][0] > 0.93:
	    cv2.putText(img,'this is Abhi',(x,y+h+30),font,1,(255, 0, 0),2)
	elif predictions[0][1] > 0.93:
	    cv2.putText(img,'this is Hampi',(x,y+h+30),font,1,(255, 0, 0),2)
	elif predictions[0][2] > 0.93:
	    cv2.putText(img,'this is Charu',(x,y+h+30),font,1,(255, 0, 0),2)
	elif predictions[0][3] > 0.93:
	    cv2.putText(img,'this is Kru',(x,y+h+30),font,1,(255, 0, 0),2)
	else:
	    cv2.putText(img,'can\'t be recognized',(x,y+h+30),font,1,(255, 0, 0),2)
	cv2.imshow('img',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

